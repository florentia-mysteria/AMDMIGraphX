/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef MIGRAPHX_GUARD_RTGLIB_SIMPLE_PAR_FOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_SIMPLE_PAR_FOR_HPP

#include <thread>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>
#include <functional>
#include <system_error>
#include <migraphx/export.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// On Windows, std::thread uses a 1 MB stack by default.  MIGraphX compile
// passes contain recursive DFS traversals (fix() combinator) over deep model
// graphs (ViT / InfiniDepth).  At 512-px resolution these exhaust the 1 MB
// stack causing 0xC00000FD.  Thread creation lives in src/par_for.cpp (a
// header self-include test forbids windows.h in public headers).
#ifdef _WIN32
namespace detail {
MIGRAPHX_EXPORT void* create_thread_64mb(void* callable);
MIGRAPHX_EXPORT void join_thread(void* handle);
} // namespace detail

struct joinable_thread
{
    void* handle = nullptr;
    std::function<void()> fn;

    joinable_thread() = default;

    template <class F>
    joinable_thread(F&& f) : fn(std::forward<F>(f)) // NOLINT
    {
        auto* p = new std::function<void()>(fn);
        handle  = detail::create_thread_64mb(static_cast<void*>(p));
        if(handle == nullptr)
        {
            delete p;
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
                                    "create_thread_64mb failed");
        }
    }

    joinable_thread& operator=(joinable_thread&& other) noexcept
    {
        if(this != &other)
        {
            handle = other.handle;
            other.handle = nullptr;
            fn = std::move(other.fn);
        }
        return *this;
    }
    joinable_thread(joinable_thread&& other) noexcept
        : handle(other.handle), fn(std::move(other.fn))
    {
        other.handle = nullptr;
    }
    joinable_thread(const joinable_thread&) = delete;
    joinable_thread& operator=(const joinable_thread&) = delete;

    bool joinable() const noexcept { return handle != nullptr; }

    void join()
    {
        if(handle != nullptr)
        {
            detail::join_thread(handle);
            handle = nullptr;
        }
    }

    ~joinable_thread() { join(); }
};
#else
struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) // NOLINT
    {
    }

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};
#endif

template <class F>
auto thread_invoke(std::size_t i, std::size_t tid, F f) -> decltype(f(i, tid))
{
    f(i, tid);
}

template <class F>
auto thread_invoke(std::size_t i, std::size_t, F f) -> decltype(f(i))
{
    f(i);
}

template <class F>
void simple_par_for_impl(std::size_t n, std::size_t threadsize, F f)
{
    if(threadsize <= 1)
    {
#ifdef _WIN32
        // On Windows the current-thread stack may only be 1 MB (Python default).
        // Run the serial loop on a 64 MB joinable_thread to prevent 0xC00000FD
        // from deep recursive graph passes in MIGraphX compile.
        joinable_thread t([&] {
            for(std::size_t i = 0; i < n; i++)
                thread_invoke(i, 0, f);
        });
        t.join();
#else
        for(std::size_t i = 0; i < n; i++)
            thread_invoke(i, 0, f);
#endif
    }
    else
    {
        std::vector<joinable_thread> threads(threadsize);
// Using const here causes gcc 5 to ICE
#if(!defined(__GNUC__) || __GNUC__ != 5)
        const
#endif
            std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());

        std::size_t work = 0;
        std::size_t tid  = 0;
        std::generate(threads.begin(), threads.end(), [=, &work, &tid] {
            auto result = joinable_thread([=] {
                std::size_t start = work;
                std::size_t last  = std::min(n, work + grainsize);
                for(std::size_t i = start; i < last; i++)
                {
                    thread_invoke(i, tid, f);
                }
            });
            work += grainsize;
            ++tid;
            return result;
        });
        assert(work >= n);
    }
}

template <class F>
void simple_par_for(std::size_t n, std::size_t min_grain, F f)
{
    const auto threadsize = std::min<std::size_t>(std::thread::hardware_concurrency(),
                                                  n / std::max<std::size_t>(1, min_grain));
    simple_par_for_impl(n, threadsize, f);
}

template <class F>
void simple_par_for(std::size_t n, F f)
{
    const int min_grain = 8;
    simple_par_for(n, min_grain, f);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
