// Thread helpers with a large stack for Windows compile passes.
// Kept in a .cpp (not the public header) because a header self-include test
// forbids windows.h in public headers.
#include <migraphx/config.hpp>
#include <migraphx/simple_par_for.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace detail {

static DWORD WINAPI thread_proc(LPVOID arg)
{
    auto* fp = static_cast<std::function<void()>*>(arg);
    (*fp)();
    delete fp;
    return 0;
}

MIGRAPHX_EXPORT void* create_thread_64mb(void* callable)
{
    static const SIZE_T STACK = 64ULL * 1024 * 1024; // 64 MB
    HANDLE h = CreateThread(
        nullptr, STACK, thread_proc, callable, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    return static_cast<void*>(h); // nullptr on failure
}

MIGRAPHX_EXPORT void join_thread(void* handle)
{
    HANDLE h = static_cast<HANDLE>(handle);
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);
}

} // namespace detail
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif