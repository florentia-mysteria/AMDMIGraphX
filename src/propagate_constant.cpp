/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/propagate_constant.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/simple_par_for.hpp>
#include <migraphx/env.hpp>
#include <migraphx/make_op.hpp>
#include <cstring>
#include <numeric>
#include <thread>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_PROPAGATE_CONSTANT)

static bool skip_propagate(instruction_ref ins)
{
    if(contains({"contiguous", "dequantizelinear", "reshape"}, ins->name()))
        return skip_propagate(ins->inputs().front());
    if(contains({"unpack_int4", "unpack_fp4"}, ins->name()))
        return true;
    auto&& s = ins->get_shape();
    if(s.broadcasted() and s.element_space() < s.elements())
        return true;
    // Do not fold an op that only expands small (e.g. broadcast) inputs, such as
    // convert(multibroadcast(bias)): the literal would be activation-sized
    // (hundreds of MB each) and is cheap to recompute inside the fused kernel.
    if(not ins->inputs().empty() and s.elements() > 1024 * 1024)
    {
        std::size_t input_space = 0;
        for(auto input : ins->inputs())
            input_space += input->get_shape().element_space();
        if(s.elements() > 2 * input_space)
            return true;
    }
    auto aliases = instruction::get_output_alias(ins, true);
    if(aliases.size() == 1 and aliases.front() != ins)
        return skip_propagate(aliases.front());
    if(ins->is_undefined())
        return true;
    return false;
}

static bool is_const_ins(instruction_ref ins, const std::unordered_set<std::string>& skip_ops)
{
    return ins->can_eval() and not skip_propagate(ins) and
           skip_ops.find(ins->name()) == skip_ops.end();
}

static literal as_packed(const argument& c)
{
    if(c.get_shape().packed())
        return {c.get_shape(), c.data()};
    auto s = c.get_shape().with_lens(c.get_shape().lens());
    literal result;
    c.visit([&](auto x) { result = literal{s, x.begin(), x.end()}; });
    return result;
}

// A folded constant can be a broadcast in disguise, e.g. dot(broadcast(beta), W) after
// simplify_algebra distributes a LayerNorm bias over a GEMM: every row is identical.
// Keep one slice along each axis whose slices are all equal, so the literal stays
// small and is re-broadcast instead of storing hundreds of MB per instance.
static literal compress_repeated_axes(const literal& lit)
{
    const auto& s = lit.get_shape();
    if(not s.standard() or s.bytes() < 16 * 1024 * 1024)
        return lit;
    auto lens             = s.lens();
    const std::size_t esz = s.type_size();
    std::vector<char> data(lit.data(), lit.data() + s.bytes());
    bool changed = false;
    for(std::size_t d = 0; d < lens.size(); d++)
    {
        if(lens[d] < 2)
            continue;
        std::size_t outer = std::accumulate(
            lens.begin(), lens.begin() + d, std::size_t{1}, std::multiplies<>{});
        std::size_t inner = std::accumulate(lens.begin() + d + 1,
                                            lens.end(),
                                            std::size_t{1},
                                            std::multiplies<>{}) *
                            esz;
        bool same = true;
        for(std::size_t o = 0; o < outer and same; o++)
        {
            const char* base = data.data() + o * lens[d] * inner;
            for(std::size_t k = 1; k < lens[d] and same; k++)
                same = std::memcmp(base, base + k * inner, inner) == 0;
        }
        if(not same)
            continue;
        std::vector<char> next(outer * inner);
        for(std::size_t o = 0; o < outer; o++)
            std::memcpy(next.data() + o * inner, data.data() + o * lens[d] * inner, inner);
        data.swap(next);
        lens[d] = 1;
        changed = true;
    }
    if(not changed)
        return lit;
    return literal{shape{s.type(), lens}, data.data()};
}

void propagate_constant::apply(module& m) const
{
    std::unordered_set<instruction_ref> const_instrs;
    auto last = std::prev(m.end());

    // Find instructions that can be evaluated to a literal
    for(auto i : iterator_for(m))
    {
        const bool is_const = is_const_ins(i, skip_ops);
        if(is_const and i != last)
            continue;

        if(i == last and is_const)
        {
            const_instrs.insert(i);
        }
        else
        {
            std::copy_if(i->inputs().begin(),
                         i->inputs().end(),
                         std::inserter(const_instrs, const_instrs.begin()),
                         [&](const instruction_ref ins) {
                             return is_const_ins(ins, skip_ops) and ins->name() != "@literal";
                         });
        }
    }

    // Compute literals in parallel
    std::vector<instruction_ref> const_instrs_vec{const_instrs.begin(), const_instrs.end()};
    std::vector<literal> literals(const_instrs_vec.size());
    std::size_t grainsize = 1;
#ifdef _WIN32
    grainsize = std::max<std::size_t>(
        const_instrs_vec.size() / (std::thread::hardware_concurrency() / 2), 1);
#else
#if !MIGRAPHX_HAS_EXECUTORS
    std::size_t n = std::max<std::size_t>(2048 / std::thread::hardware_concurrency(), 1);
    grainsize     = const_instrs_vec.size() / n;
#endif
#endif
    simple_par_for(const_instrs_vec.size(), grainsize, [&](const auto i) {
        literals[i] = as_packed(const_instrs_vec[i]->eval());
    });

    // Replace instructions in m
    for(size_t i = 0; i < const_instrs_vec.size(); i++)
    {
        if(not literals[i].empty())
        {
            if(enabled(MIGRAPHX_TRACE_PROPAGATE_CONSTANT{}))
            {
                std::cout << "Constant replace: " << std::endl;
                std::vector<instruction_ref> inss;
                fix([&](auto self, auto ins) {
                    if(contains(inss, ins))
                        return;
                    for(auto input : ins->inputs())
                        self(input);
                    inss.push_back(ins);
                })(const_instrs_vec[i]);
                m.debug_print(inss);
            }
            assert(literals[i].get_shape().lens() == const_instrs_vec[i]->get_shape().lens());
            assert(literals[i].get_shape().bytes() <= const_instrs_vec[i]->get_shape().bytes());
            auto lens = literals[i].get_shape().lens();
            auto small = compress_repeated_axes(literals[i]);
            auto l     = m.add_literal(small);
            if(small.get_shape().lens() != lens)
                l = m.insert_instruction(
                    std::next(l), make_op("multibroadcast", {{"out_lens", lens}}), l);
            m.replace_instruction(const_instrs_vec[i], l);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
