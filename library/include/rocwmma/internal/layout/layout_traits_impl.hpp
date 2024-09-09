/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_LAYOUT_TRAITS_IMPL_HPP
#define ROCWMMA_LAYOUT_TRAITS_IMPL_HPP

#include "config.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    // Common helpers for supported traits
    namespace detail
    {
        // Based on the current config, determine the compatibility of the mma dimension
        constexpr static inline bool testSupportedMmaDim(uint32_t testDim)
        {
            return ((bool)ROCWMMA_BLOCK_DIM_16_SUPPORTED && testDim == 16u)
                   || ((bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED && (testDim == 16u || testDim == 32u));
        }

        // VW can be changed from vw0 to vw1 as long as they have the same maxVW, and that maxVW
        // is a multiple of both vw values
        constexpr static inline bool testSupportedVW(uint32_t maxVW, uint32_t vw0, uint32_t vw1)
        {
            return (vw0 <= maxVW) && (vw1 <= maxVW) && (maxVW % vw0 == 0) && (maxVW % vw1 == 0);
        }

    } // namespace detail

    // Covers all other generic exact layout class matches

    // Self-compare is always true
    template <typename Layout>
    struct is_layout_same<Layout, Layout> : public true_type
    {
    };

    // Self-compare is always false
    template <typename Layout>
    struct is_layout_transpose<Layout, Layout> : public false_type
    {
    };

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_TRAITS_IMPL_HPP
