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
#ifndef ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP

#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    namespace RegisterTransform_impl
    {
        using LayoutTraits_impl::matrix_layout_traits;
        using LayoutTraits_impl::register_layout_traits;

// Keeps things a bit more tidy. Quick access to register layout traits.
#define reg_traits_lhs register_layout_traits<RegisterLayoutLhs>
#define reg_traits_rhs register_layout_traits<RegisterLayoutRhs>

// Quick access to matrix layout traits, that are embedded in the register layout traits.
#define mat_traits_lhs matrix_layout_traits<typename reg_traits_lhs::MatrixLayout>
#define mat_traits_rhs matrix_layout_traits<typename reg_traits_rhs::MatrixLayout>

        // Note: If you arrive at an undefined register_transform error, it is likely
        // the layout transformation is not currently supported. Need to either implement
        // the transform or ensure your layout transform mapping is correct.
        template <typename RegisterLayoutSrc, typename RegisterLayoutDst, typename Enabler = void>
        struct register_layout_transform;

        // No-op transform (same-layout):
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<is_layout_same_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // No-op
                return v;
            }
        };

        // AOS -> SOA transform (non-interleaved) requirements:
        // - Lhs is *Inline
        // - layouts are not interleaved
        // - layouts are orthogonal
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<(mat_traits_lhs::is_col_inline || mat_traits_lhs::is_row_inline)
                        && !mat_traits_lhs::is_interleaved
                        && is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforma::AosToSoa<mat_traits_lhs::BlockDim,
                                            mat_traits_lhs::MaxVectorWidth>::exec(forward<VecT>(v));
            }
        };

        // SOA -> AOS transform (non-interleaved) requirements:
        // - Lhs is *Ortho
        // - layouts are not interleaved
        // - layouts are orthogonal
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<(mat_traits_lhs::is_col_ortho || mat_traits_lhs::is_row_ortho)
                        && !mat_traits_lhs::is_interleaved
                        && is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::SoaToAos<mat_traits_lhs::BlockDim,
                                            mat_traits_lhs::MaxVectorWidth>::exec(forward<VecT>(v));
            }
        };

        // Interleaved layout transform:
        // - layouts are interleaved
        // - layouts are orthogonal
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<mat_traits_lhs::is_interleaved
                        && is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // TODO: replace with DimPerThread for interleaved.
                return interleave<mat_traits_lhs::BlockDim>(forward<VecT>(v));
            }
        };

    } // namespace RegisterTransform_impl

} // namespace rocWMMA

#endif // ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
