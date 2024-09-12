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
#ifndef ROCWMMA_MATRIX_LAYOUT_TRAITS_HPP
#define ROCWMMA_MATRIX_LAYOUT_TRAITS_HPP

#include "config.hpp"
#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    // Common helpers for supported traits
    namespace LayoutTraits_impl
    {
        // Based on the current config, determine the compatibility of the mma dimension
        constexpr static inline bool testSupportedMmaDim(uint32_t MmaDim)
        {
            return ((bool)ROCWMMA_BLOCK_DIM_16_SUPPORTED && MmaDim == 16u)
                   || ((bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED && (MmaDim == 16u || MmaDim == 32u));
        }

        // VW can be changed from vw0 to vw1 as long as they have the same maxVW, and that maxVW
        // is a multiple of both vw values
        constexpr static inline bool testSupportedVW(uint32_t maxVW, uint32_t vw0, uint32_t vw1)
        {
            return (vw0 <= maxVW) && (vw1 <= maxVW) && (maxVW % vw0 == 0) && (maxVW % vw1 == 0);
        }

        // Reference regular layouts
        using MatrixLayout::ColInlineVW;
        using MatrixLayout::ColOrthoVW;
        using MatrixLayout::RowInlineVW;
        using MatrixLayout::RowOrthoVW;

        // Reference interleaved layouts
        using MatrixLayout::ColInlineInt;
        using MatrixLayout::ColOrthoInt;
        using MatrixLayout::RowInlineInt;
        using MatrixLayout::RowOrthoInt;

        // NOTE: MatrixLayout assumptions
        // When determining MatrixLayout traits, there are several strong assumptions.
        // 1. Regarding same-ness: MatrixLayouts must match, as defined below:
        //  ____________________________________________________________________
        // | MatrixLayoutLhs | MatrixLayoutRhs | Compatibility test:            |
        // |                 |     (Same)      | Required Fixed Params          |
        // | ------------------------------------------------------------------ |
        // | ColOrthoVW      | ColOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | ColInlineVW     | ColInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | RowOrthoVW      | RowOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | RowInlineVW     | RowInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | ------------------------------------------------------------------ |
        // | ColOrthoInt     | ColOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | ColInlineInt    | ColInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        // | RowOrthoInt     | RowOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | RowInlineInt    | RowInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        //  --------------------------------------------------------------------
        //
        // 2. Regarding orthogonality: for all Col* layouts, their Row*
        // orthogonal counterparts are implemented by row / col coordinate swaps.
        // This is valid as long as we have some fixed parameters, as defined below:
        //  ____________________________________________________________________
        // | MatrixLayoutLhs | MatrixLayoutRhs | Compatibility test:            |
        // |                 |   (Orthogonal)  | Required Fixed Params          |
        // | ------------------------------------------------------------------ |
        // | ColOrthoVW      | RowOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | ColInlineVW     | RowInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | RowOrthoVW      | ColOrthoVW      | BlockDim, KDim, MaxVectorWidth |
        // | RowInlineVW     | ColInlineVW     | BlockDim, KDim, MaxVectorWidth |
        // | ------------------------------------------------------------------ |
        // | ColOrthoInt     | RowOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | ColInlineInt    | RowInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        // | RowOrthoInt     | ColOrthoInt     | BlockDim, KDim, MmaDim, SplitK |
        // | RowInlineInt    | ColInlineInt    | BlockDim, KDim, MmaDim, SplitK |
        //  --------------------------------------------------------------------
        // This defines the need for MatrixLayout classifiers based upon:
        // - ColOrtho / RowOrtho
        // - ColInline / RowInline
        // - Non-interleave / non-interleaved
        //
        // Following the above traits, we can build more complicated traits such as
        // is_same, is_orthogonal and orthogonal_layout.

        // Classifier for ColOrtho MatrixLayout
        template <typename MatrixLayout>
        struct is_col_ortho : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_col_ortho<ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_col_ortho<ColOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        // Classifier for RowOrtho MatrixLayout
        template <typename MatrixLayout>
        struct is_row_ortho : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_row_ortho<RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_row_ortho<RowOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        // Classifier for ColInline MatrixLayout
        template <typename MatrixLayout>
        struct is_col_inline : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_col_inline<ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_col_Inline<ColInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        // Classifier for RowInline MatrixLayout
        template <typename MatrixLayout>
        struct is_row_inline : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct is_row_inline<RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_row_inline<RowInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        // Classifier for interleaved layout
        template <typename MatrixLayout>
        struct is_interleaved : public false_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<ColOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<RowOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<ColInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim,
                  uint32_t SplitK>
        struct is_interleaved<RowInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>
            : public true_type
        {
        };

        // Convenience evaluators
        template <typename MatrixLayout>
        constexpr static bool is_col_ortho_v = is_col_ortho<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_row_ortho_v = is_row_ortho<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_col_inline_v = is_col_inline<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_row_inline_v = is_row_inline<MatrixLayout>::value;

        template <typename MatrixLayout>
        constexpr static bool is_interleaved_v = is_interleaved<MatrixLayout>::value;

        // When comparing one MatrixLayout to another, we need a way to check parameter compatibility.
        template <typename LayoutLhs, typename LayoutRhs, typename Enabler = void>
        struct is_compatible_params : public false_type
        {
        };

        // Non-interleaved matrix layouts require that BlockDim, BlockK, MaxVW are matching.
        // VectorWidth values must satisfy criterion in testSupportedVW().
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidthLhs,
                  uint32_t VectorWidthRhs,
                  uint32_t MaxVectorWidth,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayoutLhs,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayoutRhs>
        struct is_compatible_params<
            MatrixLayoutLhs<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth>,
            MatrixLayoutRhs<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth>,
            enable_if_t<
                !is_interleaved_v<
                    MatrixLayoutLhs<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth>>
                && !is_interleaved_v<
                    MatrixLayoutRhs<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth>>
                && testSupportedVW(MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>>
            : public true_type
        {
        };

        // Interleaved matrix layouts require that BlockDim, BlockK, MmaDim, SplitK are matching.
        // MmaDim values must satisfy criterion in testSupportedMmaDim().
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayoutLhs,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayoutRhs>
        struct is_compatible_params<
            MatrixLayoutLhs<BlockDim, BlockK, DataT, MmaDim, SplitK>,
            MatrixLayoutRhs<BlockDim, BlockK, DataT, MmaDim, SplitK>,
            enable_if_t<
                is_interleaved_v<MatrixLayoutLhs<BlockDim, BlockK, DataT, MmaDim, SplitK>>
                && is_interleaved_v<MatrixLayoutRhs<BlockDim, BlockK, DataT, MmaDim, SplitK>>
                && testSupportedMmaDim(MmaDim)>> : public true_type
        {
        };

        // Convenience evaluator
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        constexpr static bool is_compatible_params_v
            = is_compatible_params<MatrixLayoutLhs, MatrixLayoutRhs>::value;

        // Classifier to test same-ness, implements criterion #1 from above:
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        struct is_layout_same<
            MatrixLayoutLhs,
            MatrixLayoutRhs,
            enable_if_t<((is_col_ortho_v<MatrixLayoutLhs> && is_col_ortho_v<MatrixLayoutRhs>)
                         || (is_row_ortho_v<MatrixLayoutLhs> && is_row_ortho_v<MatrixLayoutRhs>)
                         || (is_col_inline_v<MatrixLayoutLhs> && is_col_inline_v<MatrixLayoutRhs>)
                         || (is_row_inline_v<MatrixLayoutLhs> && is_row_inline_v<MatrixLayoutRhs>))
                        && is_compatible_params_v<MatrixLayoutLhs, MatrixLayoutRhs>>>
            : public true_type
        {
        };

        // Classifier to test orthogonality, implements criterion #2 from above:
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        struct is_layout_orthogonal<
            MatrixLayoutLhs,
            MatrixLayoutRhs,
            enable_if_t<((is_col_ortho_v<MatrixLayoutLhs> && is_row_ortho_v<MatrixLayoutRhs>)
                         || (is_row_ortho_v<MatrixLayoutLhs> && is_col_ortho_v<MatrixLayoutRhs>)
                         || (is_col_inline_v<MatrixLayoutLhs> && is_row_inline_v<MatrixLayoutRhs>)
                         || (is_row_inline_v<MatrixLayoutLhs> && is_col_inline_v<MatrixLayoutRhs>))
                        && is_compatible_params_v<MatrixLayoutLhs, MatrixLayoutRhs>>>
            : public true_type
        {
        };

        // Matrix space transpose guide: Swap rows / cols
        // VW stays consistent.
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct orthogonal_layout<RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
        {
            using type = ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;
        };

        // Orthogonal guide for interleaved layouts
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<RowOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = ColOrthoInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaDim,
                  uint32_t SplitK>
        struct orthogonal_layout<RowInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>>
        {
            using type = ColInlineInt<BlockDim, BlockK, DataT, MmaDim, SplitK>;
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_MATRIX_LAYOUT_TRAITS_HPP
