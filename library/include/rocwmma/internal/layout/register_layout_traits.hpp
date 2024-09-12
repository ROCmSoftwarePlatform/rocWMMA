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
#ifndef ROCWMMA_REGISTER_LAYOUT_TRAITS_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRAITS_HPP

#include "layout.hpp"
#include "layout_traits.hpp"
#include "utility/type_traits.hpp"

namespace rocwmma
{
    namespace LayoutTraits_impl
    {

        // NOTE: RegisterLayout assumptions
        // When determining RegisterLayout traits, there are several strong assumptions.
        // 1. Regarding same-ness:
        //    - Storage<MatrixLayout> match if MatrixLayouts match, given fixed params.
        //    - Storage<MatrixLayout> match if MatrixLayouts are either both *Ortho or both *Inline
        //      orientations. Register thread mapping is the same while swapping the underlying
        //      meaning of rows for cols (e.g., implicit transpose).
        //    - Storage<*Ortho> layouts are suitable MmaInputs while Storage<*Inline> layouts are not.
        //      Given appropriate MmaDim, it is assumed MmaInput layouts are mapped to mma hardware
        //      requirements.
        //  _________________________________________________________________________________
        // | MatrixLayoutLhs       |     MatrixLayoutRhs    |    Compatibility test:         |
        // |                       |         (Same)         |  Required Fixed Params         |
        // | ------------------------------------------------------------------------------- |
        // | Storage<ColOrthoVW>   | Storage<ColOrthoVW>    | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColInlineVW>  | Storage<ColInlineVW>   | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowOrthoVW>   | Storage<RowOrthoVW>    | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowInlineVW>  | Storage<RowInlineVW>   | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColOrthoVW>   | Storage<RowOrthoVW>    | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColInlineVW>  | Storage<RowInlineVW>   | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowOrthoVW>   | Storage<ColOrthoVW>    | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowInlineVW>  | Storage<ColInlineVW>   | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColOrthoVW>   | MmaInput               | BlockDim == MmaDim             |
        // | MmaInput              | Storage<ColOrthoVW>    | BlockDim == MmaDim             |
        // | Storage<RowOrthoVW>   | MmaInput               | BlockDim == MmaDim             |
        // | MmaInput              | Storage<RowOrthoVW>    | BlockDim == MmaDim             |
        // | ------------------------------------------------------------------------------- |
        // | Storage<ColInlineInt> | Storage<ColInlineInt>  | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowOrthoInt>  | Storage<RowOrthoInt>   | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColOrthoInt>  | Storage<ColOrthoInt>   | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowInlineInt> | Storage<RowInlineInt>  | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColOrthoInt>  | Storage<RowOrthoInt>   | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColInlineInt> | Storage<RowInlineInt>  | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowOrthoInt>  | Storage<ColOrthoInt>   | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowInlineInt> | Storage<ColInlineInt>  | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColOrthoInt>  | MmaInput               | MmaDim                         |
        // | MmaInput              | Storage<ColOrthoInt>   | MmaDim                         |
        // | Storage<RowOrthoInt>  | MmaInput               | MmaDim                         |
        // | MmaInput              | Storage<RowOrthoInt>   | MmaDim                         |
        // | ------------------------------------------------------------------------------- |
        //
        // 2. Regarding orthogonality:
        //    - Storage<MatrixLayout>s are considered orthogonal if one MatrixLayout is an
        //      *Ortho layout and the other is an *Inline layout, or vice versa.
        //    - Since MmaInput layouts are same as Storage<Ortho*> layouts with appropriate
        //      MmaDim, MmaInput is also orthogonal to Storage<Inline*> layouts.
        //  _______________________________________________________________________________
        // | MatrixLayoutLhs       | MatrixLayoutRhs      | Required Fixed Params          |
        // |                       |   (Transposed)       |                                |
        // | ----------------------------------------------------------------------------- |
        // | Storage<ColOrthoVW>   | Storage<ColInlineVW> | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColInlineVW>  | Storage<ColOrthoVW>  | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowOrthoVW>   | Storage<RowInlineVW> | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowInlineVW>  | Storage<RowOrthoVW>  | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColOrthoVW>   | Storage<RowInlineVW> | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowInlineVW>  | Storage<ColOrthoVW>  | BlockDim, KDim, MaxVectorWidth |
        // | Storage<RowOrthoVW>   | Storage<ColInlineVW> | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColInlineVW>  | Storage<RowOrthoVW>  | BlockDim, KDim, MaxVectorWidth |
        // | Storage<ColInlineVW>  | MmaInput             | BlockDim == MmaDim             |
        // | MmaInput              | Storage<ColInlineVW> | BlockDim == MmaDim             |
        // | Storage<RowInlineVW>  | MmaInput             | BlockDim == MmaDim             |
        // | MmaInput              | Storage<RowInlineVW> | BlockDim == MmaDim             |
        // | ----------------------------------------------------------------------------- |
        // | Storage<ColOrthoInt>  | Storage<ColInlineInt>| BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColInlineInt> | Storage<ColOrthoInt> | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowOrthoInt>  | Storage<RowInlineInt>| BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowInlineInt> | Storage<RowOrthoInt> | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColOrthoInt>  | Storage<RowInlineInt>| BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowInlineInt> | Storage<ColOrthoInt> | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<RowOrthoInt>  | Storage<ColInlineInt>| BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColInlineInt> | Storage<RowOrthoInt> | BlockDim, KDim, MmaDim, SplitK |
        // | Storage<ColInlineInt> | MmaInput             | MmaDim                         |
        // | MmaInput              | Storage<ColInlineInt>| MmaDim                         |
        // | Storage<RowInlineInt> | MmaInput             | MmaDim                         |
        // | MmaInput              | Storage<RowInlineInt>| MmaDim                         |
        // | ----------------------------------------------------------------------------- |

        using RegisterLayout::MmaAcc;
        using RegisterLayout::MmaInput;
        using RegisterLayout::Storage;

        // Classifier for storage of col ortho
        template <typename RegisterLayout, typename Enabler = void>
        struct is_storage_col_ortho : public false_type
        {
        };

        template <typename MatrixLayout>
        struct is_storage_col_ortho<Storage<MatrixLayout> enable_if_t<is_col_ortho_v<MatrixLayout>>>
            : public true_type
        {
        };

        // Classifier for storage of row ortho
        template <typename RegisterLayout, typename Enabler = void>
        struct is_storage_row_ortho : public false_type
        {
        };

        template <typename MatrixLayout>
        struct is_storage_row_ortho<Storage<MatrixLayout>,
                                    enable_if_t<is_row_ortho_v<MatrixLayout>>> : public true_type
        {
        };

        // Classifier for storage of col inline
        template <typename RegisterLayout, typename Enabler = void>
        struct is_storage_col_inline : public false_type
        {
        };

        template <typename MatrixLayout>
        struct is_storage_col_inline<Storage<MatrixLayout>,
                                     enable_if_t<is_col_inline_v<MatrixLayout>>> : public true_type
        {
        };

        // Classifier for storage of row inline
        template <typename RegisterLayout, typename Enabler = void>
        struct is_storage_row_inline : public false_type
        {
        };

        template <typename MatrixLayout>
        struct is_storage_row_inline<Storage<MatrixLayout>,
                                     enable_if_t<is_row_inline_v<MatrixLayout>>> : public true_type
        {
        };

        // Classifier for mma inputs
        template <typename RegisterLayout>
        struct is_mma_input : public false_type
        {
        };

        template <uint32_t MmaSize>
        struct is_mma_input<MmaInput<MmaSize>> : public true_type
        {
        };

        // Convenience evaluators
        template <typename RegisterLayout>
        constexpr static bool is_storage_col_ortho_v = is_storage_col_ortho<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr static bool is_storage_row_ortho_v = is_storage_row_ortho<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr static bool is_storage_col_inline_v
            = is_storage_col_inline<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr static bool is_storage_row_inline_v
            = is_storage_row_inline<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr static bool is_mma_input_v = is_mma_input<RegisterLayout>::value;

        // Compatibility for Storage<MatrixLayout>, passthrough to MatrixLayout compatibility.
        template <typename MatrixLayoutLhs, typename MatrixLayoutRhs>
        struct is_compatible_params<Storage<MatrixLayoutLhs>, Storage<MatrixLayoutRhs>, void>
            : public is_compatible_params<MatrixLayoutLhs, MatrixLayoutRhs>
        {
        };

        // Non-interleaved MmaInput layouts require a valid MmaDim (as BlockDim).
        // MmaDim values must hold to certain criterion in testSupportedMmaDim().
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayout>
        struct is_compatible_params<
            Storage<MatrixLayout<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>,
            MmaInput<BlockDim>,
            enable_if_t<!is_interleaved_v<
                            MatrixLayout<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
                        && testSupportedMmaDim(BlockDim)>> : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayout>
        struct is_compatible_params<
            MmaInput<BlockDim>,
            Storage<MatrixLayout<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>,
            enable_if_t<!is_interleaved_v<
                            MatrixLayout<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>
                        && testSupportedMmaDim(BlockDim)>> : public true_type
        {
        };

        // Interleaved MmaInput layouts require a valid MmaDim.
        // MmaDim values must hold to certain criterion in testSupportedMmaDim().
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaSize,
                  uint32_t SplitK,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayout>
        struct is_compatible_params<
            Storage<MatrixLayout<BlockDim, BlockK, DataT, MmaSize, SplitK>>,
            MmaInput<MmaSize>,
            enable_if_t<is_interleaved_v<MatrixLayout<BlockDim, BlockK, DataT, MmaSize, SplitK>>
                        && testSupportedMmaDim(MmaSize)>> : public true_type
        {
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MmaSize,
                  uint32_t SplitK,
                  template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
                  class MatrixLayout>
        struct is_compatible_params<
            MmaInput<MmaSize>,
            Storage<MatrixLayout<BlockDim, BlockK, DataT, MmaSize, SplitK>>,
            enable_if_t<is_interleaved_v<MatrixLayout<BlockDim, BlockK, DataT, MmaSize, SplitK>>
                        && testSupportedMmaDim(MmaSize)>> : public true_type
        {
        };

        // Checks if both RegisterLayout storages are the same with compatible params
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_layout_same<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<
                // Check for same in-register layouts
                ((is_storage_col_ortho_v<RegisterLayoutLhs>
                  && is_storage_col_ortho_v<RegisterLayoutRhs>)
                 || (is_storage_row_ortho_v<RegisterLayoutLhs>
                     && is_storage_row_ortho_v<RegisterLayoutRhs>)
                 || (is_storage_col_inline_v<RegisterLayoutLhs>
                     && is_storage_col_inline_v<RegisterLayoutRhs>)
                 || (is_storage_row_inline_v<RegisterLayoutLhs>
                     && is_storage_row_inline_v<RegisterLayoutRhs>)
                 // Check for in-register implicit transposes. These have the same register layouts, but swap meaning
                 // for rows / cols.
                 || (is_storage_col_ortho_v<RegisterLayoutLhs>
                     && is_storage_row_ortho_v<RegisterLayoutRhs>)
                 || (is_storage_row_ortho_v<RegisterLayoutLhs>
                     && is_storage_col_ortho_v<RegisterLayoutRhs>)
                 || (is_storage_col_inline_v<RegisterLayoutLhs>
                     && is_storage_row_inline_v<RegisterLayoutRhs>)
                 || (is_storage_row_inline_v<RegisterLayoutLhs>
                     && is_storage_col_inline_v<RegisterLayoutRhs>)
                 // Check for mma input compatibility
                 || (is_storage_col_ortho_v<RegisterLayoutLhs> && is_mma_input_v<RegisterLayoutRhs>)
                 || (is_mma_input_v<RegisterLayoutLhs> && is_storage_col_ortho_v<RegisterLayoutRhs>)
                 || (is_storage_row_ortho_v<RegisterLayoutLhs> && is_mma_input_v<RegisterLayoutRhs>)
                 || (is_mma_input_v<RegisterLayoutLhs>
                     && is_storage_row_ortho_v<RegisterLayoutRhs>))
                && is_compatible_params_v<RegisterLayoutLhs, RegisterLayoutRhs>>> : public true_type
        {
        };

        // Checks if RegisterLayouts are transposed with compatible params
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_layout_orthogonal<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<( // Orthogonality in same orientation (e.g., col / row)
                            (is_storage_col_ortho_v<RegisterLayoutLhs>
                             && is_storage_col_inline_v<RegisterLayoutRhs>)
                            || (is_storage_col_inline_v<RegisterLayoutLhs>
                                && is_storage_col_ortho_v<RegisterLayoutRhs>)
                            || (is_storage_row_ortho_v<RegisterLayoutLhs>
                                && is_storage_row_inline_v<RegisterLayoutRhs>)
                            || (is_storage_row_inline_v<RegisterLayoutLhs>
                                && is_storage_row_ortho_v<RegisterLayoutRhs>)
                            // Orthogonality in opposite orientation (e.g., col vs row)
                            || (is_storage_col_ortho_v<RegisterLayoutLhs>
                                && is_storage_row_inline_v<RegisterLayoutRhs>)
                            || (is_storage_row_inline_v<RegisterLayoutLhs>
                                && is_storage_col_ortho_v<RegisterLayoutRhs>)
                            || (is_storage_col_inline_v<RegisterLayoutLhs>
                                && is_storage_row_ortho_v<RegisterLayoutRhs>)
                            || (is_storage_row_ortho_v<RegisterLayoutLhs>
                                && is_storage_col_inline_v<RegisterLayoutRhs>)
                            // Mma orthogonality
                            || (is_storage_col_inline_v<RegisterLayoutLhs>
                                && is_mma_input_v<RegisterLayoutRhs>)
                            || (is_mma_input_v<RegisterLayoutLhs>
                                && is_storage_col_inline_v<RegisterLayoutRhs>)
                            || (is_storage_row_inline_v<RegisterLayoutLhs>
                                && is_mma_input_v<RegisterLayoutRhs>)
                            || (is_mma_input_v<RegisterLayoutLhs>
                                && is_storage_row_inline_v<RegisterLayoutRhs>))
                        && is_compatible_params_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
            : public true_type
        {
        };

        // Use generic MatrixLayout orthogonality rules to guide the register layout transpose suggestion
        template <typename MatrixLayout>
        struct orthogonal_layout<Storage<MatrixLayout>>
        {
            using type = Storage<typename orthogonal_layout<MatrixLayout>::type>;
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_REGISTER_LAYOUT_TRAITS_HPP
