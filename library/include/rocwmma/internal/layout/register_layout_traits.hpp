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

#include "../utility/type_traits.hpp"
#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    namespace LayoutTraits_impl
    {
        using RegisterLayout::MmaAcc;
        using RegisterLayout::MmaInput;
        using RegisterLayout::Storage;

        // Start to build a basic set of meta-data classifiers.
        // We will be interested in knowing things about our register layouts:
        // - is_register_layout
        // - is_storage_layout
        // - is_mma_input_layout
        // - is_mma_acc_layout
        template <typename RegisterLayout>
        struct is_register_layout : public false_type
        {
        };

        template <typename MatrixLayout>
        struct is_register_layout<Storage<MatrixLayout>> : public is_matrix_layout<MatrixLayout>
        {
        };

        template <uint32_t MmaDim>
        struct is_register_layout<MmaInput<MmaDim>> : public true_type
        {
        };

        template <uint32_t MmaDim>
        struct is_register_layout<MmaAcc<MmaDim>> : public true_type
        {
        };

        template <typename RegisterLayout>
        struct is_storage_layout : public false_type
        {
        };

        template <typename MatrixLayout>
        struct is_storage_layout<Storage<MatrixLayout>> : public is_matrix_layout<MatrixLayout>
        {
        };

        template <typename RegisterLayout>
        struct is_mma_input_layout : public false_type
        {
        };

        template <uint32_t MmaSize>
        struct is_mma_input_layout<MmaInput<MmaSize>> : public true_type
        {
        };

        template <typename RegisterLayout>
        struct is_mma_acc_layout : public false_type
        {
        };

        template <uint32_t MmaSize>
        struct is_mma_acc_layout<MmaAcc<MmaSize>> : public true_type
        {
        };

        // Convenience evaluators
        template <typename RegisterLayout>
        constexpr inline static bool is_register_layout_v
            = is_register_layout<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr inline static bool is_storage_layout_v = is_storage_layout<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr inline static bool is_mma_input_layout_v
            = is_mma_input_layout<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr inline static bool is_mma_acc_layout_v = is_mma_acc_layout<RegisterLayout>::value;

        // Next we can build a set of base trait accessors for the RegisterLayout. These
        // will be reflective of the input template params of the RegisterLayout instance.
        template <typename RegisterLayout>
        struct register_layout_base_traits;

        template <typename MatrixLayoutInternal>
        struct register_layout_base_traits<Storage<MatrixLayoutInternal>>
        {
            using MatrixLayout = MatrixLayoutInternal;
        };

        template <uint32_t LayoutMmaDim>
        struct register_layout_base_traits<MmaInput<LayoutMmaDim>>
        {
            constexpr static uint32_t MmaDim = LayoutMmaDim;
            using MatrixLayout               = void;
        };

        template <uint32_t LayoutMmaDim>
        struct register_layout_base_traits<MmaAcc<LayoutMmaDim>>
        {
            constexpr static uint32_t MmaDim = LayoutMmaDim;
            using MatrixLayout               = void;
        };

        // Combine base instance traits with specific layout classifiers
        template <typename RegisterLayout>
        struct register_layout_traits : public register_layout_base_traits<RegisterLayout>
        {
            constexpr static bool is_register_layout  = is_register_layout_v<RegisterLayout>;
            constexpr static bool is_storage_layout   = is_storage_layout_v<RegisterLayout>;
            constexpr static bool is_mma_input_layout = is_mma_input_layout_v<RegisterLayout>;
            constexpr static bool is_mma_acc_layout   = is_mma_acc_layout_v<RegisterLayout>;
        };

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

// Keeps things a bit more tidy. Quick access to register layout traits.
#define reg_traits_lhs register_layout_traits<RegisterLayoutLhs>
#define reg_traits_rhs register_layout_traits<RegisterLayoutRhs>

// Quick access to matrix layout traits, that are embedded in the register layout traits.
#define mat_traits_lhs matrix_layout_traits<typename reg_traits_lhs::MatrixLayout>
#define mat_traits_rhs matrix_layout_traits<typename reg_traits_rhs::MatrixLayout>

        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs, typename Enabler = void>
        struct is_compatible_register_params;

        // Compatibility for Storage<MatrixLayout> is a passthrough to MatrixLayout compatibility.
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_compatible_register_params<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<reg_traits_lhs::is_storage_layout && reg_traits_rhs::is_storage_layout>>
            : public is_compatible_matrix_params<typename reg_traits_lhs::MatrixLayout,
                                                 typename reg_traits_rhs::MatrixLayout>
        {
        };

        // Compatibility for MmaInputs
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_compatible_register_params<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<reg_traits_lhs::is_mma_input_layout && reg_traits_rhs::is_mma_input_layout>>
            : public integral_constant<bool,
                                       (reg_traits_rhs::MmaDim == reg_traits_rhs::MmaDim)
                                           && testSupportedMmaDim(reg_traits_rhs::MmaDim)>
        {
        };

        // Non-interleaved register layout compatibility with MmaInput requires:
        // 1. Inner matrix layout and mma input layout must have same: BlockDim / MmaDim
        // 2. MmaDim must satisfy criterion in testSupportedMmaDim().
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_compatible_register_params<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<(reg_traits_lhs::is_storage_layout && !mat_traits_lhs::is_interleaved)
                        && reg_traits_rhs::is_mma_input_layout>>
            : public integral_constant<bool,
                                       (mat_traits_lhs::BlockDim == reg_traits_rhs::MmaDim)
                                           && testSupportedMmaDim(reg_traits_rhs::MmaDim)>
        {
        };

        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_compatible_register_params<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<reg_traits_lhs::is_mma_input_layout
                        && (reg_traits_rhs::is_storage_layout && !mat_traits_rhs::is_interleaved)>>
            : public integral_constant<bool,
                                       (mat_traits_rhs::BlockDim == reg_traits_lhs::MmaDim)
                                           && testSupportedMmaDim(reg_traits_lhs::MmaDim)>
        {
        };

        // Interleaved register layout compatibility with MmaInput requires:
        // 1. Inner matrix layout and mma input layout must have same: MmaDim
        // 2. MmaDim must satisfy criterion in testSupportedMmaDim().
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_compatible_register_params<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<(reg_traits_lhs::is_storage_layout && mat_traits_lhs::is_interleaved)
                        && reg_traits_rhs::is_mma_input_layout>>
            : public integral_constant<bool,
                                       (mat_traits_lhs::MmaDim == reg_traits_rhs::MmaDim)
                                           && testSupportedMmaDim(reg_traits_lhs::MmaDim)>
        {
        };

        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_compatible_register_params<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<reg_traits_lhs::is_mma_input_layout
                        && (reg_traits_rhs::is_storage_layout && mat_traits_rhs::is_interleaved)>>
            : public integral_constant<bool,
                                       (mat_traits_rhs::MmaDim == reg_traits_lhs::MmaDim)
                                           && testSupportedMmaDim(reg_traits_lhs::MmaDim)>
        {
        };

        // Convenience evaluator
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        constexpr static inline bool is_compatible_register_params_v
            = is_compatible_register_params<RegisterLayoutLhs, RegisterLayoutRhs>::value;

        // Checks if both RegisterLayout storages are the same with compatible params
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_layout_same<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<reg_traits_lhs::is_register_layout && reg_traits_rhs::is_register_layout>>
            : public integral_constant<
                  bool,
                  // Check for same in-register layouts
                  ((mat_traits_lhs::is_col_ortho && mat_traits_rhs::is_col_ortho)
                   || (mat_traits_lhs::is_row_ortho && mat_traits_rhs::is_row_ortho)
                   || (mat_traits_lhs::is_col_inline && mat_traits_rhs::is_col_inline)
                   || (mat_traits_lhs::is_row_inline && mat_traits_rhs::is_row_inline)

                   // Check for in-register implicit transposes. These have the same register layouts,
                   // but swap meaning for rows / cols.
                   || (mat_traits_lhs::is_col_ortho && mat_traits_rhs::is_row_ortho)
                   || (mat_traits_lhs::is_row_ortho && mat_traits_rhs::is_col_ortho)
                   || (mat_traits_lhs::is_col_inline && mat_traits_rhs::is_row_inline)
                   || (mat_traits_lhs::is_row_inline && mat_traits_rhs::is_col_inline)

                   // Check mma input sameness
                   || (reg_traits_lhs::is_mma_input && reg_traits_rhs::is_mma_input)
                   || (mat_traits_lhs::is_col_ortho && reg_traits_rhs::is_mma_input)
                   || (reg_traits_lhs::is_mma_input && mat_traits_rhs::is_col_ortho)
                   || (mat_traits_lhs::is_row_ortho && reg_traits_rhs::is_mma_input)
                   || (reg_traits_lhs::is_mma_input && mat_traits_rhs::is_row_ortho))
                      && is_compatible_register_params_v<RegisterLayoutLhs, RegisterLayoutRhs>>
        {
        };

        // Checks if RegisterLayouts are transposed with compatible params
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_layout_orthogonal<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<reg_traits_lhs::is_register_layout && reg_traits_rhs::is_register_layout>>
            : public integral_constant<
                  bool,
                  // Orthogonality in same orientation (e.g., col / row)
                  ((mat_traits_lhs::is_col_ortho && mat_traits_rhs::is_col_inline)
                   || (mat_traits_lhs::is_col_inline && mat_traits_rhs::is_col_ortho)
                   || (mat_traits_lhs::is_row_ortho && mat_traits_rhs::is_row_inline)
                   || (mat_traits_lhs::is_row_inline && mat_traits_rhs::is_row_ortho)

                   // Orthogonality in opposite orientation (e.g., col vs row)
                   || (mat_traits_lhs::is_col_ortho && mat_traits_rhs::is_row_inline)
                   || (mat_traits_lhs::is_row_inline && mat_traits_rhs::is_col_ortho)
                   || (mat_traits_lhs::is_col_inline && mat_traits_rhs::is_row_ortho)
                   || (mat_traits_lhs::is_row_ortho && mat_traits_rhs::is_col_inline)

                   // Check mma input compatibility
                   || (mat_traits_lhs::is_col_inline && reg_traits_rhs::is_mma_input)
                   || (reg_traits_lhs::is_mma_input && mat_traits_rhs::is_col_inline)
                   || (mat_traits_lhs::is_row_inline && reg_traits_rhs::is_mma_input)
                   || (reg_traits_lhs::is_mma_input && mat_traits_rhs::is_row_inline))
                      && is_compatible_register_params_v<RegisterLayoutLhs, RegisterLayoutRhs>>
        {
        };

#undef reg_traits_lhs
#undef reg_traits_rhs
#undef mat_traits_lhs
#undef mat_traits_rhs

        // Use generic MatrixLayout orthogonality rules to guide the register layout transpose suggestion
        template <typename MatrixLayout>
        struct orthogonal_layout<Storage<MatrixLayout>>
        {
            using type = Storage<typename orthogonal_layout<MatrixLayout>::type>;
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_REGISTER_LAYOUT_TRAITS_HPP
