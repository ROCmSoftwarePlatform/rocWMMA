/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_REGISTER_LAYOUT_TRAITS_IMPL_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRAITS_IMPL_HPP

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
        // - is_storage
        // - is_mma_input
        // - is_mma_acc
        template <typename RegisterLayout>
        struct is_register_layout : public false_type
        {
        };

        template <typename MatrixLayout, typename DataLayout>
        struct is_register_layout<Storage<MatrixLayout, DataLayout>>
            : public is_matrix_layout<MatrixLayout>
        {
        };

        template <uint32_t MmaDim, bool IsInterLeaved>
        struct is_register_layout<MmaInput<MmaDim, IsInterLeaved>> : public true_type
        {
        };

        template <uint32_t MmaDim, bool IsInterleaved>
        struct is_register_layout<MmaAcc<MmaDim, IsInterleaved>> : public true_type
        {
        };

        template <typename RegisterLayout>
        struct is_storage : public false_type
        {
        };

        template <typename MatrixLayout, typename DataLayout>
        struct is_storage<Storage<MatrixLayout, DataLayout>> : public is_matrix_layout<MatrixLayout>
        {
        };

        template <typename RegisterLayout>
        struct is_mma_input : public false_type
        {
        };

        template <uint32_t MmaSize, bool IsInterleaved>
        struct is_mma_input<MmaInput<MmaSize, IsInterleaved>> : public true_type
        {
        };

        template <typename RegisterLayout>
        struct is_mma_acc : public false_type
        {
        };

        template <uint32_t MmaSize, bool IsInterleaved>
        struct is_mma_acc<MmaAcc<MmaSize, IsInterleaved>> : public true_type
        {
        };

        // Convenience evaluators
        template <typename RegisterLayout>
        constexpr inline static bool is_register_layout_v
            = is_register_layout<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr inline static bool is_storage_v = is_storage<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr inline static bool is_mma_input_v = is_mma_input<RegisterLayout>::value;

        template <typename RegisterLayout>
        constexpr inline static bool is_mma_acc_v = is_mma_acc<RegisterLayout>::value;

        template <typename RegisterLayout>
        struct register_layout_classifier_traits
        {
            constexpr static bool is_register_layout = is_register_layout_v<RegisterLayout>;
            constexpr static bool is_storage         = is_storage_v<RegisterLayout>;
            constexpr static bool is_mma_input       = is_mma_input_v<RegisterLayout>;
            constexpr static bool is_mma_acc         = is_mma_acc_v<RegisterLayout>;
        };

        template <typename RegisterLayout>
        struct register_layout_traits;

        // Test the consistency of matrix layouts under different data layouts.
        // RegisterLayouts are consistent for both data layouts if we restrict
        // VectorWidth to 1 in the opposite data layout grain.
        // This applies to all matrix layouts.
        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testStorageLayoutIdentity()
        {
            using traits = register_layout_traits<RegisterLayout>;

            if constexpr(traits::is_col_inline)
            {
                return (traits::is_col_major || traits::VectorWidth == 1);
            }
            else if constexpr(traits::is_row_inline)
            {
                return (traits::is_row_major || traits::VectorWidth == 1);
            }
            else if constexpr(traits::is_col_ortho)
            {
                return (traits::is_row_major || traits::VectorWidth == 1u);
            }
            else if constexpr(traits::is_row_ortho)
            {
                return (traits::is_col_major || traits::VectorWidth == 1u);
            }

            return false;
        }

        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testStorageLayoutAos()
        {
            using traits = register_layout_traits<RegisterLayout>;

            // AOS is a strict register layout where contiguous elements
            // capture contiguous BlockDim elements and must be consistent.
            return (traits::is_col_inline || traits::is_row_inline)
                   && testStorageLayoutIdentity<RegisterLayout>();
        }

        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testStorageLayoutSoa()
        {
            using traits = register_layout_traits<RegisterLayout>;

            // SOA is a strict register layout where contiguous elements
            // capture contiguous BlockK elements and must be consistent.
            return (traits::is_col_ortho || traits::is_row_ortho)
                   && testStorageLayoutIdentity<RegisterLayout>();
        }

        // Based on the current config, mma dimensions supported
        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static inline bool testSupportedMmaDim()
        {
            using traits = register_layout_traits<RegisterLayout>;
            return ((bool)ROCWMMA_BLOCK_DIM_16_SUPPORTED && traits::MmaDim == 16u)
                   || ((bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED
                       && (traits::MmaDim == 16u || traits::MmaDim == 32u));
        }

        template <typename RegisterLayout>
        struct register_layout_derived_traits
        {
        };

        template <typename MatrixLayoutInternal, typename DataLayoutInternal>
        struct register_layout_derived_traits<Storage<MatrixLayoutInternal, DataLayoutInternal>>
            : public matrix_layout_traits<MatrixLayoutInternal>,
              public data_layout_traits<DataLayoutInternal>
        {
            using MatrixLayout = MatrixLayoutInternal;
            using DataLayout   = DataLayoutInternal;

            constexpr static bool is_aos_format
                = testStorageLayoutAos<Storage<MatrixLayout, DataLayout>>();
            constexpr static bool is_soa_format
                = testStorageLayoutSoa<Storage<MatrixLayout, DataLayout>>();
            constexpr static bool is_valid
                = testStorageLayoutIdentity<Storage<MatrixLayout, DataLayout>>();

            constexpr static RegisterLayout::Format Format
                = is_aos_format ? RegisterLayout::Format::AOS
                                : (is_soa_format ? RegisterLayout::Format::SOA
                                                 : RegisterLayout::Format::None);
        };

        template <uint32_t LayoutMmaDim, bool LayoutIsInterleaved, RegisterLayout::Format Fmt>
        struct register_layout_derived_traits<MmaInput<LayoutMmaDim, LayoutIsInterleaved, Fmt>>
            : public matrix_layout_traits<void>, public data_layout_traits<void>
        {
            using MatrixLayout = void;
            using DataLayout   = void;

            // Overrides
            constexpr static bool     is_interleaved = LayoutIsInterleaved;
            constexpr static uint32_t MmaDim         = LayoutMmaDim;

            constexpr static bool is_aos_format = (Fmt == RegisterLayout::Format::AOS);
            constexpr static bool is_soa_format = (Fmt == RegisterLayout::Format::SOA);
            constexpr static bool is_valid
                = testSupportedMmaDim<MmaInput<LayoutMmaDim, LayoutIsInterleaved, Fmt>>();

            constexpr static RegisterLayout::Format Format = Fmt;
        };

        template <uint32_t LayoutMmaDim, bool LayoutIsInterleaved, RegisterLayout::Format Fmt>
        struct register_layout_derived_traits<MmaAcc<LayoutMmaDim, LayoutIsInterleaved, Fmt>>
            : public matrix_layout_traits<void>, public data_layout_traits<void>
        {
            using MatrixLayout = void;
            using DataLayout   = void;

            // Overrides
            constexpr static bool     is_interleaved = LayoutIsInterleaved;
            constexpr static uint32_t MmaDim         = LayoutMmaDim;

            constexpr static bool is_aos_format = (Fmt == RegisterLayout::Format::AOS);
            constexpr static bool is_soa_format = (Fmt == RegisterLayout::Format::SOA);
            constexpr static bool is_valid
                = testSupportedMmaDim<MmaAcc<LayoutMmaDim, LayoutIsInterleaved, Fmt>>();

            constexpr static RegisterLayout::Format Format = Fmt;
        };

        // Combine base instance traits with specific layout classifiers
        template <typename RegisterLayout>
        struct register_layout_traits : public register_layout_derived_traits<RegisterLayout>,
                                        public register_layout_classifier_traits<RegisterLayout>
        {
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
        // | Storage<ColOrthoVW>   | MmaAcc                 | BlockDim == MmaDim, MaxVW = 4* |
        // | MmaAcc                | Storage<ColOrthoVW>    | BlockDim == MmaDim, MaxVW = 4* |
        // | Storage<RowOrthoVW>   | MmaAcc                 | BlockDim == MmaDim, MaxVW = 4* |
        // | MmaAcc                | Storage<RowOrthoVW>    | BlockDim == MmaDim, MaxVW = 4* | * = arch dependent
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
        // |                       |   (Orthogonal)       |                                |
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
        // | Storage<ColInlineVW>  | MmaAcc               | BlockDim == MmaDim             |
        // | MmaAcc                | Storage<ColInlineVW> | BlockDim == MmaDim             |
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
        // | Storage<RowInlineInt> | MmaInput             | MmaDim                         |
        // | MmaInput              | Storage<RowInlineInt>| MmaDim                         |
        // | ----------------------------------------------------------------------------- |

// Keeps things a bit more tidy. Quick access to register layout traits.
#define traits_lhs register_layout_traits<RegisterLayoutLhs>
#define traits_rhs register_layout_traits<RegisterLayoutRhs>
#define traits register_layout_traits<RegisterLayout>

        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static inline bool testSupportedMmaAccMaxVW()
        {
            // Test the MaxVectorWidth of storage layouts for MMA requirements.
            if constexpr(traits::is_storage)
            {
                // Interleaved storage layouts not compatible with MmaAcc
                if constexpr(traits::is_interleaved)
                {
                    return false;
                }
                else if constexpr((bool)ROCWMMA_ARCH_GFX12)
                {
                    return traits::MaxVectorWidth == 8u;
                }
                else if constexpr((bool)ROCWMMA_ARCH_GFX11
                                  || is_same<typename traits::DataT, float64_t>::value)
                {
                    return traits::MaxVectorWidth == 1u;
                }
                else // General case
                {
                    return traits::MaxVectorWidth == 4u;
                }
            }

            // Mma input not compatible with acc
            return traits::is_mma_acc;
        }

        // Test the consistency of matrix layouts under different data layouts.
        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testRegisterLayoutIdentity()
        {
            if constexpr(traits::is_storage)
            {
                // RegisterLayouts are consistent for both data layouts if we restrict
                // VectorWidth to 1 in the opposite data layout grain.
                if constexpr(traits::is_col_inline)
                {
                    return (traits::is_col_major || traits::VectorWidth == 1);
                }
                else if constexpr(traits::is_row_inline)
                {
                    return (traits::is_row_major || traits::VectorWidth == 1);
                }
                else if constexpr(traits::is_col_ortho)
                {
                    return (traits::is_row_major || traits::VectorWidth == 1u);
                }
                else if constexpr(traits::is_row_ortho)
                {
                    return (traits::is_col_major || traits::VectorWidth == 1u);
                }
            }

            // Mma input and acc are symbolic register layouts.
            // Both are consistent in either row/col major data layouts.
            return traits::is_mma_input || traits::is_mma_acc;
        }

        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testRegisterLayoutAos()
        {
            // AOS is a strict register layout where contiguous elements
            // capture contiguous BlockDim elements and must be consistent.
            if constexpr(traits::is_storage)
            {
                return (traits::is_col_inline || traits::is_row_inline)
                       && testRegisterLayoutIdentity<RegisterLayout>();
            }
            else
            {
                // None of the MMA inputs are AOS
                return !traits::is_mma_input && !traits::is_mma_acc;
            }
        }

        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testRegisterLayoutSoa()
        {
            // SOA is a strict register layout where contiguous elements
            // capture contiguous BlockK elements and must be consistent.
            if constexpr(traits::is_storage)
            {
                return (traits::is_col_ortho || traits::is_row_ortho)
                       && testRegisterLayoutIdentity<RegisterLayout>();
            }
            else
            {
                // Interleaved acc is not SOA
                return traits::is_mma_input || (traits::is_mma_acc && !traits::is_interleaved);
            }
        }

        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testRegisterLayoutMmaInput()
        {
            // MMA inputs must be compatible with MMA size support
            if constexpr(traits::is_storage)
            {
                return traits::is_soa_format && testSupportedMmaDim<RegisterLayout>();
            }
            else
            {
                return traits::is_mma_input && testSupportedMmaDim<RegisterLayout>();
            }
        }

        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testRegisterLayoutMmaAcc()
        {
            // MMA acc must be compatible with MMA dim and MaxVW
            if constexpr(traits::is_storage && !traits::is_interleaved)
            {
                return testRegisterLayoutSoa<RegisterLayout>()
                       && testSupportedMmaDim<RegisterLayout>()
                       && testSupportedMmaAccMaxVW<RegisterLayout>();
            }
            else
            {
                // Interleaved storage layouts and MmaInput are not compatible
                // with MMA acc format
                return traits::is_mma_acc && testSupportedMmaDim<RegisterLayout>();
            }
        }

        // As a predicate to is_layout_same or is_layout_orthogonal, their register parameters must
        // be compatible (see above table).
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testCompatibleRegisterParams()
        {
            // Basic test:
            // Matching MmaDim, interleaving and validity
            constexpr bool BaseTest = (traits_lhs::MmaDim == traits_rhs::MmaDim)
                                      && (traits_lhs::is_interleaved == traits_rhs::is_interleaved)
                                      && (traits_lhs::is_valid == traits_rhs::is_valid);

            // Storage <-> Storage must check Matrix compatibility
            if constexpr(traits_lhs::is_storage && traits_rhs::is_storage)
            {
                return testCompatibleMatrixParams<typename traits_lhs::MatrixLayout,
                                                  typename traits_rhs::MatrixLayout>()
                       && BaseTest;
            }
            // MmaInput <-> MmaInput
            // MmaAcc <-> MmaAcc
            // Storage <-> MmaInput
            else if constexpr((traits_lhs::is_mma_input && traits_rhs::is_mma_input)
                              || (traits_lhs::is_mma_acc && traits_rhs::is_mma_acc)
                              || (traits_lhs::is_storage && traits_rhs::is_mma_input)
                              || (traits_lhs::is_mma_input && traits_rhs::is_storage))
            {
                return BaseTest;
            }

            // Storage <-> MmaAcc must also check MaxVW
            else if constexpr((traits_lhs::is_storage && traits_rhs::is_mma_acc)
                              || (traits_lhs::is_mma_acc && traits_rhs::is_storage))
            {
                using test_traits = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;

                constexpr uint32_t ExpectedAccMaxVW
                    = ((bool)ROCWMMA_ARCH_GFX12) ? 8u
                      : ((bool)ROCWMMA_ARCH_GFX11
                         || is_same<typename test_traits::DataT, float64_t>::value)
                          ? 1u
                          : 4u;

                constexpr bool TestMmaAccMaxVW = (ExpectedAccMaxVW == test_traits::MaxVectorWidth);

                return TestMmaAccMaxVW && BaseTest;
            }
            // MmaInput <-> MmaAcc not compatible
            else
            {
                return false;
            }
        }

        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testRegisterLayoutSame()
        {
            // Required compatibility
            constexpr bool TestCompatibleParams
                = testCompatibleRegisterParams<RegisterLayoutLhs, RegisterLayoutRhs>();

            // Test both register layouts in same format
            constexpr bool TestFormatMatch = (traits_lhs::Format == traits_rhs::Format);

            if constexpr(traits_lhs::is_storage && traits_rhs::is_storage)
            {
                // Exact match for same matrix and data layouts
                constexpr bool TestExactMatch
                    = testMatrixLayoutSame<typename traits_lhs::MatrixLayout,
                                           typename traits_rhs::MatrixLayout>()
                      && testDataLayoutSame<typename traits_lhs::DataLayout,
                                            typename traits_rhs::DataLayout>();

                // Orthogonal matrix layout and orthogonal data layout (implicit transpose)
                constexpr bool TestImplicitTranspose
                    = testMatrixLayoutOrthogonal<typename traits_lhs::MatrixLayout,
                                                 typename traits_rhs::MatrixLayout>()
                      && testDataLayoutOrthogonal<typename traits_lhs::DataLayout,
                                                  typename traits_rhs::DataLayout>();

                // Special case: interleaved VW dimension
                // Check matching dims and if either one is == 1u
                if constexpr(traits_lhs::is_interleaved && traits_rhs::is_interleaved)
                {
                    constexpr bool TestIdentityQuirks
                        = (traits_lhs::DimPerThread == traits_rhs::DimPerThread)
                          && (traits_lhs::KPerThread == traits_rhs::KPerThread)
                          && ((traits_lhs::DimPerThread == 1u) || (traits_lhs::KPerThread == 1u));

                    return (TestExactMatch || TestImplicitTranspose || TestFormatMatch
                            || TestIdentityQuirks)
                           && TestCompatibleParams;
                }

                return (TestExactMatch || TestImplicitTranspose || TestFormatMatch)
                       && TestCompatibleParams;
            }
            else // Mix of storage, MmaInput, MmaAcc
            {
                // Test both sides for MmaInput compatibility
                constexpr bool TestMmaInputMatch
                    = testRegisterLayoutMmaInput<RegisterLayoutLhs>()
                      && testRegisterLayoutMmaInput<RegisterLayoutRhs>() && TestCompatibleParams;

                // Test both sides for MmaAcc compatibility
                constexpr bool TestMmaAccMatch = testRegisterLayoutMmaAcc<RegisterLayoutLhs>()
                                                 && testRegisterLayoutMmaAcc<RegisterLayoutRhs>()
                                                 && TestCompatibleParams;

                return (TestMmaInputMatch || TestMmaAccMatch || TestFormatMatch)
                       && TestCompatibleParams;
            }
        }

        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testRegisterLayoutOrthogonal()
        {
            // Required not same and compatible params
            constexpr bool TestNotSame
                = !testRegisterLayoutSame<RegisterLayoutLhs, RegisterLayoutRhs>();
            constexpr bool TestCompatibleParams
                = testCompatibleRegisterParams<RegisterLayoutLhs, RegisterLayoutRhs>();

            // Path between valid AOS and SOA formats
            constexpr bool TestOpposingFormat
                = (traits_lhs::is_soa_format && traits_rhs::is_aos_format)
                  || (traits_lhs::is_aos_format && traits_rhs::is_soa_format);

            // (testRegisterLayoutAos<RegisterLayoutLhs>() && testRegisterLayoutSoa<RegisterLayoutRhs>())
            // || (testRegisterLayoutSoa<RegisterLayoutLhs>() && testRegisterLayoutAos<RegisterLayoutRhs>());

            if constexpr((traits_lhs::is_interleaved && traits_rhs::is_interleaved)
                         && (traits_lhs::is_mma_acc || traits_rhs::is_mma_acc))
            {
                using RegisterLayoutMmaAcc
                    = conditional_t<traits_lhs::is_mma_acc, RegisterLayoutLhs, RegisterLayoutRhs>;
                using RegisterLayoutOther
                    = conditional_t<traits_lhs::is_mma_acc, RegisterLayoutRhs, RegisterLayoutLhs>;

                // Special case: path between valid interleaved AOS/SOA and MmaAcc register layouts exists.
                constexpr bool TestStorageToAcc
                    = testRegisterLayoutMmaAcc<RegisterLayoutMmaAcc>()
                      && (testRegisterLayoutAos<RegisterLayoutOther>()
                          || testRegisterLayoutSoa<RegisterLayoutOther>());

                return (TestOpposingFormat || TestStorageToAcc) && TestNotSame
                       && TestCompatibleParams;
            }
            else
            {
                return TestOpposingFormat && TestNotSame && TestCompatibleParams;
            }
        }

        // Checks if both RegisterLayout storages are the same with compatible params
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_layout_same<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<traits_lhs::is_register_layout && traits_rhs::is_register_layout>>
            : public integral_constant<
                  bool,
                  testRegisterLayoutSame<RegisterLayoutLhs, RegisterLayoutRhs>()>
        {
        };

        // Checks if RegisterLayouts are transposed with compatible params
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct is_layout_orthogonal<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<traits_lhs::is_register_layout && traits_rhs::is_register_layout>>
            : public integral_constant<
                  bool,
                  testRegisterLayoutOrthogonal<RegisterLayoutLhs, RegisterLayoutRhs>()>
        {
        };

#undef traits_lhs
#undef traits_rhs
#undef traits

        // Use generic MatrixLayout orthogonality rules to guide the register layout transpose suggestion
        // TODO: fix
        template <typename MatrixLayout, typename DataLayout>
        struct orthogonal_layout<Storage<MatrixLayout, DataLayout>>
        {
            using type = Storage<typename orthogonal_layout<MatrixLayout>::type,
                                 typename orthogonal_layout<DataLayout>::type>;
        };

        template <typename RegisterLayout>
        struct layout_traits<RegisterLayout, enable_if_t<is_register_layout_v<RegisterLayout>>>
            : public register_layout_traits<RegisterLayout>
        {
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#if !defined(__HIPCC_RTC__)
namespace std
{

    template <typename RegisterLayout>
    inline ostream&
        operator<<(ostream&                                                                  stream,
                   rocwmma::LayoutTraits_impl::register_layout_traits<RegisterLayout> const& traits)
    {
        using register_traits = decay_t<decltype(traits)>;

        stream << "RegisterLayout Traits: " << RegisterLayout{} << std::endl;
        stream << "is_register_layout: " << traits.is_register_layout << std::endl;
        stream << "is_storage: " << traits.is_storage << std::endl;
        stream << "is_mma_input: " << traits.is_mma_input << std::endl;
        stream << "is_mma_acc: " << traits.is_mma_acc << std::endl;
        stream << "is_interleaved: " << traits.is_interleaved << std::endl;
        stream << "MmaDim: " << traits.MmaDim << std::endl;
        stream << "is_aos_format: " << traits.is_aos_format << std::endl;
        stream << "is_soa_format: " << traits.is_soa_format << std::endl;
        stream << "is_valid: " << traits.is_valid << std::endl;
        stream << "Format: " << traits.Format << std::endl;

        return stream;
    }

} // namespace std

#endif // !defined(__HIPCC_RTC__)

#endif // ROCWMMA_REGISTER_LAYOUT_TRAITS_IMPL_HPP
