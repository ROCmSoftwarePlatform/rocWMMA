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

        // AOS is a strict register layout where thread VW is inline
        // with contiguous BlockDim elements.
        // To be valid, the layout must be consistent across row_major
        // and col_major data layouts.
        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testStorageLayoutAos()
        {
            using traits = register_layout_traits<RegisterLayout>;
            return (traits::is_col_inline || traits::is_row_inline);
        }

        // SOA is a strict register layout where thread VW is inline
        // with contiguous BlockK elements, orthogonal to BlockDim.
        // To be valid, the layout must be consistent across row_major
        // and col_major data layouts.
        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static bool testStorageLayoutSoa()
        {
            using traits = register_layout_traits<RegisterLayout>;
            return (traits::is_col_ortho || traits::is_row_ortho);
        }

        // Based on the current architecture, which mma dimensions supported
        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static inline bool testSupportedMmaDim()
        {
            using traits = register_layout_traits<RegisterLayout>;
            return ((bool)ROCWMMA_BLOCK_DIM_16_SUPPORTED && traits::MmaDim == 16u)
                   || ((bool)ROCWMMA_BLOCK_DIM_32_SUPPORTED && traits::MmaDim == 32u);
        }

        // Based on the current architecture, which register layout formats currently supported
        template <typename RegisterLayout>
        ROCWMMA_HOST_DEVICE constexpr static inline bool testSupportedFormat()
        {
            using traits = register_layout_traits<RegisterLayout>;
            using rocwmma::RegisterLayout::Format;
            if constexpr(traits::is_mma_input)
            {
                return (traits::Format == Format::SOA) || (traits::Format == Format::AOS);
            }
            else if constexpr(traits::is_mma_acc)
            {
                if constexpr(traits::is_interleaved)
                {
                    // Intermediate accumulation format for interleaved layout
                    return (traits::Format == Format::ACC_INT_A_MAJOR)
                           || (traits::Format == Format::ACC_INT_B_MAJOR);
                }
                else
                {
                    return (traits::Format == Format::SOA) || (traits::Format == Format::AOS);
                }
            }
            else
            {
                return traits::is_storage
                       && ((traits::Format == Format::SOA) || (traits::Format == Format::AOS)
                           || (traits::Format == Format::Invalid));
            }
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

            // Determine the register format of the current storage layout
            constexpr static RegisterLayout::Format Format
                = testStorageLayoutAos<Storage<MatrixLayout, DataLayout>>()
                      ? RegisterLayout::Format::AOS
                      : (testStorageLayoutSoa<Storage<MatrixLayout, DataLayout>>()
                             ? RegisterLayout::Format::SOA
                             : RegisterLayout::Format::Invalid);

            constexpr static bool is_valid
                = testStorageLayoutIdentity<Storage<MatrixLayout, DataLayout>>()
                  && testSupportedFormat<Storage<MatrixLayout, DataLayout>>();
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

            // Template param driven format
            constexpr static RegisterLayout::Format Format = Fmt;

            constexpr static bool is_valid
                = testSupportedMmaDim<MmaInput<LayoutMmaDim, LayoutIsInterleaved, Fmt>>()
                  && testSupportedFormat<MmaInput<LayoutMmaDim, LayoutIsInterleaved, Fmt>>();
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

            // Template param driven format
            constexpr static RegisterLayout::Format Format = Fmt;

            constexpr static bool is_valid
                = testSupportedMmaDim<MmaAcc<LayoutMmaDim, LayoutIsInterleaved, Fmt>>()
                  && testSupportedFormat<MmaAcc<LayoutMmaDim, LayoutIsInterleaved, Fmt>>();
        };

        // Combine base instance traits with specific layout classifiers
        template <typename RegisterLayout>
        struct register_layout_traits : public register_layout_classifier_traits<RegisterLayout>,
                                        public register_layout_derived_traits<RegisterLayout>

        {
        };

        // NOTE: RegisterLayout comparison assumptions
        // When determining RegisterLayout traits, there are several strong assumptions.
        // Register layouts are assigned Formats, based on their given matrix and data layouts.
        // 1. Regarding same-ness:
        //    - Register formats match, if tested for matching register layout traits:
        //      MmaDim, is_interleaved and is_valid.
        //    - Register layouts match if register formats match, and there is congruency between
        //      Storage, MmaInput and MmaAcc types.
        //    - Congruency between Storage, MmaInput and MmaAcc types is partly defined by how
        //      MmaInput and MmaAcc register format template parameters are set for the Mma workflow,
        //      and partly by architecture (e.g., MmaAcc layout VW per block is fixed).
        //
        // 2. Regarding orthogonality:
        //    - Format orthogonality is defined as having an in-register transition from one distinct
        //      format to another.
        //      E.g,. AOS <-> SOA, SOA <-> ACC_INT_A_MAJOR, SOA <-> ACC_INT_B_MAJOR,
        //      AOS <-> ACC_INT_A_MAJOR or AOS <-> ACC_INT_B_MAJOR.
        //      These require matching MmaDim, is_interleaved and is_valid traits.

// Keeps things a bit more tidy. Quick access to register layout traits.
#define traits_lhs register_layout_traits<RegisterLayoutLhs>
#define traits_rhs register_layout_traits<RegisterLayoutRhs>
#define traits register_layout_traits<RegisterLayout>

        // As a predicate to is_layout_same or is_layout_orthogonal, their register traits must
        // be compatible as per above.
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        ROCWMMA_HOST_DEVICE constexpr static bool testCompatibleRegisterParams()
        {
            // Basic test:
            // Matching MmaDim, interleaving and validity
            // Note: matching validity does not imply valid!
            // Cannot mix valid with invalid layouts
            constexpr bool BaseTest = (traits_lhs::MmaDim == traits_rhs::MmaDim)
                                      && (traits_lhs::is_interleaved == traits_rhs::is_interleaved)
                                      && (traits_lhs::is_valid == traits_rhs::is_valid);

            // MmaInput <-> MmaInput
            // MmaAcc <-> MmaAcc
            // Storage <-> MmaInput
            // MmaDim must match and be supported
            if constexpr((traits_lhs::is_mma_input && traits_rhs::is_mma_input)
                         || (traits_lhs::is_mma_acc && traits_rhs::is_mma_acc)
                         || (traits_lhs::is_storage && traits_rhs::is_mma_input)
                         || (traits_lhs::is_mma_input && traits_rhs::is_storage))
            {
                return BaseTest && testSupportedMmaDim<RegisterLayoutLhs>();
            }
            // Storage <-> MmaAcc
            // MmaAcc must check MaxVW
            // MmaDim must match and be supported
            else if constexpr((traits_lhs::is_storage && traits_rhs::is_mma_acc)
                              || (traits_lhs::is_mma_acc && traits_rhs::is_storage))
            {
                using test_traits = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;

                if constexpr(test_traits::is_interleaved)
                {
                    return ((test_traits::Format == RegisterLayout::Format::ACC_INT_A_MAJOR)
                            || (test_traits::Format == RegisterLayout::Format::ACC_INT_B_MAJOR))
                           && BaseTest && testSupportedMmaDim<RegisterLayoutLhs>();
                }
                else
                {
                    // Acc layout architecture quirks
                    constexpr uint32_t ExpectedAccMaxVW
                        = ((bool)ROCWMMA_ARCH_GFX12) ? 8u
                          : ((bool)ROCWMMA_ARCH_GFX11
                             || is_same<typename test_traits::DataT, float64_t>::value)
                              ? 1u
                              : 4u;

                    constexpr bool TestMmaAccMaxVW
                        = (ExpectedAccMaxVW == test_traits::MaxVectorWidth);

                    return TestMmaAccMaxVW && BaseTest && testSupportedMmaDim<RegisterLayoutLhs>();
                }
            }
            // Storage <-> Storage
            // Must check Matrix compatibility
            // Not necessary to check MmaDim because doesn't involve MmaInput of MmaAcc
            else if constexpr(traits_lhs::is_storage && traits_rhs::is_storage)
            {
                return testCompatibleMatrixParams<typename traits_lhs::MatrixLayout,
                                                  typename traits_rhs::MatrixLayout>()
                       && BaseTest;
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

            if constexpr((traits_lhs::is_storage && traits_rhs::is_storage)
                         && (traits_lhs::is_interleaved && traits_rhs::is_interleaved))
            {
                // Special case: interleaved layouts
                // Check matching thread dims and if either one is == 1u.
                // Format match not required because the in this case,
                // register contents for SOA and AOS are identical
                constexpr bool TestIdentityQuirks
                    = (traits_lhs::DimPerThread == traits_rhs::DimPerThread)
                      && (traits_lhs::KPerThread == traits_rhs::KPerThread)
                      && ((traits_lhs::DimPerThread == 1u) || (traits_lhs::KPerThread == 1u));

                return TestIdentityQuirks && TestCompatibleParams;
            }
            else
            {
                // Test both register layouts in same format
                return TestCompatibleParams && (traits_lhs::Format == traits_rhs::Format);
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

            // Identify valid paths in orthogonality.
            // SOA <-> AOS
            // ACC_INT_A_MAJOR <-> AOS, SOA
            // ACC_INT_B_MAJOR <-> AOS, SOA
            // Register layouts must be valid to be orthogonal
            using RegisterLayout::Format;
            constexpr bool TestOpposingFormat
                = ((traits_lhs::Format == Format::SOA && traits_rhs::Format == Format::AOS)
                   || (traits_lhs::Format == Format::AOS && traits_rhs::Format == Format::SOA)
                   || (traits_lhs::Format == Format::ACC_INT_A_MAJOR
                       && traits_rhs::Format == Format::SOA)
                   || (traits_lhs::Format == Format::ACC_INT_A_MAJOR
                       && traits_rhs::Format == Format::AOS)
                   || (traits_lhs::Format == Format::SOA
                       && traits_rhs::Format == Format::ACC_INT_A_MAJOR)
                   || (traits_lhs::Format == Format::AOS
                       && traits_rhs::Format == Format::ACC_INT_A_MAJOR)
                   || (traits_lhs::Format == Format::ACC_INT_B_MAJOR
                       && traits_rhs::Format == Format::SOA)
                   || (traits_lhs::Format == Format::ACC_INT_B_MAJOR
                       && traits_rhs::Format == Format::AOS)
                   || (traits_lhs::Format == Format::SOA
                       && traits_rhs::Format == Format::ACC_INT_B_MAJOR)
                   || (traits_lhs::Format == Format::AOS
                       && traits_rhs::Format == Format::ACC_INT_B_MAJOR))
                  && (traits_lhs::is_valid && traits_rhs::is_valid);

            return TestNotSame && TestCompatibleParams && TestOpposingFormat;
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
        // stream << "is_aos_format: " << traits.is_aos_format << std::endl;
        // stream << "is_soa_format: " << traits.is_soa_format << std::endl;
        stream << "is_valid: " << traits.is_valid << std::endl;
        stream << "Format: " << traits.Format << std::endl;

        return stream;
    }

} // namespace std

#endif // !defined(__HIPCC_RTC__)

#endif // ROCWMMA_REGISTER_LAYOUT_TRAITS_IMPL_HPP
