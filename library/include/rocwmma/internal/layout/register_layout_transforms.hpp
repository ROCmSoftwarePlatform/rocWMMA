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
#ifndef ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP

#include "../transforms.hpp"
#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    template<uint32_t DimPerThread, uint32_t KPerThread>
    struct soa_int_to_aos_int
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
        {
            // interleave<1, KPT, VecSize>
            constexpr uint32_t GatherSize = 1u;
            return interleave<GatherSize, KPerThread>(forward<VecT>(v));
        }
    };

    template<uint32_t DimPerThread, uint32_t KPerThread>
    struct aos_int_to_soa_int
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
        {
            // interleave<1, DPT, VecSize>
            constexpr uint32_t GatherSize = 1u;
            return interleave<GatherSize, DimPerThread>(forward<VecT>(v));
        }
    };

    struct to_wmma_input_gfx11
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline auto exec(VecT&& v)
        {
            // Swap + concat
            // v is unpacked
            using VecTraits = VecTraits<decay_t<VecT>>;
            using PackUtil = PackUtil<typename VecTraits::DataT>;

            // Swap upper / lower 16's and then concatenate them
            // to make sure we have each K value in each half.
            // GFX11 wmma layout quirk needs the duplication.
            auto packed = PackUtil::pack(v);
            auto swapped = Swizzle::Swap16::exec(packed);
            auto result = PackUtil::unpack(concat(packed, swapped));
            return result; // Return by copy
        }
    };

    struct from_wmma_input_gfx11
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
        {
            // Discard the swapped dups
            return extractLo(v);
        }
    };

    struct to_wmma_acc_gfx11
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            // pad to wmma accumulator on gfx11.
            // f16 -> padded to f32, with data in lower 16
            // f32 -> nop
            using PackUtil = PackUtil<typename VecTraits::DataT>;
            auto accum = PackUtil::unpack(PackUtil::template pad<>(v));
            return accum; // Return by copy
        }
    };

    struct from_wmma_acc_gfx11
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            // unpad from wmma accumulator on gfx11.
            // f16 -> padded to f32 in lower 16
            // f32 -> nop
            using PackUtil = PackUtil<typename VecTraits::DataT>;
            return PackUtil::template unpad<>(PackUtil::pack(v));
        }
    };

    template<uint32_t BlockDim, uint32_t BlockK, uint32_t MaxVW, uint32_t MmaDim, uint32_t DimPerThread, uint32_t KPerThread>
    struct soa_int_to_mma_acc_int_a_major
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;

            // Vector size per acc block
            constexpr uint32_t AccVecSize = MmaDim * MmaDim / Constants::AMDGCN_WAVE_SIZE;
            constexpr uint32_t MmaBlocksA = BlockK / MmaDim;
            constexpr uint32_t MmaBlocksB = BlockDim / MmaDim;

            if constexpr((bool)ROCWMMA_ARCH_GFX9)
            {
                if constexpr (MaxVW == 1u)
                {
                    // First, interleave full vector
                    // interleave<1, MmaBlocksA, VecSize>
                    auto result = interleave<1u, MmaBlocksA>(v);

                    // For each subvector of AccVecSize:
                    // unpackLoHi16 + unpackLoHi32
                    return vector_for_each<AccVecSize>(
                        result,
                        [](auto&& v, uint32_t idx)
                        {
                            return unpackLoHi32(unpackLoHi16(v));
                        });
                }
                else if constexpr (MaxVW == 4u)
                {
                    // Interleave full vector
                    return interleave<1u, DimPerThread>(forward<VecT>(v));
                }
                else
                {
                    static_assert(0, "Shouldn't get here");
                    return forward<VecT>(v);
                }
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX11)
            {
                using interleave_idx0 = interleave_idx<1u, MmaBlocksA, VecTraits::size()>;
                using interleave_idx1 = interleave_idx<1u, 2u, AccVecSize>;

                // First perform combined interleave on full vector
                // interleave<1, MmaBlocksA, VecSize> + interleave<1, 2, AccVecSize>
                auto result = interleave_combine<interleave_idx0, interleave_idx1>(forward<VecT>(v));

                // For each subvector of AccVecSize:
                // unpackLoHi16
                return vector_for_each<AccVecSize>(
                    result,
                    [](auto&& v, uint32_t idx)
                    {
                        return unpackLoHi16(extractLo(v), extractHi(v));
                    });
            }
            else if constexpr((bool)ROCWMMA_ARCH_GFX12)
            {

            }
            else
            {
                static_assert(0, "Shouldn't get here");
                return forward<VecT>(v);
            }

        }
    };

    template<uint32_t DimPerThread, uint32_t KPerThread>
    struct aos_int_to_mma_acc_int_a_major
    {
        template<typename VecT>
        ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
        {
            return interleave<1u, DimPerThread>(forward<VecT>(v));
        }
    };

    namespace RegisterTransform_impl
    {
        using LayoutTraits_impl::matrix_layout_traits;
        using LayoutTraits_impl::register_layout_traits;

// Keeps things a bit more tidy. Quick access to register layout traits.
#define traits_lhs register_layout_traits<RegisterLayoutLhs>
#define traits_rhs register_layout_traits<RegisterLayoutRhs>

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

        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<!is_layout_same_v<RegisterLayoutLhs, RegisterLayoutRhs>
                        && (!traits_lhs::is_register_layout || !traits_rhs::is_register_layout
                            || !is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>)>>
        {
            template <typename VecT>
            ROCWMMA_UNSUPPORTED_IMPL("Register layout transform is not supported")
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // No-op
                return v;
            }
        };

        // Apply paths between orthogonal transforms
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            enable_if_t<(traits_lhs::is_register_layout && traits_rhs::is_register_layout)
                        && is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using RegisterLayout::Format;
                using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;

                // Non-interleaved
                if constexpr(traits_lhs::Format == Format::AOS
                          && traits_rhs::Format == Format::SOA)
                {
                    return Transforms::
                        AosToSoa<storage_traits::BlockDim, storage_traits::MaxVectorWidth>::exec(
                            forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::SOA
                               && traits_rhs::Format == Format::AOS)
                {
                    return Transforms::
                        SoaToAos<storage_traits::BlockDim, storage_traits::MaxVectorWidth>::exec(
                            forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::AOS
                               && traits_rhs::Format == Format::WMMA_INPUT_GFX11)
                {
                    return to_wmma_input_gfx11::exec(Transforms::AosToSoa<storage_traits::BlockDim, storage_traits::MaxVectorWidth>::exec(forward<VecT>(v)));
                }
                else if constexpr(traits_lhs::Format == Format::SOA
                               && traits_rhs::Format == Format::WMMA_INPUT_GFX11)
                {
                    return to_wmma_input_gfx11::exec(forward<VecT>(v));
                }
                // Interleaved
                else if constexpr(traits_lhs::Format == Format::AOS_INT
                               && traits_rhs::Format == Format::SOA_INT)
                {
                    return aos_int_to_soa_int<storage_traits::DimPerThread, storage_traits::KPerThread>::exec(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::SOA_INT
                               && traits_rhs::Format == Format::AOS_INT)
                {
                    return soa_int_to_aos_int<storage_traits::DimPerThread, storage_traits::KPerThread>::exec(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::SOA_INT
                               && traits_rhs::Format == Format::WMMA_INPUT_GFX11)
                {
                    return to_wmma_input_gfx11::exec(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::AOS_INT
                               && traits_rhs::Format == Format::WMMA_INPUT_GFX11)
                {
                    return to_wmma_input_gfx11::exec(aos_int_to_soa_int<storage_traits::DimPerThread, storage_traits::KPerThread>::exec(forward<VecT>(v)));
                }
                else if constexpr(traits_lhs::Format == Format::ACC_INT_A_MAJOR
                                  && traits_rhs::Format == Format::AOS_INT)
                {
                    return interleave<1u, 4u>(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::AOS_INT
                               && traits_rhs::Format == Format::ACC_INT_A_MAJOR)
                {
                    return interleave<1u, storage_traits::KPerThread>(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::SOA_INT
                                  && traits_rhs::Format == Format::ACC_INT_A_MAJOR)
                {
                    return interleave<1u, 4u>(forward<VecT>(v));
                }

                else if constexpr(traits_lhs::Format == Format::ACC_INT_A_MAJOR
                                  && traits_rhs::Format == Format::SOA_INT)
                {
                    return interleave<1u, storage_traits::KPerThread>(forward<VecT>(v));
                }

                else if constexpr((traits_lhs::Format == Format::SOA
                                || traits_lhs::Format == Format::ACC_INT_A_MAJOR
                                || traits_lhs::Format == Format::ACC_INT_B_MAJOR)
                               && (traits_rhs::Format == Format::WMMA_ACC_GFX11))
                {
                    return to_wmma_acc_gfx11::exec(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::AOS
                               && traits_rhs::Format == Format::WMMA_ACC_GFX11)
                {
                    return to_wmma_acc_gfx11::exec(forward<VecT>(v));
                }
                else if constexpr((traits_lhs::Format == Format::WMMA_ACC_GFX11)
                               && (traits_rhs::Format == Format::SOA
                                  || traits_rhs::Format == Format::ACC_INT_A_MAJOR
                                  || traits_rhs::Format == Format::ACC_INT_B_MAJOR))
                {
                    // Padded wmma acc (gfx11) back to SOA format.
                    // f16 -> padded to f32 in lower 16
                    // f32 -> nop
                    using PackUtil = PackUtil<typename traits_lhs::DataT>;
                    return PackUtil::template unpad<>(PackUtil::pack(v));
                }
                else
                {
                    static_assert(0, "Register layout transform is not implemented");
                    return v;
                }
            }
        };

#undef traits_lhs
#undef traits_rhs

    } // namespace RegisterTransform_impl

    /*! \class register_layout_transform
    *  \brief  Invokes an in-register transform from one register layout to the other
    *  @tparam RegisterLayoutLhs Source register layout
    *  @tparam RegisterLayoutRhs Target register layout
    */
    template <typename RegisterLayoutLhs, typename RegisterLayoutRhs>
    using register_layout_transform
        = RegisterTransform_impl::register_layout_transform<RegisterLayoutLhs, RegisterLayoutRhs>;

    using register_layout_transform_nop = register_layout_transform<void, void>;

} // namespace rocWMMA

#endif // ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
