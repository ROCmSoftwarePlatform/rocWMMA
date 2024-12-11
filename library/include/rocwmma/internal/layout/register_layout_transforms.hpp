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

                // Non-interleaved AOS to SOA
                if constexpr(traits_lhs::Format == Format::AOS && traits_rhs::Format == Format::SOA)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                    return Transforms::
                        AosToSoa<storage_traits::BlockDim, storage_traits::MaxVectorWidth>::exec(
                            forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::SOA
                                  && traits_rhs::Format == Format::AOS)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                    return Transforms::
                        SoaToAos<storage_traits::BlockDim, storage_traits::MaxVectorWidth>::exec(
                            forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::AOS_INT
                                  && traits_rhs::Format == Format::SOA_INT)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                    return interleave<1u, storage_traits::DimPerThread>(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::SOA_INT
                                  && traits_rhs::Format == Format::AOS_INT)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                    return interleave<1u, storage_traits::KPerThread>(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::AOS_INT
                                  && traits_rhs::Format == Format::ACC_INT_A_MAJOR)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                    return interleave<1u, storage_traits::KPerThread>(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::SOA_INT
                                  && traits_rhs::Format == Format::ACC_INT_A_MAJOR)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;

                    return interleave<1u, 4u>(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::ACC_INT_A_MAJOR
                                  && traits_rhs::Format == Format::AOS_INT)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                    return interleave<1u, 4u>(forward<VecT>(v));
                }
                else if constexpr(traits_lhs::Format == Format::ACC_INT_A_MAJOR
                                  && traits_rhs::Format == Format::SOA_INT)
                {
                    using storage_traits
                        = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                    return interleave<1u, storage_traits::KPerThread>(forward<VecT>(v));
                }
                else if constexpr((traits_lhs::Format == Format::SOA)
                               && (traits_rhs::Format == Format::WMMA_INPUT_GFX11))
                {
                    // Input is unpacked
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
                else if constexpr((traits_lhs::Format == Format::AOS)
                               && (traits_rhs::Format == Format::WMMA_INPUT_GFX11))
                {

                    //auto toSOA =
                    // Input is unpacked
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
                else if constexpr((traits_lhs::Format == Format::SOA)
                               && (traits_rhs::Format == Format::WMMA_ACC_GFX11))
                {
                    // SOA format to wmma acc padded accumulator (gfx11).
                    // f16 -> padded to f32 in lower 16
                    // f32 -> nop
                    using PackUtil = PackUtil<typename traits_lhs::DataT>;
                    auto accum = PackUtil::unpack(PackUtil::template pad<>(v));
                    return accum; // Return by copy
                }
                else if constexpr((traits_lhs::Format == Format::WMMA_ACC_GFX11)
                               && (traits_rhs::Format == Format::SOA))
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
