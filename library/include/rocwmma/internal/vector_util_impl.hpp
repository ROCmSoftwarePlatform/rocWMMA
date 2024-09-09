/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_VECTOR_UTIL_IMPL_HPP
#define ROCWMMA_VECTOR_UTIL_IMPL_HPP

#include "blend.hpp"
#include "types.hpp"
#include "utility/vector.hpp"

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto concat(VecT<DataT, VecSize> const& v0,
                                                       VecT<DataT, VecSize> const& v1)
    {
        auto concat = [](auto&& idx, auto&& v0, auto&& v1) {
            constexpr auto Index = decay_t<decltype(idx)>::value;
            return (Index < VecSize) ? get<Index>(v0) : get<Index - VecSize>(v1);
        };

        return vector_generator<DataT, VecSize * 2u>()(concat, v0, v1);
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractLo(VecT<DataT, VecSize> const& v)
    {
        if constexpr(VecSize > 1)
        {
            auto lo = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(lo, v);
        }
        // Self-forwarding case
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractHi(VecT<DataT, VecSize> const& v)
    {
        if constexpr(VecSize > 1)
        {
            auto hi = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index + VecSize / 2u>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(hi, v);
        }
        else
        {
            return v;
        }
    }

} // namespace rocwmma

#include "pack_util.hpp"

namespace rocwmma
{
    template <typename DataT, uint32_t VecSize>
    ROCWMMA_HOST_DEVICE constexpr static inline auto extractEven(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes with minimum 2 packed vectors
        // Extract even elements only.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize >= 2u)
        {
            auto evens = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u) ? Blend::ExtractWordEven::exec(get<Index * 2u>(v),
                                                                          get<Index * 2u + 1u>(v))
                                           : Blend::ExtractByteEven::exec(get<Index * 2u>(v),
                                                                          get<Index * 2u + 1u>(v));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed   = PackUtil::paddedPack(v);
            auto result   = vector_generator<PackedT, PackedVecSize / 2u>()(evens, packed);
            return PackUtil::template paddedUnpack<VecSize / 2u>(result);
        }
        // General case:
        // Re-arrangement of dword+ data sizes isn't super costly and can
        // be achieved with a simple static shuffle.
        else if constexpr(VecSize > 1)
        {
            auto evens = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index * 2>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(evens, v);
        }
        // Forwarding case: vector size is 1
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto extractOdd(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes with minimum 2 packed vectors
        // Extract odd elements only.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize >= 2u)
        {
            auto odds = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u) ? Blend::ExtractWordOdd::exec(get<Index * 2u>(v),
                                                                         get<Index * 2u + 1u>(v))
                                           : Blend::ExtractByteOdd::exec(get<Index * 2u>(v),
                                                                         get<Index * 2u + 1u>(v));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed   = PackUtil::paddedPack(v);
            auto result   = vector_generator<PackedT, PackedVecSize / 2u>()(odds, packed);
            return PackUtil::template paddedUnpack<VecSize / 2u>(result);
        }
        // General case:
        // Re-arrangement of dword+ data sizes isn't super costly and can
        // be achieved with a simple static shuffle.
        else if constexpr(VecSize > 1)
        {
            auto odds = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return get<Index * 2 + 1>(v);
            };

            return vector_generator<DataT, VecSize / 2u>()(odds, v);
        }
        // Forwarding case: vector size is 1
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto reorderEvenOdd(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes, maximum one packed vector.
        // Extract even elements followed by odd elements.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize == 1)
        {
            auto evenOdds = [](auto&& idx, auto&& v) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u)
                           ? Blend::ExtractWordEvenOdd::exec(get<Index>(v), get<Index>(v))
                           : Blend::ExtractByteEvenOdd::exec(get<Index>(v), get<Index>(v));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed   = PackUtil::paddedPack(v);
            auto result   = vector_generator<PackedT, PackedVecSize>()(evenOdds, packed);
            return PackUtil::template paddedUnpack<VecSize>(result);
        }
        // General case: Concatenate evens and odds
        else if constexpr(VecSize > 1)
        {
            return concat(extractEven(v), extractOdd(v));
        }
        // Forwarding case: return self
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto reorderOddEven(VecT<DataT, VecSize> const& v)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes, maximum one packed vector.
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;
        if constexpr(ElementSize < 4u && PackedVecSize <= 1)
        {
            using PackedT = typename PackTraits::PackedT;

            // Exactly one packed vector
            if constexpr(PackedVecSize == 1)
            {
                auto oddEvens = [](auto&& idx, auto&& v) {
                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return (ElementSize == 2u)
                               ? Blend::ExtractWordOddEven::exec(get<Index>(v), get<Index>(v))
                               : Blend::ExtractByteOddEven::exec(get<Index>(v), get<Index>(v));
                };

                // Pack, extract and unpack
                auto packed = PackUtil::paddedPack(v);
                auto result = vector_generator<PackedT, PackedVecSize>()(oddEvens, packed);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
            // Corner case: Swap bytes
            else if constexpr(ElementSize == 1 && VecSize == 2)
            {
                auto oddEvens = [](auto&& idx, auto&& v) {
                    // Manually swap bytes
                    using SwapBytes = Blend::Driver<BlendImpl::Ops::PermByte<1u, 0u, 3u, 2u>>;

                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return SwapBytes::exec(get<Index>(v), get<Index>(v));
                };

                // Pack, extract and unpack
                auto packed = PackUtil::paddedPack(v);
                auto result = vector_generator<PackedT, 1u>()(oddEvens, packed);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
            // ElementSize = 1, 2 VecSize = 1
            else
            {
                return v;
            }
        }
        // General case: Concatenate evens and odds
        else if constexpr(VecSize > 1)
        {
            return concat(extractOdd(v), extractEven(v));
        }
        // Forwarding case: return self
        else
        {
            return v;
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto zip(VecT<DataT, VecSize> const& v0,
                                                    VecT<DataT, VecSize> const& v1)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = max(VecSize / PackTraits::PackRatio, 1u);
        if constexpr(ElementSize < 4u)
        {
            auto zip = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u) ? Blend::ZipWord::exec(get<Index>(v0), get<Index>(v1))
                                           : Blend::ZipByte::exec(get<Index>(v0), get<Index>(v1));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed0  = PackUtil::paddedPack(v0);
            auto packed1  = PackUtil::paddedPack(v1);
            auto result   = vector_generator<PackedT, PackedVecSize>()(zip, packed0, packed1);
            return PackUtil::template paddedUnpack<VecSize>(result);
        }
        else
        {
            auto zip = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (Index % 2u == 0u) ? get<Index>(v0) : get<Index>(v1);
            };

            return vector_generator<DataT, VecSize>()(zip, v0, v1);
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackLo(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = max(VecSize / PackTraits::PackRatio, 1u);

        // The optimization should only be applied on a pair of register. So v0 and v1
        // should not be larger than a register
        if constexpr(ElementSize < 4u && PackedVecSize <= 1)
        {
            auto unpackLo = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (ElementSize == 2u)
                           ? Blend::UnpackWordLo::exec(get<Index>(v0), get<Index>(v1))
                           : Blend::UnpackByteLo::exec(get<Index>(v0), get<Index>(v1));
            };

            // Pack, extract and unpack
            using PackedT = typename PackTraits::PackedT;
            auto packed0  = PackUtil::paddedPack(v0);
            auto packed1  = PackUtil::paddedPack(v1);
            auto result   = vector_generator<PackedT, PackedVecSize>()(unpackLo, packed0, packed1);
            return PackUtil::template paddedUnpack<VecSize>(result);
        }
        else
        {
            auto unpackLo = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto Index = decay_t<decltype(idx)>::value;
                return (Index % 2u == 0u) ? get<Index / 2u>(v0) : get<Index / 2u>(v1);
            };

            return vector_generator<DataT, VecSize>()(unpackLo, v0, v1);
        }
    }

    template <typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto unpackHi(VecT<DataT, VecSize> const& v0,
                                                         VecT<DataT, VecSize> const& v1)
    {
        using PackUtil   = PackUtil<DataT>;
        using PackTraits = typename PackUtil::Traits;

        // Special case: Sub-dword data sizes
        // Optimize data-reorder with cross-lane ops.
        constexpr auto ElementSize   = sizeof(DataT);
        constexpr auto PackedVecSize = VecSize / PackTraits::PackRatio;

        // The optimization should only be applied on a pair of register. So v0 and v1
        // should not be larger than a register
        if constexpr(ElementSize < 4u && PackedVecSize <= 1)
        {
            if constexpr(ElementSize < 2u && PackedVecSize == 0)
            {
                auto unpackHi = [](auto&& idx, auto&& v0, auto&& v1) {
                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return Blend::UnpackByte3BCast::exec(get<Index>(v0), get<Index>(v1));
                };

                // Pack, extract and unpack
                using PackedT = typename PackTraits::PackedT;
                auto packed0  = PackUtil::paddedPack(v0);
                auto packed1  = PackUtil::paddedPack(v1);
                auto result   = vector_generator<PackedT, 1u>()(unpackHi, packed0, packed1);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
            else
            {
                auto unpackHi = [](auto&& idx, auto&& v0, auto&& v1) {
                    constexpr auto Index = decay_t<decltype(idx)>::value;
                    return (ElementSize == 2u)
                               ? Blend::UnpackWordHi::exec(get<Index>(v0), get<Index>(v1))
                               : Blend::UnpackByteHi::exec(get<Index>(v0), get<Index>(v1));
                };

                // Pack, extract and unpack
                using PackedT = typename PackTraits::PackedT;
                auto packed0  = PackUtil::paddedPack(v0);
                auto packed1  = PackUtil::paddedPack(v1);
                auto result   = vector_generator<PackedT, 1u>()(unpackHi, packed0, packed1);
                return PackUtil::template paddedUnpack<VecSize>(result);
            }
        }
        else
        {
            auto unpackHi = [](auto&& idx, auto&& v0, auto&& v1) {
                constexpr auto startIdx = VecSize / 2u;
                constexpr auto Index    = decay_t<decltype(idx)>::value;
                return (Index % 2u == 0u) ? get<startIdx + Index / 2u>(v0)
                                          : get<startIdx + Index / 2u>(v1);
            };

            return vector_generator<DataT, VecSize>()(unpackHi, v0, v1);
        }
    }

    template <uint32_t GroupSize, typename DataT, uint32_t VecSize>
    ROCWMMA_DEVICE constexpr static inline auto interleave(VecT<DataT, VecSize> const& v0)
    {
        // Interleave groups
        auto offset = [](auto&& idx, auto&& v0) {
            constexpr auto Index   = decay_t<decltype(idx)>::value;
            constexpr auto Offset0 = Index * GroupSize;
            constexpr auto Offset1 = Index / (VecSize / GroupSize);
            return get<(Offset0 + Offset1) % VecSize>(v0);
        };

        return vector_generator<DataT, VecSize>()(offset, v0);
    }

} // namespace rocwmma

#endif // ROCWMMA_VECTOR_UTIL_IMPL_HPP
