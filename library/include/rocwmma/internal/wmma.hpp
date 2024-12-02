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
#ifndef ROCWMMA_WMMA_HPP
#define ROCWMMA_WMMA_HPP

#include "permute.hpp"
#include "vector.hpp"
#include "vector_iterator.hpp"

#include "wmma_impl.hpp"

namespace rocwmma
{
    // Wmma interface
    template <typename InputT,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename Enabler = void>
    struct Wmma : public detail::amdgcn_wmma<InputT, ComputeT, BlockM, BlockN>
    {
        template <typename InputARegsT, typename InputBRegsT, typename InputCRegsT>
        ROCWMMA_DEVICE static inline auto
            exec(InputARegsT const& regsA, InputBRegsT const& regsB, InputCRegsT const& regsC)
        {
            return regsC;
        }
    };

#if ROCWMMA_ARCH_GFX11 || ROCWMMA_ARCH_GFX12

    // Unlock the WMMA builtins for gfx11 cards
    // Supported Input/Compute types:
    // float16_t / float16_t
    // float16_t / float32_t
    // hfloat16_t / float16_t
    // hfloat16_t / float32_t
    // bfloat16_t / bfloat16_t
    // bfloat16_t / float32_t
    // int8_t / int32_t
    // Supported block sizes (M, N) = 16
    template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct Wmma<
        InputT,
        ComputeT,
        BlockM,
        BlockN,
        BlockK,
        typename enable_if<
            ((is_same<InputT, float16_t>::value && is_same<ComputeT, float16_t>::value)
             || (is_same<InputT, float16_t>::value && is_same<ComputeT, float32_t>::value)
#if !ROCWMMA_NO_HALF
             || (is_same<InputT, hfloat16_t>::value && is_same<ComputeT, hfloat16_t>::value)
             || (is_same<InputT, hfloat16_t>::value && is_same<ComputeT, float32_t>::value)
#endif // !ROCWMMA_NO_HALF
             || (is_same<InputT, bfloat16_t>::value && is_same<ComputeT, bfloat16_t>::value)
             || (is_same<InputT, bfloat16_t>::value && is_same<ComputeT, float32_t>::value)
             || (is_same<InputT, int8_t>::value && is_same<ComputeT, int32_t>::value)
             || (is_same<InputT, float8_t>::value && is_same<ComputeT, float32_t>::value)
             || (is_same<InputT, bfloat8_t>::value && is_same<ComputeT, float32_t>::value))
            && (BlockM == 16) && (BlockN == 16) && (BlockK >= 16) // 16 block size only
            >::type>
    {
        // Functional backend
        using WMMA     = detail::amdgcn_wmma<InputT, ComputeT, BlockM, BlockN>;
        using PackUtil = PackUtil<ComputeT>;

        // Full-fragment IO traits
        using IOTraitsA   = IOTraits<BlockM, BlockK, InputT>;
        using IOTraitsB   = IOTraits<BlockK, BlockN, InputT>;
        using IOTraitsAcc = IOTraits<BlockM, BlockN, ComputeT>;

        struct Traits
        {
            enum : uint32_t
            {
                WmmaCount         = BlockK / WMMA::Traits::KPerWmma,
                MinK              = WMMA::Traits::KPerWmma,
                InputSizeModifier = ROCWMMA_ARCH_GFX11 ? 2u : 1u
            };

            // Sanity checks
            static_assert(BlockK >= MinK, "BlockK is not a minimum of MinK");
            static_assert(BlockK % MinK == 0, "BlockK is not a multiple of MinK");
        };

        // Per-WMMA iterative vector requirements
        using VecTraitsA = VecTraits<typename WMMA::Traits::ARegsT>;
        using VecTraitsB = VecTraits<typename WMMA::Traits::BRegsT>;
        using VecTraitsC = VecTraits<typename WMMA::Traits::CRegsT>;
        using VecTraitsD = VecTraits<typename WMMA::Traits::DRegsT>;

        // amdgcn backend requires duplicated packed inputs A / B, and unpacked accumulator
        static_assert(VecTraitsA::size() * Traits::WmmaCount
                          == IOTraitsA::PackedSize * Traits::InputSizeModifier,
                      "WMMA backend input size mismatch");
        static_assert(VecTraitsB::size() * Traits::WmmaCount
                          == IOTraitsB::PackedSize * Traits::InputSizeModifier,
                      "WMMA backend input size mismatch");
        static_assert(VecTraitsC::size() == IOTraitsAcc::UnpackedSize,
                      "WMMA backend input size mismatch");

        template <typename InputARegsT, typename InputBRegsT, typename InputCRegsT>
        ROCWMMA_DEVICE static inline auto
            exec(InputARegsT const& regsA, InputBRegsT const& regsB, InputCRegsT const& regsC)
        {
            // Inputs from outside will come in as fully packed
            static_assert(VecTraits<InputARegsT>::size() == VecTraitsA::size() * Traits::WmmaCount,
                          "WMMA input size mismatch");
            static_assert(VecTraits<InputBRegsT>::size() == VecTraitsA::size() * Traits::WmmaCount,
                          "WMMA input size mismatch");
            static_assert(VecTraits<InputCRegsT>::size() == IOTraitsAcc::PackedSize,
                          "WMMA input size mismatch");

            // WMMA accumulator operates on unpacked, padded data in separate 32b elements.
            // In the case of f16, what needs to happen is extend each unpacked element to 32b wide
            // and shift the 16b data to the correct spot (determined by the WMMA backend).
            // The nasty bit is that due of the extended 32b element size, the final accumulation vector
            // is masqueraded as a 'packed' type, but with the same vector size as unpacked.
            auto accum = PackUtil::template pad<WMMA::Traits::AccumBits>(PackUtil::unpack(regsC));

            // Iterate over packed WMMA inputs
            auto const aIt = makeVectorIterator<VecTraitsA::size()>(regsA).begin();
            auto const bIt = makeVectorIterator<VecTraitsB::size()>(regsB).begin();

            // Accumulate over WMMA count
#pragma unroll
            for(uint32_t i = 0; i < Traits::WmmaCount; i++)
            {
                accum = WMMA::exec(*aIt, *bIt, accum);
                aIt++;
                bIt++;
            }

            return PackUtil::pack(PackUtil::template unpad<WMMA::Traits::AccumBits>(accum));
        }
    };

#endif // ROCWMMA_ARCH_GFX11

} // namespace rocwmma

#endif // ROCWMMA_WMMA_HPP