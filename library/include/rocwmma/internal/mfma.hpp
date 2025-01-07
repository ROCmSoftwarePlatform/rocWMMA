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
#ifndef ROCWMMA_MFMA_HPP
#define ROCWMMA_MFMA_HPP

#include "config.hpp"
#include "vector.hpp"
#include "vector_iterator.hpp"

#include "mfma_impl.hpp"

namespace rocwmma
{
    // MFMA interface
    template <typename InputT,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename Enabler = void>
    struct Mfma : public detail::amdgcn_mfma<InputT, ComputeT, BlockM, BlockN>
    {
    };

    // Unlock the mfma backend only on MI cards
    template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK>
    struct Mfma<InputT,
                ComputeT,
                BlockM,
                BlockN,
                BlockK,
                enable_if_t<ROCWMMA_ARCH_GFX9 && (BlockM == BlockN)>>
    {
        // Full-fragment IO traits
        using IOTraitsA   = IOTraits<BlockM, BlockK, InputT>;
        using IOTraitsB   = IOTraits<BlockK, BlockN, InputT>;
        using IOTraitsAcc = IOTraits<BlockM, BlockN, ComputeT>;

        // Functional
        using MFMA = detail::amdgcn_mfma<InputT, ComputeT, BlockM, BlockN>;

        // Per-MFMA iterative vector requirements
        using VecTraitsA = VecTraits<typename MFMA::Traits::ARegsT>;
        using VecTraitsB = VecTraits<typename MFMA::Traits::BRegsT>;
        using VecTraitsC = VecTraits<typename MFMA::Traits::CRegsT>;
        using VecTraitsD = VecTraits<typename MFMA::Traits::DRegsT>;

        struct Traits
        {
            enum : uint32_t
            {
                MfmaCount = BlockK / MFMA::Traits::KPerMfma,
                MinK      = MFMA::Traits::KPerMfma,
            };

            // Create full-fragment vector sizes
            using ARegsT = typename VecTraitsA::template VecT<typename VecTraitsA::DataT,
                                                              MfmaCount * VecTraitsA::size()>;
            using BRegsT = typename VecTraitsB::template VecT<typename VecTraitsA::DataT,
                                                              MfmaCount * VecTraitsB::size()>;
            using CRegsT = typename VecTraitsC::template VecT<>;
            using DRegsT = typename VecTraitsD::template VecT<>;

            // Sanity checks
            static_assert(BlockK >= MinK, "BlockK is not a minimum of MinK");
            static_assert(BlockK % MinK == 0, "BlockK is not a multiple of MinK");

            // A / B  and C / D types must match
            static_assert(is_same<typename VecTraitsA::DataT, typename VecTraitsB::DataT>::value,
                          "A and B registers must be of same type");
            static_assert(is_same<typename VecTraitsC::DataT, typename VecTraitsD::DataT>::value,
                          "C and D registers must be of same type");

            // Full fragment counts must match packed IO counts
            // MFMA expects packed elements
            static_assert(VecTraits<ARegsT>::size() == IOTraitsA::PackedSize,
                          "Unexpected packed vector size for A");
            static_assert(VecTraits<BRegsT>::size() == IOTraitsB::PackedSize,
                          "Unexpected packed vector size for B");
            static_assert(VecTraits<CRegsT>::size() == IOTraitsAcc::PackedSize,
                          "Unexpected packed vector size for C");
            static_assert(VecTraits<DRegsT>::size() == IOTraitsAcc::PackedSize,
                          "Unexpected packed vector size for D");
        };

        ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                               typename Traits::BRegsT const& regsB,
                                               typename Traits::CRegsT const& regsC) ->
            typename Traits::DRegsT
        {
            typename Traits::DRegsT result = regsC;

            // Iterate over MFMA input requirements
            auto aIt = makeVectorIterator<VecTraitsA::size()>(regsA).begin();
            auto bIt = makeVectorIterator<VecTraitsB::size()>(regsB).begin();

            // Accumulate over MFMA count
#pragma unroll
            for(unsigned i = 0; i < Traits::MfmaCount; i++)
            {
                result = MFMA::exec(*aIt, *bIt, result);
                aIt++;
                bIt++;
            }
            return result;
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_MFMA_HPP
