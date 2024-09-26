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
#ifndef ROCWMMA_COOP_STORE_HPP
#define ROCWMMA_COOP_STORE_HPP

#include "io_traits.hpp"
#include "opaque_store.hpp"
#include "types.hpp"
#include "utils.hpp"

namespace rocwmma
{

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataLayout,
              class MatrixLayout,
              uint32_t VectorWidth>
    struct CooperativeStore
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            enum : uint32_t
            {
                MaxSplit = IOTraits::IOCount
            };

            // Store implementation
            // Iteratively stores the entire block
            using Storer = detail::amdgcn_opaque_store<DataT, VectorWidth>;
            using StoreT = typename Storer::StoreT;

            // Block input vector
            using InputT = VecT<DataT, IOTraits::UnpackedSize>;
        };

        using StoreVecTraits = VecTraits<typename Traits::StoreT>;

        // Outer loop = index 0,
        // Inner loop = index N-1
        template <size_t Depth = 0, typename Iterator, typename StrideSpace, typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_right(DataT*        dataPtr,
                                                       Iterator&     in,
                                                       uint32_t      ldm,
                                                       StrideSpace&& strideCounts,
                                                       Strides2d&&   strides2d)
        {
            static_assert(VecTraits<decay_t<StrideSpace>>::size()
                              == VecTraits<decay_t<Strides2d>>::size(),
                          "Mismatched size");
            auto strideOffset = DataLayout::fromMatrixCoord(get<Depth>(strides2d), ldm);
            auto strideCount  = get<Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (VecTraits<decay_t<StrideSpace>>::size() - 1u))
            {
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    Traits::Storer::exec(dataPtr, *in);
                    dataPtr += strideOffset;
                    in++;
                }
            }
            // Recurse to the next nested layer
            else
            {
                for(int i = 0; i < strideCount; i++)
                {
                    unroll_right<Depth + 1>(dataPtr, in, ldm, strideCounts, strides2d);
                    dataPtr += strideOffset;
                }
            }
        }

        constexpr static uint32_t calcMaxWaves(uint32_t workItems, uint32_t waveCount)
        {
            return (workItems % waveCount == 0 ? waveCount
                                               : calcMaxWaves(workItems, waveCount / 2));
        };

        ROCWMMA_DEVICE static inline void exec(DataT*                         dataPtr,
                                               typename Traits::InputT const& data,
                                               uint32_t                       ldm,
                                               uint32_t                       waveIndex,
                                               uint32_t                       waveCount)
        {
            // Full fragment work
            constexpr auto strideSpace = MatrixLayout::strideCounts();
            constexpr auto strides     = MatrixLayout::strides();

            // Drop the VW strides for splitting (reduced stride space).
            constexpr auto strideSpaceR = pop_right(strideSpace);
            constexpr auto stridesR     = pop_right(strides);
            constexpr auto totalWorkItems
                = flatten_coord_left((strideSpaceR - 1u), strideSpaceR) + 1u;

            // Determine max waves possible.
            auto maxWaves = calcMaxWaves((uint32_t)totalWorkItems, (uint32_t)waveCount);

            // maxWaves is the maximum amount of waves split the work into.
            // For the rest of the waves, bail out
            if(__builtin_amdgcn_readfirstlane(waveIndex) >= maxWaves)
            {
                return; // bail
            }

            // Split the reduced stride space.
            auto workItemsPerWave = max(totalWorkItems / maxWaves, 1u);
            auto strideSpaceS     = inflate_coord_left(workItemsPerWave - 1u, strideSpaceR) + 1u;

            // Add back in the VW dimension, for the full stride
            // space of the current wave
            auto strideSpaceW = vector_cat(strideSpaceS, make_vector(get_last(strideSpace)));

            auto it = makeVectorIterator<StoreVecTraits::size()>(data).begin();

            // Align threads to starting matrix offset coordinates
            auto baseOffset = MatrixLayout::baseOffset();

            // Find current wave offset
            constexpr auto sum               = [](auto... items) { return (items + ...); };
            auto           currentWaveOffset = apply(
                sum, inflate_coord_left(waveIndex * workItemsPerWave, strideSpaceR) * stridesR);

            unroll_right(dataPtr + DataLayout::fromMatrixCoord(baseOffset + currentWaveOffset, ldm),
                         it,
                         ldm,
                         strideSpaceW,
                         strides);
        }

        template <uint32_t WaveCount>
        ROCWMMA_DEVICE static inline void exec(DataT*                         dataPtr,
                                               typename Traits::InputT const& data,
                                               uint32_t                       ldm,
                                               uint32_t                       waveIndex)
        {
            // Full fragment work
            constexpr auto strideSpace = MatrixLayout::strideCounts();
            constexpr auto strides     = MatrixLayout::strides();

            // Drop the VW strides for splitting (reduced stride space).
            constexpr auto strideSpaceR = pop_right(strideSpace);
            constexpr auto stridesR     = pop_right(strides);
            constexpr auto totalWorkItems
                = flatten_coord_left((strideSpaceR - 1u), strideSpaceR) + 1u;

            // Determine max waves possible.
            constexpr auto maxWaves = calcMaxWaves((uint32_t)totalWorkItems, (uint32_t)WaveCount);

            // maxWaves is the maximum amount of waves split the work into.
            // For the rest of the waves, bail out
            if constexpr(WaveCount != maxWaves)
            {
                if(__builtin_amdgcn_readfirstlane(waveIndex) >= maxWaves)
                {
                    return; // bail
                }
            }

            // Split the reduced stride space.
            constexpr auto workItemsPerWave = max(totalWorkItems / maxWaves, 1u);
            constexpr auto strideSpaceS
                = inflate_coord_left(workItemsPerWave - 1u, strideSpaceR) + 1u;

            // Add back in the VW dimension, for the full stride
            // space of the current wave
            constexpr auto strideSpaceW
                = vector_cat(strideSpaceS, make_vector(get_last(strideSpace)));

            // Alias the original frag due to smaller split size
            auto& dataR = (typename StoreVecTraits::template VecT<
                           DataT,
                           workItemsPerWave * StoreVecTraits::size()> const&)(data);
            auto  it    = makeVectorIterator<StoreVecTraits::size()>(dataR).begin();

            // Align threads to starting matrix offset coordinates
            auto baseOffset = MatrixLayout::baseOffset();

            // Find current wave offset
            constexpr auto sum               = [](auto... items) { return (items + ...); };
            auto           currentWaveOffset = apply(
                sum, inflate_coord_left(waveIndex * workItemsPerWave, strideSpaceR) * stridesR);

            unroll_right(dataPtr + DataLayout::fromMatrixCoord(baseOffset + currentWaveOffset, ldm),
                         it,
                         ldm,
                         strideSpaceW,
                         strides);
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_COOP_STORE_HPP
