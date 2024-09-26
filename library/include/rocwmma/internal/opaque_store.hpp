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
#ifndef ROCWMMA_OPAQUE_STORE_HPP
#define ROCWMMA_OPAQUE_STORE_HPP

#include "io_traits.hpp"
#include "types.hpp"
#include "vector_iterator.hpp"

namespace rocwmma
{

    namespace detail
    {

        template <typename DataT, uint32_t VectorWidth>
        struct amdgcn_opaque_store
        {
            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(VecT<DataT, VectorWidth>),
                          "Cannot vectorize output");

            using StoreT = VecT<DataT, VectorWidth>;
            ROCWMMA_DEVICE static inline void
                exec(DataT* dataPtr, StoreT const& data, index_t offset = 0)
            {
                *reinterpret_cast<StoreT*>(&(dataPtr[offset])) = data;
            }
        };

    } // namespace detail

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataLayout,
              class MatrixLayout,
              uint32_t VectorWidth>
    struct OpaqueStore
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
        struct Traits
        {
            // Raw IO on unpacked register data.
            using Storer = detail::amdgcn_opaque_store<DataT, VectorWidth>;
            using StoreT = typename Storer::StoreT;
            using InputT = VecT<DataT, IOTraits::UnpackedSize>;
        };

        using StoreVecTraits = VecTraits<typename Traits::StoreT>;

        template <size_t Depth = 0, typename Iterator, typename StrideCounts, typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_right(DataT*         dataPtr,
                                                       Iterator&      in,
                                                       uint32_t       ldm,
                                                       StrideCounts&& strideCounts,
                                                       Strides2d&&    strides2d)
        {
            auto strideOffset = DataLayout::fromMatrixCoord(get<Depth>(strides2d), ldm);
            auto strideCount  = get<Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (VecTraits<decay_t<StrideCounts>>::size() - 1u))
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
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    unroll_right<Depth + 1>(dataPtr, in, ldm, strideCounts, strides2d);
                    dataPtr += strideOffset;
                }
            }
        }

        ROCWMMA_DEVICE static void
            exec(DataT* dataPtr, typename Traits::InputT const& data, uint32_t ldm)
        {
            // Arrange wave threads to starting matrix layout offsets.
            auto baseOffset2d = MatrixLayout::baseOffset();
            auto it           = makeVectorIterator<StoreVecTraits::size()>(data).begin();

            static_assert(decltype(it)::range() == IOTraits::IOCount,
                          "IOCount inconsistent with iterator range");

            // Make sure that the IOCount is consistent with the number of total strides
            static_assert(IOTraits::IOCount
                              == apply([](auto... items) { return (items * ...); },
                                       MatrixLayout::strideCounts()),
                          "IOCount inconsistent with total strides");

            unroll_right(dataPtr + DataLayout::fromMatrixCoord(baseOffset2d, ldm),
                         it,
                         ldm,
                         MatrixLayout::strideCounts(),
                         MatrixLayout::strides());
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_OPAQUE_STORE_HPP
