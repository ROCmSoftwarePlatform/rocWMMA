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
#ifndef ROCWMMA_OPAQUE_LOAD_HPP
#define ROCWMMA_OPAQUE_LOAD_HPP

#include "io_traits.hpp"
#include "tuple.hpp"
#include "types.hpp"
#include "vector_iterator.hpp"

namespace rocwmma
{

    namespace detail
    {

        template <typename DataT, uint32_t VectorWidth>
        struct amdgcn_opaque_load
        {
            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(VecT<DataT, VectorWidth>),
                          "Cannot vectorize input");

            using LoadT = VecT<DataT, VectorWidth>;

            ROCWMMA_DEVICE static inline void
                exec(LoadT& data, DataT const* dataPtr, index_t offset = 0)
            {
                data = *reinterpret_cast<LoadT const*>(&(dataPtr[offset]));
            }
        };

    } // namespace detail

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              class DataLayout,
              class MatrixLayout,
              uint32_t VectorWidth,
              bool     Debug = false>
    struct OpaqueLoad
    {
        using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;

        struct Traits
        {
            // Raw IO on unpacked register data.
            using Loader  = detail::amdgcn_opaque_load<DataT, VectorWidth>;
            using LoadT   = typename Loader::LoadT;
            using OutputT = VecT<DataT, IOTraits::UnpackedSize>;
        };

        using LoadVecTraits = VecTraits<typename Traits::LoadT>;

        // Outer loop = index 0,
        // Inner loop = index N-1
        template <size_t Depth = 0, typename Iterator, typename StrideCounts, typename Strides2d>
        ROCWMMA_DEVICE static inline auto unroll_right(Iterator&      out,
                                                       DataT const*   dataPtr,
                                                       uint32_t       ldm,
                                                       StrideCounts&& strideCounts,
                                                       Strides2d&&    strides2d)
        {
            auto strideOffset = DataLayout::fromMatrixCoord(get<Depth>(strides2d), ldm);
            auto strideCount  = get<Depth>(strideCounts);

            // Last depth layer will invoke the load
            if constexpr(Depth == (VecTraits<decay_t<StrideCounts>>::size() - 1u))
            {
                if constexpr(Debug)
                {
                    printf("Depth: %d, StrideCount: %d\n", Depth, get<Depth>(strideCounts));
                    printf("StrideX: %d, StrideY: %d\n",
                           get<0>(get<Depth>(strides2d)),
                           get<1>(get<Depth>(strides2d)));
                    printf("Executing!\n");
                }
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    Traits::Loader::exec(*out, dataPtr);
                    dataPtr += strideOffset;
                    out++;
                }
            }
            // Recurse to the next nested layer
            else
            {
                if constexpr(Debug)
                {
                    printf("Depth: %d, StrideCount: %d\n", Depth, get<Depth>(strideCounts));
                    printf("StrideX: %d, StrideY: %d\n",
                           get<0>(get<Depth>(strides2d)),
                           get<1>(get<Depth>(strides2d)));
                    printf("Recursing!\n");
                }
#pragma unroll
                for(int i = 0; i < strideCount; i++)
                {
                    unroll_right<Depth + 1>(out, dataPtr, ldm, strideCounts, strides2d);
                    dataPtr += strideOffset;
                }
            }
        }

        ROCWMMA_DEVICE static void
            exec(typename Traits::OutputT& data, DataT const* dataPtr, uint32_t ldm)
        {
            //MatrixLayout::debug();
            // Arrange wave threads to starting matrix layout offsets.
            auto baseOffset2d = MatrixLayout::baseOffset();
            auto it           = makeVectorIterator<LoadVecTraits::size()>(data).begin();

            static_assert(decltype(it)::range() == IOTraits::IOCount,
                          "IOCount inconsistent with iterator range");

            // Make sure that the IOCount is consistent with the number of total strides
            static_assert(IOTraits::IOCount
                              == apply([](auto... items) { return (items * ...); },
                                       MatrixLayout::strideCounts()),
                          "IOCount inconsistent with total strides");

            // Unroll loading in each strided dimension
            unroll_right(it,
                         dataPtr + DataLayout::fromMatrixCoord(baseOffset2d, ldm),
                         ldm,
                         MatrixLayout::strideCounts(),
                         MatrixLayout::strides());
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_OPAQUE_LOAD_HPP
