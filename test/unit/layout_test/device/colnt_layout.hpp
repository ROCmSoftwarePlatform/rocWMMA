/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021-2023 Advanced Micro Devices, Inc.
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

#ifndef ROCWMMA_DEVICE_COLNT_LAYOUT_HPP
#define ROCWMMA_DEVICE_COLNT_LAYOUT_HPP

#include "unit_test_traits.hpp"
#include <rocwmma/internal/io_traits.hpp>
#include <rocwmma/internal/layout.hpp>
#include <rocwmma/internal/mapping_util.hpp>

namespace rocwmma
{
    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayoutT,
              typename std::enable_if_t<
                  FragSize_guard<BlockM,
                                 BlockN,
                                 DataT,
                                 DataLayoutT,
                                 Constants::AMDGCN_WAVE_SIZE,
                                 Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void ColNTLayout(uint32_t     m,
                                uint32_t     n,
                                DataT const* in,
                                DataT*       out,
                                uint32_t     ld,
                                DataT        param1,
                                DataT        param2)
    {
        enum : uint32_t
        {
            MaxVectorWidth
            = detail::MaxVWSelector<matrix_a, BlockM, BlockN, DataT, DataLayoutT>::Result,
            VectorWidth = std::is_same_v<DataLayoutT, row_major> ? MaxVectorWidth : 1
        };

        using IOTraits = IOTraits<BlockM, BlockN, DataT, VectorWidth>;
        using LayoutT
            = MatrixLayout::ColNT<BlockM, BlockN, DataT, DataLayoutT, VectorWidth, MaxVectorWidth>;
        using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayoutT>;

        auto baseOffset  = LayoutT::baseOffset();
        auto iocount     = IOTraits::IOCount;
        auto matrixCoord = Mapping::matrixCoord();

        enum : uint32_t
        {
            MajorIndex = std::is_same_v<DataLayoutT, row_major> ? 0 : 1,
            MinorIndex = std::is_same_v<DataLayoutT, row_major> ? 1 : 0
        };

        for(uint32_t i = 0; i < iocount; ++i)
        {
            for(int j = 0; j < VectorWidth; j++)
            {
                auto index = (get<MajorIndex>(matrixCoord) * ld + get<MinorIndex>(matrixCoord))
                             + Mapping::dataOffset(baseOffset, ld) + j;
                out[index] = in[index];
            }
            baseOffset += LayoutT::incrementalOffset(i);
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayoutT,
              typename std::enable_if_t<
                  !FragSize_guard<BlockM,
                                  BlockN,
                                  DataT,
                                  DataLayoutT,
                                  Constants::AMDGCN_WAVE_SIZE,
                                  Constants::AMDGCN_CURRENT_ARCH_ID>::enable()>* = nullptr>
    __global__ void ColNTLayout(uint32_t     m,
                                uint32_t     n,
                                DataT const* in,
                                DataT*       out,
                                uint32_t     ld,
                                DataT        param1,
                                DataT        param2)
    {
    }
} // namespace rocwmma

#endif // ROCWMMA_DEVICE_COLNT_LAYOUT_HPP
