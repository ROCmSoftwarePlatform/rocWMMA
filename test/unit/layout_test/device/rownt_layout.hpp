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

#ifndef ROCWMMA_DEVICE_ROWNT_LAYOUT_HPP
#define ROCWMMA_DEVICE_ROWNT_LAYOUT_HPP

#include "unit_test_traits.hpp"
#include <rocwmma/internal/io_layout.hpp>
#include <rocwmma/internal/io_traits.hpp>
#include <rocwmma/internal/mapping_util.hpp>

namespace rocwmma
{

    template <uint32_t BlockM, uint32_t BlockN, typename DataT, typename DataLayoutT>
    __global__ void RowNTLayout(uint32_t     m,
                                uint32_t     n,
                                DataT const* in,
                                DataT*       out,
                                uint32_t     ld,
                                DataT        param1,
                                DataT        param2)
    {
        if constexpr(FragSize_guard<BlockM,
                                    BlockN,
                                    DataT,
                                    DataLayoutT,
                                    Constants::AMDGCN_WAVE_SIZE,
                                    Constants::AMDGCN_CURRENT_ARCH_ID>::enable())
        {
            enum : uint32_t
            {
                BlockHeight = BlockM,
                BlockWidth  = BlockN,

                BlockDim = BlockN,
                KDim     = BlockM,

                MaxVectorWidth
                = std::is_same_v<DataT, float64_t>
                      ? 1
                      : detail::MaxVWSelector<matrix_b, BlockM, BlockN, DataT, DataLayoutT>::Result,
                VectorWidth = std::is_same_v<DataLayoutT, col_major> ? MaxVectorWidth : 1,
            };

            using IOTraits = IOTraits<BlockDim, KDim, DataT, VectorWidth>;
            using LayoutT  = typename LayoutProfile::
                RowNT<BlockDim, KDim, DataT, DataLayoutT, VectorWidth, MaxVectorWidth>::
                    MatrixLayout;
            using Mapping = MappingUtil<BlockHeight, BlockWidth, DataT, DataLayoutT>;

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
                for(uint32_t j = 0; j < VectorWidth; j++)
                {
                    auto index = (get<MajorIndex>(matrixCoord) * ld + get<MinorIndex>(matrixCoord))
                                 + Mapping::dataOffset(baseOffset, ld) + j;
                    out[index] = in[index];
                }
                baseOffset += LayoutT::incrementalOffset(i);
            }
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_ROWNT_LAYOUT_HPP
