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
#ifndef ROCWMMA_MATRIX_LAYOUT_IMPL_HPP
#define ROCWMMA_MATRIX_LAYOUT_IMPL_HPP

#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{

    // Implementations for the MatrixLayout classes
    namespace MatrixLayout
    {
        /* Pattern that maps threads contiguously to matrix columns and assumes
        * that VW will be mapped orthogonally to the column.
        * This pattern considers VW up to MaxVW, BlockDim <= 64 and BlockDim > 64.
        *
        * Iterative thread stride cycles (same for all threads):
        *   Fill MaxVW => Fill BlockK => Fill BlockDim
        *
        * Example:
        *  BlockDim = 128   BlockK = 16
        *  MaxVW = 4       VW = 1
        *
        *  BlockDim Stride Count = 2, BlockDimStride = (64, 0)
        *  BlockK   Stride Count = 4, BlockKStride   = (0,  4)
        *  VW       Stride Count = 4, VWStride       = (0,  1)
        *
        *  Stride mapping (BlockDim, BlockK, VW)
        *  C_n = Matrix column
        *  i_n = cumulative iteration
        *
        *   kDim --------->
        *                     VW Stride
        *   BlockDim          |--1--|
        *   |                 |-- BlockK Stride = 4 --|
        *   |                 i0(0,0,0)   i2(0,0,2)   i4(0,1,0)   i6(0,1,2)         i14(0,3,2)
        *   |            --   v_____ _____v_____ _____v_____ _____v_____ _____      v_____  _____
        *   v            |    |     |     |     |     |     |     |     |     |     |     ||     |
        *                |    |     |     |     |     |     |     |     |     |     |     ||     |
        *       BlockDim 64   | C0  |  C1 |  C2 |  C3 |  C4 |  C5 | C6  | C7  | ... | C14 || C15 |
        *        Stride  |    |     |     |     |     |     |     |     |     |     |     ||     |
        *                --   |_____|_____|_____|_____|_____|_____|_____|_____|     |_____||_____|
        *                     i16(1,0,0)  i18(1,0,2)  i20(1,1,0)  i22(1,1,2)        i30(1,3,2)
        *                     v_____ _____v_____ _____v_____ _____v_____ _____      v_____  _____
        *                     |     |     |     |     |     |     |     |     |     |     ||     |
        *                     |     |     |     |     |     |     |     |     |     |     ||     |
        *                     | C0  |  C1 |  C2 |  C3 | C4  | C5  | C6  | C7  | ... | C14 || C15 |
        *                     |     |     |     |     |     |     |     |     |     |     ||     |
        *                     |_____|_____|_____|_____|_____|_____|_____|_____|     |_____||_____|
        *                     ^(128, 0)                                                           ^(BlockDim, BlockK)
        *   ...                                          ...
        *
        * Register file (for all VectorWidths = [1, MaxVectorWidth]):
        *
        * Elements 0..............63
        *           ______________
        *  Reg0    |  C0 [63:0]    |
        *  Reg1    |  C1 [63:0]    |
        *  Reg2    |  C2 [63:0]    |
        *  ...       ...
        *  Reg15   |  C15[63:0]    |
        *  Reg16   |  C0 [127:64]  |
        *  ...       ...
        *  Reg31   |  C15 [127:64] |
        }*/

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColOrthoVW
        {
            using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
            struct Traits
            {
                enum : uint32_t
                {
                    // Number of threads per wave
                    WaveSize = IOTraits::ThreadsPerIO,

                    // Strides
                    BlockDimStride_X = min(BlockDim, WaveSize),
                    BlockDimStride_Y = 0u,

                    BlockKStride_X = 0u,
                    BlockKStride_Y = WaveSize * MaxVectorWidth / BlockDimStride_X,

                    VWStride_X = 0u,
                    VWStride_Y = VectorWidth,

                    // Stride space
                    BlockDimSegs = BlockDim / BlockDimStride_X,
                    BlockKSegs   = BlockK / BlockKStride_Y,
                    VWSegs       = MaxVectorWidth / VWStride_Y,
                };

                static_assert(BlockDim >= (uint32_t)Traits::BlockDimStride_X,
                              "BlockDim must be larger than BlockDimStride_X");
                static_assert(BlockDim % (uint32_t)Traits::BlockDimStride_X == 0,
                              "BlockDim must be a multiple of BlockDimStride_X");
                static_assert(BlockK >= (uint32_t)Traits::BlockKStride_Y,
                              "BlockK must be larger than BlockKStride_Y");
                static_assert(BlockK % (uint32_t)Traits::BlockKStride_Y == 0,
                              "BlockK must be a multiple of BlockKStride_Y");
                static_assert(MaxVectorWidth >= (uint32_t)Traits::VWStride_Y,
                              "MaxVectorWidth must larger than VWStride_Y");
                static_assert(MaxVectorWidth % (uint32_t)Traits::VWStride_Y == 0,
                              "MaxVectorWidth must be a multiple of VWStride_Y");

                // Orthogonal layout, coordinates are reversed
                using OrthoLayout
                    = RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;

                using MatrixCoordT = Coord2d;
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return make_vector((uint32_t)Traits::BlockDimSegs, // BlockDim Segments
                                   (uint32_t)Traits::BlockKSegs, // BlockK Segments
                                   (uint32_t)Traits::VWSegs); // VW Segments
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(
                    make_coord2d((uint32_t)Traits::BlockDimStride_X,
                                 (uint32_t)Traits::BlockDimStride_Y),
                    make_coord2d((uint32_t)Traits::BlockKStride_X,
                                 (uint32_t)Traits::BlockKStride_Y),
                    make_coord2d((uint32_t)Traits::VWStride_X, (uint32_t)Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
            {
                if constexpr((uint32_t)Traits::BlockDimStride_X >= (uint32_t)Traits::WaveSize)
                {
                    // Don't need initial offset calc in Y direction: all threads fit in neighbouring rows
                    return make_coord2d(threadIdx.x % (uint32_t)Traits::BlockDimStride_X, 0u);
                }
                else
                {
                    // Threads need to spread over the Y direction as well
                    return make_coord2d(threadIdx.x % (uint32_t)Traits::BlockDimStride_X,
                                        (threadIdx.x / (uint32_t)Traits::BlockDimStride_X)
                                            * MaxVectorWidth % (uint32_t)Traits::BlockKStride_Y);
                }
            }

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                incrementalOffset(uint32_t iteration)
            {
                // Reference:
                // VWOffsetY = VWStride_Y - ((i+1) % VWSegs ? 0u : VWStride_Y * VWSegs);
                // Every set of VWSegs, we must iteratively reset the VWOffset back to 0, hence
                // the subtraction.
                // Optimization 1: if VWSegs == 1, there are no contributions from this stride
                // Optimization 2: if BlockKSegs == 1 and BlockDimSegs == 1, there are no "reset"
                // contributions from this stride
                int32_t VWOffsetY = 0;
                if constexpr((int32_t)Traits::VWSegs > 1)
                {
                    // Offset contribution
                    VWOffsetY = (int32_t)Traits::VWStride_Y;
                    if constexpr(((int32_t)Traits::BlockKSegs > 1)
                                 || ((int32_t)Traits::BlockDimSegs > 1))
                    {
                        // "Reset" cycle
                        VWOffsetY -= (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                          ? 0
                                          : (int32_t)Traits::VWStride_Y * (int32_t)Traits::VWSegs);
                    }
                }

                // Reference:
                // BlockKOffsetY = ((i+1) % VWSegs ? 0u : BlockKStride_Y) -
                // ((i+1) % (VWSegs * BlockKSegs) ? 0u : BlockKSegs * BlockKStride_Y);
                // Every set of BlockKSegs, we must iteratively reset the BlockKOffsetY back to 0, hence
                // the subtraction.
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: if BlockDimSegs == 1, there are no "reset" contributions from this stride
                int32_t BlockKOffsetY = 0;
                if constexpr((int32_t)Traits::BlockKSegs > 1)
                {
                    // Offset contribution
                    BlockKOffsetY = (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                         ? 0
                                         : (int32_t)Traits::BlockKStride_Y);
                    if constexpr((int32_t)Traits::BlockDimSegs > 1)
                    {
                        // "Reset" cycle
                        BlockKOffsetY
                            -= (((int32_t)iteration + 1)
                                        % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                                    ? 0
                                    : (int32_t)Traits::BlockKSegs
                                          * (int32_t)Traits::BlockKStride_Y);
                    }
                }

                // Reference:
                // BlockDimOffsetX = ((i+1) % VWSegs * BlockKSegs) ? 0u : BlockDimStride_X);
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: There are no "reset" contributions from this stride because it is the last dim
                int32_t BlockDimOffsetX = 0;
                if constexpr((int32_t)Traits::BlockDimSegs > 1)
                {
                    // Offset contribution
                    BlockDimOffsetX
                        = (((int32_t)iteration + 1)
                                   % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                               ? 0
                               : (int32_t)Traits::BlockDimStride_X);
                }

                return make_coord2d(BlockDimOffsetX, VWOffsetY + BlockKOffsetY);
            }

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                cumulativeOffset(uint32_t iteration)
            {
                int32_t cumVWOffsetY
                    = (int32_t)Traits::VWStride_Y * ((int32_t)iteration % (int32_t)Traits::VWSegs);
                int32_t cumBlockKOffsetY = ((int32_t)iteration / (int32_t)Traits::VWSegs)
                                           % (int32_t)Traits::BlockKSegs
                                           * (int32_t)Traits::BlockKStride_Y;
                int32_t cumBlockDimOffsetX
                    = ((int32_t)iteration / ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs))
                      * (int32_t)Traits::BlockDimStride_X;

                return make_coord2d(cumBlockDimOffsetX, cumVWOffsetY + cumBlockKOffsetY);
            }

            ROCWMMA_DEVICE static inline auto debug() {}
        };

        /* Pattern that maps threads to matrix columns and assumes
        * that VW will be mapped inline with the column.
        * This pattern considers VW up to MaxVW, BlockDim <= 64 and BlockDim > 64.
        *
        * Iterative thread stride cycles (same for all threads):
        *   Fill MaxVW => Fill BlockK => Fill BlockDim
        *
        * Example:
        * BlockDim = 256   BlockK = 4
        * MaxVW = 2       VW = 1
        *
        * BlockDim Stride Count = 4, BlockDimStride = (64, 0)
        * BlockK   Stride Count = 2, BlockKStride   = (0,  2)
        * VW       Stride Count = 2, VWStride       = (1,  0)
        *
        * Stride mapping (BlockDim, BlockK, VW)
        *  C_n = Matrix column
        *  i_n = cumulative iteration
        *
        *  Cartesian iteration offsets (row, col):
        *  i0  = (0,   0) i1  = (1,   0) i2  = (0,   2) i3  = (1,   2)
        *  i4  = (64,  0) i5  = (65,  0) i6  = (64,  2) i7  = (65,  2)
        *  i8  = (128, 0) i9  = (129, 0) i10 = (128, 2) i11 = (129, 2)
        *  i12 = (192, 0) i13 = (193, 0) i14 = (192, 2) i15 = (192, 2)
        *
        *  Strides iteration offsets (BlockDim, BlockK, VW):
        *  i0  = (0,0,0) i1  = (0,0,1)
        *  i2  = (0,1,0) i3  = (0,1,1)
        *  i4  = (1,0,0) i5  = (1,0,1)
        *  i6  = (1,1,0) i7  = (1,1,1)
        *  i8  = (2,0,0) i9  = (2,0,1)
        *  i10 = (2,1,0) i11 = (2,1,1)
        *  i12 = (3,0,0) i13 = (3,0,1)
        *  i14 = (3,1,0) i15 = (3,1,1)
        *
        * Let's follow thread 0:
        *
        *   kDim --------->
        *
        *   BlockDim1
        *   |                           |-- BlockK Stride = 2 --|
        *   |                           i0(0,0,0)   i2(0,1,0)
        *   |            _         _    v_____ _____v_____ _____
        *   v            |         |    |     |     |     |     |
        *                |  VW     1    |     |     |     |     |
        *       BlockDim |  Stride |    | C0  |  C1 |  C2 |  C3 |
        *        Stride  |         _    v     |     v     |     |
        *               64              i1(0,0,1)   i3(0,1,1)   |
        *                |              |     |     |     |     |
        *                |              |     |     |     |     |
        *                |              | C0  |  C1 |  C2 |  C3 |
        *                _              |_____|_____|_____|_____|
        *                               i4(1,0,0)   i6(1,1,0)
        *                               v_____ _____v_____ _____
        *                               |     |     |     |     |
        *                               |     |     |     |     |
        *                               | C0  |  C1 |  C2 |  C3 |
        *                               v     |     v     |     |
        *                               i5(1,0,1)   i7(1,1,1)   |
        *                               |     |     |     |     |
        *                               |     |     |     |     |
        *                               | C0  |  C1 |  C2 |  C3 |
        *                               |_____|_____|_____|_____|
        *                               ...                     ...
        *                               ...                     ...
        *                               ...                     ...
        *                               v     |     v     |     |
        *                               i13(3,0,1)   i14(3,1,1)   |
        *                               |     |     |     |     |
        *                               |     |     |     |     |
        *                               | C0  |  C1 |  C2 |  C3 |
        *                               |_____|_____|_____|_____|
        *
        *                               ^(BlockDim, 0)          ^(BlockDim, BlockK)
        *
        * Register file (for all VectorWidths = [MaxVectorWidth, 1]):
        *
        * Elements 0...........1........................................... ............64
        *         ________________________________________________________________________
        * Reg0   |  C0E0   |  C0E2   | ... |  C0E62   |  C1E0   |  C1E2   | ... |  C1E62  |
        * Reg1   |  C0E1   |  C0E3   | ... |  C0E63   |  C1E1   |  C1E3   | ... |  C1E63  |
        * Reg2   |  C2E0   |  C2E2   | ... |  C2E62   |  C3E0   |  C3E2   | ... |  C3E62  |
        * Reg3   |  C2E1   |  C2E3   | ... |  C2E63   |  C3E1   |  C3E3   | ... |  C3E63  |
        * Reg4   |  C0E64  |  C0E66  | ... |  C0E126  |  C1E64  |  C1E66  | ... |  C1E126 |
        * Reg5   |  C0E65  |  C0E67  | ... |  C0E127  |  C1E65  |  C1E67  | ... |  C1E127 |
        * ...      ...
        * Reg10  |  C2E192 |  C2E194 | ... |  C2E254  |  C3E192 |  C3E194 | ... |  C3E254 |
        * Reg11  |  C2E193 |  C2E195 | ... |  C2E255  |  C3E193 |  C3E195 | ... |  C3E255 |
        *
        */

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColInlineVW
        {
            using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
            struct Traits
            {
                enum : uint32_t
                {
                    // Number of threads per wave
                    WaveSize = IOTraits::ThreadsPerIO,

                    // Strides
                    BlockDimStride_X = min(BlockDim, WaveSize),
                    BlockDimStride_Y = 0u,

                    BlockKStride_X = 0u,
                    BlockKStride_Y = WaveSize * MaxVectorWidth / BlockDimStride_X,

                    VWStride_X = VectorWidth,
                    VWStride_Y = 0u,

                    // Stride Space
                    BlockDimSegs = BlockDim / BlockDimStride_X,
                    BlockKSegs   = BlockK / BlockKStride_Y,
                    VWSegs       = MaxVectorWidth / VWStride_X,
                };

                // Sanity checks for strides sizes
                static_assert(BlockDim >= (uint32_t)Traits::BlockDimStride_X,
                              "BlockDim must be larger than BlockDimStride_X");
                static_assert(BlockDim % (uint32_t)Traits::BlockDimStride_X == 0,
                              "BlockDim must be a multiple of BlockDimStride_X");
                static_assert(BlockK >= (uint32_t)Traits::BlockKStride_Y,
                              "BlockK must be larger than BlockKStride_Y");
                static_assert(BlockK % (uint32_t)Traits::BlockKStride_Y == 0,
                              "BlockK must be a multiple of BlockKStride_Y");
                static_assert(MaxVectorWidth >= (uint32_t)Traits::VWStride_X,
                              "MaxVectorWidth must larger than VWStride_X");
                static_assert(MaxVectorWidth % (uint32_t)Traits::VWStride_X == 0,
                              "MaxVectorWidth must be a multiple of VWStride_X");

                // Orthogonal layout, coordinates are reversed
                using OrthoLayout
                    = RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>;

                using MatrixCoordT = Coord2d;
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return make_vector((uint32_t)Traits::BlockDimSegs, // BlockDim Segments
                                   (uint32_t)Traits::BlockKSegs, // BlockK Segments
                                   (uint32_t)Traits::VWSegs); // VW Segments
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(
                    make_coord2d((uint32_t)Traits::BlockDimStride_X,
                                 (uint32_t)Traits::BlockDimStride_Y),
                    make_coord2d((uint32_t)Traits::BlockKStride_X,
                                 (uint32_t)Traits::BlockKStride_Y),
                    make_coord2d((uint32_t)Traits::VWStride_X, (uint32_t)Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
            {
                if constexpr(((uint32_t)Traits::BlockDimStride_X >= (uint32_t)Traits::WaveSize)
                             && (MaxVectorWidth == 1))
                {
                    // Don't need initial offset calc in Y direction: all threads fit in neighbouring rows
                    return make_coord2d(threadIdx.x % (uint32_t)Traits::BlockDimStride_X, 0u);
                }
                else
                {
                    // Threads need to spread over the Y direction as well
                    return make_coord2d(
                        threadIdx.x * MaxVectorWidth % (uint32_t)Traits::BlockDimStride_X,
                        threadIdx.x * MaxVectorWidth / (uint32_t)Traits::BlockDimStride_X
                            % (uint32_t)Traits::BlockKStride_Y);
                }
            }

            // Incremental iteration offset
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                incrementalOffset(uint32_t iteration)
            {
                // Reference:
                // VWOffsetX = VWStride_X - ((i+1) % VWSegs ? 0u : VWStride_X * VWSegs);
                // Every set of VWSegs, we must iteratively reset the VWOffset back to 0, hence
                // the subtraction.
                // Optimization 1: if VWSegs == 1, there are no contributions from this stride
                // Optimization 2: if BlockKSegs == 1 and BlockDimSegs == 1, there are no "reset"
                // contributions from this stride
                int32_t VWOffsetX = 0;
                if constexpr((int32_t)Traits::VWSegs > 1)
                {
                    // Offset contribution
                    VWOffsetX = (int32_t)Traits::VWStride_X;
                    if constexpr(((int32_t)Traits::BlockKSegs > 1)
                                 || ((int32_t)Traits::BlockDimSegs > 1))
                    {
                        // "Reset" cycle
                        VWOffsetX -= (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                          ? 0
                                          : (int32_t)Traits::VWStride_X * (int32_t)Traits::VWSegs);
                    }
                }

                // Reference:
                // BlockKOffsetY = ((i+1) % VWSegs ? 0u : BlockKStride_Y) -
                // ((i+1) % (VWSegs * BlockKSegs) ? 0u : BlockKSegs * BlockKStride_Y);
                // Every set of BlockKSegs, we must iteratively reset the BlockKOffsetY back to 0, hence
                // the subtraction.
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: if BlockDimSegs == 1, there are no "reset" contributions from this stride
                int32_t BlockKOffsetY = 0;
                if constexpr((int32_t)Traits::BlockKSegs > 1)
                {
                    // Offset contribution
                    BlockKOffsetY = (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                         ? 0
                                         : (int32_t)Traits::BlockKStride_Y);
                    if constexpr((int32_t)Traits::BlockDimSegs > 1)
                    {
                        // "Reset" cycle
                        BlockKOffsetY
                            -= (((int32_t)iteration + 1)
                                        % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                                    ? 0
                                    : (int32_t)Traits::BlockKSegs
                                          * (int32_t)Traits::BlockKStride_Y);
                    }
                }

                // Reference:
                // BlockDimOffsetX = ((i+1) % VWSegs * BlockKSegs) ? 0u : BlockDimStride_X);
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: There are no "reset" contributions from this stride because it is the last dim
                int32_t BlockDimOffsetX = 0;
                if constexpr((int32_t)Traits::BlockDimSegs > 1)
                {
                    // Offset contribution
                    BlockDimOffsetX
                        = (((int32_t)iteration + 1)
                                   % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                               ? 0
                               : (int32_t)Traits::BlockDimStride_X);
                }

                return make_coord2d(VWOffsetX + BlockDimOffsetX, BlockKOffsetY);
            }

            // Cumulative iteration offset
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                cumulativeOffset(uint32_t iteration)
            {
                int32_t cumVWOffsetX
                    = (int32_t)Traits::VWStride_X * ((int32_t)iteration % (int32_t)Traits::VWSegs);
                int32_t cumBlockKOffsetY = ((int32_t)iteration / (int32_t)Traits::VWSegs)
                                           % (int32_t)Traits::BlockKSegs
                                           * (int32_t)Traits::BlockKStride_Y;
                int32_t cumBlockDimOffsetX
                    = ((int32_t)iteration / ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs))
                      * (int32_t)Traits::BlockKStride_X;

                return make_coord2d(cumVWOffsetX + cumBlockDimOffsetX, cumBlockKOffsetY);
            }
            ROCWMMA_DEVICE static inline auto debug() {}
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK /* = 1*/> // # of splits
        struct ColInlineInt
        {
            using IOTraits = IOTraits<BlockDim, BlockK, DataT, (BlockDim / MfmaDim)>;
            struct Traits
            {
                enum : uint32_t
                {
                    // Number of threads per wave
                    WaveSize = IOTraits::ThreadsPerIO,

                    // Number of elements each thread will fetch in BlockDim direction
                    DimPerThread = BlockDim / MfmaDim,

                    // Number of elements each thread will fetch in BlockK direction
                    KPerThread = BlockK * MfmaDim / (WaveSize * SplitK),

                    // Number of elements that each thread is responsible for
                    ElementsPerThread = DimPerThread * KPerThread,

                    // Strides
                    SplitKStride_X = 0u,
                    SplitKStride_Y = BlockK / SplitK,

                    BlockKStride_X = 0u,
                    BlockKStride_Y = 1u,

                    VWStride_X = DimPerThread,
                    VWStride_Y = 0u,

                    // Stride Space
                    SplitKSegs = BlockK / SplitKStride_Y,
                    BlockKSegs = KPerThread / BlockKStride_Y,
                    VWSegs     = DimPerThread / VWStride_X,
                };

                // // Check VectorWidth validity
                // static_assert((uint32_t)Traits::DimPerThread >= VectorWidth, "Invalid VectorWidth");
                // static_assert((uint32_t)Traits::DimPerThread % VectorWidth == 0,
                //               "DimPerThread not a multiple of VectorWidth");

                // Check KPerThread validity
                static_assert(BlockK >= (uint32_t)Traits::KPerThread, "Invalid KPerThread");
                static_assert(BlockK % (uint32_t)Traits::KPerThread == 0,
                              "BlockK is not a multiple of KPerThread");

                // Check SplitK validity
                static_assert(BlockK >= SplitK, "Invalid SplitK");
                static_assert(BlockK % SplitK == 0, "BlockK is not a multiple of SplitK");

                // Check MfmaDim validity
                static_assert(BlockDim >= MfmaDim, "BlockDim must be larger than MfmaDim");
                static_assert(BlockDim % MfmaDim == 0, "BlockDim must be a multiple of MfmaDim");

                // Orthogonal layout, coordinates are reversed
                using OrthoLayout = RowInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>;

                using MatrixCoordT = Coord2d;
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {

                return make_vector((uint32_t)Traits::SplitKSegs,
                                   (uint32_t)Traits::BlockKSegs,
                                   (uint32_t)Traits::VWSegs);
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(
                    make_coord2d((uint32_t)Traits::SplitKStride_X,
                                 (uint32_t)Traits::SplitKStride_Y),
                    make_coord2d((uint32_t)Traits::BlockKStride_X,
                                 (uint32_t)Traits::BlockKStride_Y),
                    make_coord2d((uint32_t)Traits::VWStride_X, (uint32_t)Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
            {
                return make_coord2d((threadIdx.x * (uint32_t)Traits::DimPerThread) % BlockDim,
                                    (threadIdx.x / MfmaDim * (uint32_t)Traits::KPerThread)
                                        % BlockK);
            }

            // Incremental iteration offset
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                incrementalOffset(uint32_t iteration)
            {
                // Reference:
                // VWOffsetX = VWStride_X - ((i+1) % VWSegs ? 0u : VWStride_X * VWSegs);
                // Every set of VWSegs, we must iteratively reset the VWOffset back to 0, hence
                // the subtraction.
                // Optimization 1: if VWSegs == 1, there are no contributions from this stride
                // Optimization 2: if BlockKSegs == 1 and SplitKSegs == 1, there are no "reset"
                // contributions from this stride
                int32_t VWOffsetX = 0;
                if constexpr((int32_t)Traits::VWSegs > 1)
                {
                    // Offset contribution
                    VWOffsetX = (int32_t)Traits::VWStride_X;
                    if constexpr(((int32_t)Traits::BlockKSegs > 1)
                                 || ((int32_t)Traits::SplitKSegs > 1))
                    {
                        // "Reset" cycle
                        VWOffsetX -= (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                          ? 0
                                          : (int32_t)Traits::VWStride_X * (int32_t)Traits::VWSegs);
                    }
                }

                // Reference:
                // BlockKOffsetY = ((i+1) % VWSegs ? 0u : BlockKStride_Y) -
                // ((i+1) % (VWSegs * BlockKSegs) ? 0u : BlockKSegs * BlockKStride_Y);
                // Every set of BlockKSegs, we must iteratively reset the BlockKOffsetY back to 0, hence
                // the subtraction.
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: if SplitKSegs == 1, there are no "reset" contributions from this stride
                int32_t BlockKOffsetY = 0;
                if constexpr((int32_t)Traits::BlockKSegs > 1)
                {
                    // Offset contribution
                    BlockKOffsetY = (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                         ? 0
                                         : (int32_t)Traits::BlockKStride_Y);
                    if constexpr((int32_t)Traits::SplitKSegs > 1)
                    {
                        // "Reset" cycle
                        BlockKOffsetY
                            -= (((int32_t)iteration + 1)
                                        % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                                    ? 0
                                    : (int32_t)Traits::BlockKSegs
                                          * (int32_t)Traits::BlockKStride_Y);
                    }
                }

                // Reference:
                // BlockDimOffsetX = ((i+1) % VWSegs * BlockKSegs) ? 0u : SplitKStride_X);
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: There are no "reset" contributions from this stride because it is the last dim
                int32_t BlockDimOffsetX = 0;
                if constexpr((int32_t)Traits::SplitKSegs > 1)
                {
                    // Offset contribution
                    BlockDimOffsetX
                        = (((int32_t)iteration + 1)
                                   % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                               ? 0
                               : (int32_t)Traits::SplitKStride_X);
                }

                return make_coord2d(VWOffsetX + BlockDimOffsetX, BlockKOffsetY);
            }

            // Cumulative iteration offset
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                cumulativeOffset(uint32_t iteration)
            {
                int32_t cumVWOffsetX
                    = (int32_t)Traits::VWStride_X * ((int32_t)iteration % (int32_t)Traits::VWSegs);
                int32_t cumBlockKOffsetY = ((int32_t)iteration / (int32_t)Traits::VWSegs)
                                           % (int32_t)Traits::BlockKSegs
                                           * (int32_t)Traits::BlockKStride_Y;
                int32_t cumBlockDimOffsetX
                    = ((int32_t)iteration / ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs))
                      * (int32_t)Traits::SplitKStride_X;

                return make_coord2d(cumVWOffsetX + cumBlockDimOffsetX, cumBlockKOffsetY);
            }
            ROCWMMA_DEVICE static inline auto debug()
            {
                if(threadIdx.x == 0 && threadIdx.y == 0)
                {
                    printf("SplitKSegs: %d, BlockKSegs: %d, VWSegs: %d\n",
                           (uint32_t)Traits::SplitKSegs,
                           (uint32_t)Traits::BlockKSegs,
                           (uint32_t)Traits::VWSegs);

                    printf("SplitKStride_X: %d, SplitKStride_Y: %d\nBlockKStride_X: %d, "
                           "BlockKStride_Y: %d\nVWStride_X: %d, VWStride_Y: %d\n",
                           (uint32_t)Traits::SplitKStride_X,
                           (uint32_t)Traits::SplitKStride_Y,
                           (uint32_t)Traits::BlockKStride_X,
                           (uint32_t)Traits::BlockKStride_Y,
                           (uint32_t)Traits::VWStride_X,
                           (uint32_t)Traits::VWStride_Y);
                }
                if(threadIdx.x <= 63 && threadIdx.y == 0)
                {
                    printf("Tid: (%d) Base offset(X, Y): = (%d, %d)\n",
                           threadIdx.x,
                           get<0>(baseOffset()),
                           get<1>(baseOffset()));
                }
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK /*= 1*/> // # of splits
        struct ColOrthoInt
        {
            using IOTraits = IOTraits<BlockDim, BlockK, DataT, BlockDim / MfmaDim>;
            struct Traits
            {
                enum : uint32_t
                {
                    // Number of threads per wave
                    WaveSize = IOTraits::ThreadsPerIO,

                    // Number of elements each thread will fetch in BlockDim direction
                    DimPerThread = BlockDim / MfmaDim,

                    // Number of elements each thread will fetch in BlockK direction
                    KPerThread = BlockK * MfmaDim / (WaveSize * SplitK),

                    // Number of elements that each thread is responsible for
                    ElementsPerThread = DimPerThread * KPerThread,

                    // Strides
                    SplitKStride_X = 0u,
                    SplitKStride_Y = BlockK / SplitK,

                    BlockKStride_X = 1u,
                    BlockKStride_Y = 0u,

                    VWStride_X = 0u,
                    VWStride_Y = DimPerThread,

                    // Stride Space
                    SplitKSegs = BlockK / SplitKStride_Y,
                    BlockKSegs = DimPerThread / BlockKStride_X,
                    VWSegs     = KPerThread / VWStride_Y,
                };

                // Check KPerThread validity
                static_assert(BlockK >= (uint32_t)Traits::KPerThread, "Invalid KPerThread");
                static_assert(BlockK % (uint32_t)Traits::KPerThread == 0,
                              "BlockK is not a multiple of KPerThread");

                // // Check VectorWidth validity
                // static_assert((uint32_t)Traits::KPerThread >= VectorWidth, "Invalid VectorWidth");
                // static_assert((uint32_t)Traits::KPerThread % VectorWidth == 0,
                //               "KPerThread not a multiple of VectorWidth");

                // Check SplitK validity
                static_assert(BlockK >= SplitK, "Invalid SplitK");
                static_assert(BlockK % SplitK == 0, "BlockK is not a multiple of SplitK");

                // Check MfmaDim validity
                static_assert(BlockDim >= MfmaDim, "BlockDim must be larger than MfmaDim");
                static_assert(BlockDim % MfmaDim == 0, "BlockDim must be a multiple of MfmaDim");

                // Orthogonal layout, coordinates are reversed
                using OrthoLayout = RowOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>;

                using MatrixCoordT = Coord2d;
            };

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return make_vector((uint32_t)Traits::SplitKSegs, // WaveKSegs Segments
                                   (uint32_t)Traits::BlockKSegs, // BlockK Segments
                                   (uint32_t)Traits::VWSegs); // VW Segments
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                return make_vector(
                    make_coord2d((uint32_t)Traits::SplitKStride_X,
                                 (uint32_t)Traits::SplitKStride_Y),
                    make_coord2d((uint32_t)Traits::BlockKStride_X,
                                 (uint32_t)Traits::BlockKStride_Y),
                    make_coord2d((uint32_t)Traits::VWStride_X, (uint32_t)Traits::VWStride_Y));
            }

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
            {
                return make_coord2d((threadIdx.x * (uint32_t)Traits::DimPerThread) % BlockDim,
                                    (threadIdx.x / MfmaDim * (uint32_t)Traits::KPerThread)
                                        % BlockK);
            }

            // Incremental iteration offset
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                incrementalOffset(uint32_t iteration)
            {
                // Reference:
                // VWOffsetX = VWStride_X - ((i+1) % VWSegs ? 0u : VWStride_X * VWSegs);
                // Every set of VWSegs, we must iteratively reset the VWOffset back to 0, hence
                // the subtraction.
                // Optimization 1: if VWSegs == 1, there are no contributions from this stride
                // Optimization 2: if BlockKSegs == 1 and SplitKSegs == 1, there are no "reset"
                // contributions from this stride
                int32_t VWOffsetX = 0;
                if constexpr((int32_t)Traits::VWSegs > 1)
                {
                    // Offset contribution
                    VWOffsetX = (int32_t)Traits::VWStride_X;
                    if constexpr(((int32_t)Traits::BlockKSegs > 1)
                                 || ((int32_t)Traits::SplitKSegs > 1))
                    {
                        // "Reset" cycle
                        VWOffsetX -= (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                          ? 0
                                          : (int32_t)Traits::VWStride_X * (int32_t)Traits::VWSegs);
                    }
                }

                // Reference:
                // BlockKOffsetY = ((i+1) % VWSegs ? 0u : BlockKStride_Y) -
                // ((i+1) % (VWSegs * BlockKSegs) ? 0u : BlockKSegs * BlockKStride_Y);
                // Every set of BlockKSegs, we must iteratively reset the BlockKOffsetY back to 0, hence
                // the subtraction.
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: if SplitKSegs == 1, there are no "reset" contributions from this stride
                int32_t BlockKOffsetY = 0;
                if constexpr((int32_t)Traits::BlockKSegs > 1)
                {
                    // Offset contribution
                    BlockKOffsetY = (((int32_t)iteration + 1) % (int32_t)Traits::VWSegs
                                         ? 0
                                         : (int32_t)Traits::BlockKStride_Y);
                    if constexpr((int32_t)Traits::SplitKSegs > 1)
                    {
                        // "Reset" cycle
                        BlockKOffsetY
                            -= (((int32_t)iteration + 1)
                                        % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                                    ? 0
                                    : (int32_t)Traits::BlockKSegs
                                          * (int32_t)Traits::BlockKStride_Y);
                    }
                }

                // Reference:
                // BlockDimOffsetX = ((i+1) % VWSegs * BlockKSegs) ? 0u : SplitKStride_X);
                // Optimization 1: if BlockKSegs == 1, there are no contributions from this stride
                // Optimization 2: There are no "reset" contributions from this stride because it is the last dim
                int32_t BlockDimOffsetX = 0;
                if constexpr((int32_t)Traits::SplitKSegs > 1)
                {
                    // Offset contribution
                    BlockDimOffsetX
                        = (((int32_t)iteration + 1)
                                   % ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs)
                               ? 0
                               : (int32_t)Traits::SplitKStride_X);
                }

                return make_coord2d(VWOffsetX + BlockDimOffsetX, BlockKOffsetY);
            }

            // Cumulative iteration offset
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                cumulativeOffset(uint32_t iteration)
            {
                int32_t cumVWOffsetX
                    = (int32_t)Traits::VWStride_X * ((int32_t)iteration % (int32_t)Traits::VWSegs);
                int32_t cumBlockKOffsetY = ((int32_t)iteration / (int32_t)Traits::VWSegs)
                                           % (int32_t)Traits::BlockKSegs
                                           * (int32_t)Traits::BlockKStride_Y;
                int32_t cumBlockDimOffsetX
                    = ((int32_t)iteration / ((int32_t)Traits::VWSegs * (int32_t)Traits::BlockKSegs))
                      * (int32_t)Traits::SplitKStride_X;

                return make_coord2d(cumVWOffsetX + cumBlockDimOffsetX, cumBlockKOffsetY);
            }

            ROCWMMA_DEVICE static inline auto debug()
            {
                // if(threadIdx.x == 0 && threadIdx.y == 0)
                // {
                //     printf("SplitKSegs: %d, BlockKSegs: %d, VWSegs: %d\n",
                //         (uint32_t)Traits::SplitKSegs,
                //         (uint32_t)Traits::BlockKSegs,
                //         (uint32_t)Traits::VWSegs);

                //     printf("SplitKStride_X: %d, SplitKStride_Y: %d\nBlockKStride_X: %d, BlockKStride_Y: %d\nVWStride_X: %d, VWStride_Y: %d\n",
                //         (uint32_t)Traits::SplitKStride_X,
                //         (uint32_t)Traits::SplitKStride_Y,
                //         (uint32_t)Traits::BlockKStride_X,
                //         (uint32_t)Traits::BlockKStride_Y,
                //         (uint32_t)Traits::VWStride_X,
                //         (uint32_t)Traits::VWStride_Y);

                // }
                // if(threadIdx.x <= 63 && threadIdx.y == 0)
                // {
                //     printf("Base offset(X, Y): = (%d, %d)", get<0>(baseOffset()), get<1>(baseOffset()));
                // }
            }
        };

        template <typename MatrixLayout>
        struct OrthoImpl
        {
            struct Traits
            {
                using OrthoLayout = orthogonal_layout_t<MatrixLayout>;
            };

            // Matrix coord offsets
            ROCWMMA_DEVICE static inline typename auto baseOffset()
            {
                return swap(Traits::OrthoLayout::baseOffset());
            }

            ROCWMMA_DEVICE constexpr static inline auto strideCounts()
            {
                return Traits::OrthoLayout::strideCounts();
            }

            ROCWMMA_DEVICE constexpr static inline auto strides()
            {
                auto t = Traits::OrthoLayout::strides();
                return make_vector(swap(get<0>(t)), swap(get<1>(t)), swap(get<2>(t)));
            }

            ROCWMMA_DEVICE static inline auto incrementalOffset(uint32_t iteration)
            {
                return swap(Traits::OrthoLayout::incrementalOffset(iteration));
            }

            ROCWMMA_DEVICE static inline auto cumulativeOffset(uint32_t iteration)
            {
                return swap(Traits::OrthoLayout::cumulativeOffset(iteration));
            }

            ROCWMMA_DEVICE static inline auto debug() {}
        };

        using RowOrthoVW   = OrthoImpl<ColOrthoVW>;
        using RowInlineVW  = OrthoImpl<ColInlineVW>;
        using RowOrthoInt  = OrthoImpl<ColOrthoInt>;
        using RowInlineInt = OrthoImpl<ColInlineInt>;

    } // namespace MatrixLayout

} // namespace rocwmma

#endif // ROCWMMA_MATRIX_LAYOUT_IMPL_HPP
