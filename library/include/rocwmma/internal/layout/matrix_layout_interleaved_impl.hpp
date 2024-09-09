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
#ifndef ROCWMMA_MATRIX_LAYOUT_INTERLEAVED_IMPL_HPP
#define ROCWMMA_MATRIX_LAYOUT_INTERLEAVED_IMPL_HPP

#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{

    // Implementations for the interleaved MatrixLayout classes
    namespace MatrixLayout
    {
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth, //
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK /* = 1*/> // # of splits
        struct ColInlineInt
        {
            using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
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

                    VWStride_X = VectorWidth,
                    VWStride_Y = 0u,

                    // Stride Space
                    SplitKSegs = BlockK / SplitKStride_Y,
                    BlockKSegs = KPerThread / BlockKStride_Y,
                    VWSegs     = DimPerThread / VWStride_X,
                };

                // Check VectorWidth validity
                static_assert((uint32_t)Traits::DimPerThread >= VectorWidth, "Invalid VectorWidth");
                static_assert((uint32_t)Traits::DimPerThread % VectorWidth == 0,
                              "DimPerThread not a multiple of VectorWidth");

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
                using OrthoLayout = RowInlineInt<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 VectorWidth,
                                                 MaxVectorWidth,
                                                 MfmaDim,
                                                 SplitK>;

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
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth, //
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK /*= 1*/> // # of splits
        struct ColOrthoInt
        {
            using IOTraits = IOTraits<BlockDim, BlockK, DataT, VectorWidth>;
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
                    VWStride_Y = VectorWidth,

                    // Stride Space
                    SplitKSegs = BlockK / SplitKStride_Y,
                    BlockKSegs = DimPerThread / BlockKStride_X,
                    VWSegs     = KPerThread / VWStride_Y,
                };

                // Check KPerThread validity
                static_assert(BlockK >= (uint32_t)Traits::KPerThread, "Invalid KPerThread");
                static_assert(BlockK % (uint32_t)Traits::KPerThread == 0,
                              "BlockK is not a multiple of KPerThread");

                // Check VectorWidth validity
                static_assert((uint32_t)Traits::KPerThread >= VectorWidth, "Invalid VectorWidth");
                static_assert((uint32_t)Traits::KPerThread % VectorWidth == 0,
                              "KPerThread not a multiple of VectorWidth");

                // Check SplitK validity
                static_assert(BlockK >= SplitK, "Invalid SplitK");
                static_assert(BlockK % SplitK == 0, "BlockK is not a multiple of SplitK");

                // Check MfmaDim validity
                static_assert(BlockDim >= MfmaDim, "BlockDim must be larger than MfmaDim");
                static_assert(BlockDim % MfmaDim == 0, "BlockDim must be a multiple of MfmaDim");

                // Orthogonal layout, coordinates are reversed
                using OrthoLayout = RowOrthoInt<BlockDim,
                                                BlockK,
                                                DataT,
                                                VectorWidth,
                                                MaxVectorWidth,
                                                MfmaDim,
                                                SplitK>;

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

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth,
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK /*= 1*/>
        struct RowInlineInt
        {
            // RowInlineInt is orthogonal to ColInlineInt, therefore we can use reversed coordinates
            struct Traits
            {
                using OrthoLayout = ColInlineInt<BlockDim,
                                                 BlockK,
                                                 DataT,
                                                 VectorWidth,
                                                 MaxVectorWidth,
                                                 MfmaDim,
                                                 SplitK>;

                using MatrixCoordT = Coord2d;
            };

            // Matrix coord offsets
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
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

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                incrementalOffset(uint32_t iteration)
            {
                return swap(Traits::OrthoLayout::incrementalOffset(iteration));
            }
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                cumulativeOffset(uint32_t iteration)
            {
                return swap(Traits::OrthoLayout::cumulativeOffset(iteration));
            }

            ROCWMMA_DEVICE static inline auto debug()
            {
                Traits::OrthoLayout::debug();
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth,
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK /*= 1*/>
        struct RowOrthoInt
        {
            // RowOrthoInt is orthogonal to ColOrthoInt, therefore we can use reversed coordinates
            struct Traits
            {
                using OrthoLayout = ColOrthoInt<BlockDim,
                                                BlockK,
                                                DataT,
                                                VectorWidth,
                                                MaxVectorWidth,
                                                MfmaDim,
                                                SplitK>;

                using MatrixCoordT = Coord2d;
            };

            // Matrix coord offsets
            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT baseOffset()
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

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                incrementalOffset(uint32_t iteration)
            {
                return swap(Traits::OrthoLayout::incrementalOffset(iteration));
            }

            ROCWMMA_DEVICE static inline typename Traits::MatrixCoordT
                cumulativeOffset(uint32_t iteration)
            {
                return swap(Traits::OrthoLayout::cumulativeOffset(iteration));
            }

            ROCWMMA_DEVICE static inline auto debug() {}
        };

    } // namespace MatrixLayout

    ////////////////////////////////////
    /// MatrixLayout specializations ///
    ////////////////////////////////////

    // Matrix layout matching test criteria are if all parameters match, with some flexibility in VectorWidth.
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK,
              template <uint32_t, uint32_t, typename, uint32_t, uint32_t, uint32_t, uint32_t>
              class MatrixLayout>
    struct is_layout_same<
        MatrixLayout<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth, MmaDim, SplitK>,
        MatrixLayout<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth, MmaDim, SplitK>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    // Matrix layout transpose test with flexibility in the VectorWidth.
    // Transposed matrix layouts swap matrix space rows / cols.
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                  BlockK,
                                                                  DataT,
                                                                  VectorWidthLhs,
                                                                  MaxVectorWidth,
                                                                  MmaDim,
                                                                  SplitK>,
                               MatrixLayout::template RowOrthoInt<BlockDim,
                                                                  BlockK,
                                                                  DataT,
                                                                  VectorWidthRhs,
                                                                  MaxVectorWidth,
                                                                  MmaDim,
                                                                  SplitK>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                  BlockK,
                                                                  DataT,
                                                                  VectorWidthLhs,
                                                                  MaxVectorWidth,
                                                                  MmaDim,
                                                                  SplitK>,
                               MatrixLayout::template ColOrthoInt<BlockDim,
                                                                  BlockK,
                                                                  DataT,
                                                                  VectorWidthRhs,
                                                                  MaxVectorWidth,
                                                                  MmaDim,
                                                                  SplitK>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<MatrixLayout::template ColInlineInt<BlockDim,
                                                                   BlockK,
                                                                   DataT,
                                                                   VectorWidthLhs,
                                                                   MaxVectorWidth,
                                                                   MmaDim,
                                                                   SplitK>,
                               MatrixLayout::template RowInlineInt<BlockDim,
                                                                   BlockK,
                                                                   DataT,
                                                                   VectorWidthRhs,
                                                                   MaxVectorWidth,
                                                                   MmaDim,
                                                                   SplitK>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<MatrixLayout::template RowInlineInt<BlockDim,
                                                                   BlockK,
                                                                   DataT,
                                                                   VectorWidthLhs,
                                                                   MaxVectorWidth,
                                                                   MmaDim,
                                                                   SplitK>,
                               MatrixLayout::template ColInlineInt<BlockDim,
                                                                   BlockK,
                                                                   DataT,
                                                                   VectorWidthRhs,
                                                                   MaxVectorWidth,
                                                                   MmaDim,
                                                                   SplitK>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    // Matrix space transpose guide: Swap rows / cols
    // VW stays consistent.
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct layout_transpose<MatrixLayout::template ColOrthoInt<BlockDim,
                                                               BlockK,
                                                               DataT,
                                                               VectorWidth,
                                                               MaxVectorWidth,
                                                               MmaDim,
                                                               SplitK>>
    {
        using type = MatrixLayout::template RowOrthoInt<BlockDim,
                                                        BlockK,
                                                        DataT,
                                                        VectorWidth,
                                                        MaxVectorWidth,
                                                        MmaDim,
                                                        SplitK>;
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct layout_transpose<MatrixLayout::template RowOrthoInt<BlockDim,
                                                               BlockK,
                                                               DataT,
                                                               VectorWidth,
                                                               MaxVectorWidth,
                                                               MmaDim,
                                                               SplitK>>
    {
        using type = MatrixLayout::template ColOrthoInt<BlockDim,
                                                        BlockK,
                                                        DataT,
                                                        VectorWidth,
                                                        MaxVectorWidth,
                                                        MmaDim,
                                                        SplitK>;
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct layout_transpose<MatrixLayout::template ColInlineInt<BlockDim,
                                                                BlockK,
                                                                DataT,
                                                                VectorWidth,
                                                                MaxVectorWidth,
                                                                MmaDim,
                                                                SplitK>>
    {
        using type = MatrixLayout::template RowInlineInt<BlockDim,
                                                         BlockK,
                                                         DataT,
                                                         VectorWidth,
                                                         MaxVectorWidth,
                                                         MmaDim,
                                                         SplitK>;
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct layout_transpose<MatrixLayout::template RowInlineInt<BlockDim,
                                                                BlockK,
                                                                DataT,
                                                                VectorWidth,
                                                                MaxVectorWidth,
                                                                MmaDim,
                                                                SplitK>>
    {
        using type = MatrixLayout::template ColInlineInt<BlockDim,
                                                         BlockK,
                                                         DataT,
                                                         VectorWidth,
                                                         MaxVectorWidth,
                                                         MmaDim,
                                                         SplitK>;
    };

    ///////////////////////////////////////
    /// Register layout specializations ///
    ///////////////////////////////////////

    // Register layouts are the same if all test parameters match, with some flexibility in VectorWidth.
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK,
              template <uint32_t, uint32_t, typename, uint32_t, uint32_t>
              class MatrixLayout>
    struct is_layout_same<
        RegisterLayout::template Storage<
            MatrixLayout<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth, MmaDim, SplitK>>,
        RegisterLayout::template Storage<
            MatrixLayout<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth, MmaDim, SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    // ColOrthoInt and RowOrthoInt layouts are already in mma input format for mma sized BlockDim (16 or 32)
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidth,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template MmaInput<BlockDim>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template MmaInput<BlockDim>,
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidth,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidth,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template MmaInput<BlockDim>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template MmaInput<BlockDim>,
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidth,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    // TODO: necessary?
    // In-register layouts for transposed RowOrthoInt / ColOrthoInt and RowInlineInt / ColInline are technically 'the same'
    // for each thread, even though the data interpretation is different (e.g., row elements vs col elements).
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthLhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthRhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthLhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthRhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthLhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthRhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_same<
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthLhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthRhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    // ColInlineInt and RowInlineInt layouts are transposed to mma input format for mma sized BlockDim (16 or 32)
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidth,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template MmaInput<BlockDim>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template MmaInput<BlockDim>,
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidth,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidth,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template MmaInput<BlockDim>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidth,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template MmaInput<BlockDim>,
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidth,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool, detail::testSupportedMmaDim(BlockDim)>
    {
    };

    // In-register layouts for (ColInlineInt / ColOrthoInt) and (RowInlineInt / RowOrthoInt) are the orthogonal register transposes.
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthLhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthRhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthLhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthRhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthLhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthRhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthLhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthRhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    // In-register layouts for (RowOrthoInt / ColInlineInt) and (ColOrthoInt / RowInlineInt) are the orthogonal register transposes.
    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthLhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthRhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template ColInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthLhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template RowOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthRhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthLhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthRhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

    template <uint32_t BlockDim,
              uint32_t BlockK,
              typename DataT,
              uint32_t VectorWidthLhs,
              uint32_t VectorWidthRhs,
              uint32_t MaxVectorWidth,
              uint32_t MmaDim,
              uint32_t SplitK>
    struct is_layout_transpose<
        RegisterLayout::template Storage<MatrixLayout::template RowInlineInt<BlockDim,
                                                                             BlockK,
                                                                             DataT,
                                                                             VectorWidthLhs,
                                                                             MaxVectorWidth,
                                                                             MmaDim,
                                                                             SplitK>>,
        RegisterLayout::template Storage<MatrixLayout::template ColOrthoInt<BlockDim,
                                                                            BlockK,
                                                                            DataT,
                                                                            VectorWidthRhs,
                                                                            MaxVectorWidth,
                                                                            MmaDim,
                                                                            SplitK>>>
        : public integral_constant<bool,
                                   detail::testSupportedVW(
                                       MaxVectorWidth, VectorWidthLhs, VectorWidthRhs)>
    {
    };

} // namespace rocwmma

#endif // ROCWMMA_MATRIX_LAYOUT_INTERLEAVED_IMPL_HPP
