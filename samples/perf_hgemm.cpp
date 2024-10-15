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
#include <iostream>
#include <vector>

#include <hip/hip_ext.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

#include "common.hpp"

/* Motivation
*
* For this particular GEMM kernel, high performance can be
* achieved through two general principles:
* 1) Data re-use
* 2) Latency hiding
*
* From the simple_gemm implementation, we know that the GEMM
* equation takes the form:
*
* D = alpha * AxB + beta * C, where
*
* A, B = input tiles of MxK and KxN, respectively
* C = input tile of MxN and
* D = final output tile, MxN
* alpha, beta are scalar factors
* (M, N and K are block dimensions)
*
* In the simple_gemm sample, each warp is responsible for computing
* one output D tile of the final result. In the current sample, each
* warp is now responsible for computing multiple D tiles, what we
* might call a Warp Tile. Because Warp Tile blocks share data locality
* in either the same row or column direction, warps can re-use input
* data from A and B as they step through the K dimension for each block.
*
* Moreover, Warp Tiles processed by warps in a thread block
* have common locality in the larger Macro Tile. In the Global D layout
* shown below, data re-use opportunities await in D tiles aligned in the
* same rows / columns. These will pass over the same input A/B values as
* they march through the K dimension.
*
* Block size:      (BlockM x BlockN)
* Warp tile size:  (BlocksX * BlockSize.x) x (BlocksY * BlockSize.y)
* Macro Tile size: (TBlock.x * WarpTileSize.x) x (TBlock.y * WarpTileSize.y)
*
* Wave data share input A: same row
* Wave data share input B: same col
*
* Global D layout & warp assignment for BlocksX = BlocksY = 2, 2x2 Warps
*
* W (X, Y) = wave row X, col Y
*                                     |--------- Macro Tile Y-------------|
*                                     |-- Wave Tile Y --|
*                                     |-BlockN-|
*
*                                      BlockN x BlocksY   BlockN x BlocksY
*                                     |<--------------->|<--------------->|
*      _ _   _ _      _ _          ___  ________ ________ ________ ________
*       |     |        |            ^  |        |        |        |        |
*       | Wave| BlockM |   BlockM   |  |        W        |        W        |
*       | Tile|       _|_     x     |  |__   (0, 0)    __|__   (0, 1)    __|
*       |  X  |            BlocksX  |  |                 |                 |
* Macro |     |                     |  |                 |                 |
*  Tile |    _|_                   _v_ |________|________|________|________|
*   X   |                           ^  |        |        |        |        |
*       |                  BlockM   |  |        W        |        W        |
*       |                     x     |  |__   (1, 0)    __|__   (1, 1)    __|
*       |                  BlocksX  |  |                 |                 |
*       |                           |  |                 |                 |
*      _|_                         _v_ |________|________|________|________|
*
*
* From the above diagram, we can see that input A/B data can be shared within warps,
* as well as between warps in the same threadblock. This means that warps in the same
* thread block can share the input loading responsibilities if they synchronize stepping
* through the K dimension for tiles at the same time.
*
* rocWMMA Cooperative API allows thread blocks to collaboratively move data from
* one location to another. In this case, we will move data from global memory space to
* local storage such that inter-warp data sharing is possible. Maximizing data re-use
* in this way reduces costly access to global memory and improves performance.
*
* To maximize efficiency, we can structure the kernel to maximize bandwidth usage and keep
* the compute resources as busy as possible at the same time. Using a pre-fetch technique,
* we can fetch A/B inputs for the next K-step while keeping the compute resources busy
* processing the current K-step. This helps to hide memory fetching latency.
*
* In general, the process would flow like the following:
*
*       Start
*         |
*   Pre-Fetch Global A/B for K0
*         |
*   Store LDS buffer0
*         |
*         v
*   Loop: i = 1:K-1
*   ^         |
*   |    Fetch Global A/B i+1; store LDS Buffer 1
*   |         |
*   |    Load LDS buffer0; Accum A x B
*   |         |
*   |    Swap buffer0, buffer1
*   |         |
*   |         |
*   end_loop <-
*         |
*   Load LDS buffer0; Accum A x B
*         |
*   Load Global C Tile
*         |
*   D = alpha * AccumAB + beta * C
*         |
*   Write D Tile
*         |
*         v
*        End
*
* Lds Mapping
* Buffer Width = LDS Width = BlockK
* Matrix geometry for inputs A and B have a common dimension (BlockK).
* We can fix one of the LDS dimensions to BlockK (in this case the width),
* and insert blocks of different heights (BlockM, BlockN) to use the space
* without the need of extra padding.
*
* Fragments of B must be transposed to fit this geometry,
* and both fragments from A and B must accomodate LDS data layout.
*
* Local Layout (LDS):
*
* Non - transposed A fragments [A0 ... AX-1] are placed first and occupy a total height
* of Macro Tile X, where X = number of A blocks and Ck is the kth column of the A block.
*
* Transposed B fragments [B0 (T) ... BY-1 (T)] follow A fragments and occupy a total height of
* Macro Tile Y, where Y = number of B blocks, and Rk is the kth row of the B block.
*
*
*                        _____________BlockK_____________
*                       |                                |
*                       v                                v
*                  (0,0) ----------------------------------->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockM |   | C0 | C1 | C2               | Ck-1 |   A0
*  Tile X  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...          AX-1
*          -->
*          -->       -->  ______________    ...        ______
*          |         |   |    |    |                  |      |
*          |         |   |    |    |                  |      |
*  Macro   |  BlockN |   | R0 | R1 | R2               | Rk-1 |   B0 (T)
*  Tile Y  |         |   |    |    |                  |      |
*          |         --> |___ |___ |____    ...       |______|
*          |         .
*          |         .          ...  ...  ...  ...        BY-1 (T)
*          -->                                           (MacroTileX + MacroTileY - 1, BlockK -1)
*
* Depending on the locality of the block being processed, warps load the corresponding
* A and B inputs from LDS buffer and use them for the accumulation of AxB calculations.
*/

using namespace rocwmma;

///
/// Parameter configuration
///

/* Depending on the GPU architecture this sample is run on, the following kernel parameters need to
*  be modified in order to obtain high performance.
* _________________________________________________________________________________________
*|         |           |           |           |          |          |          |          |
*|         | ROCWMMA_M | ROCWMMA_N | ROCWMMA_K | BLOCKS_X | BLOCKS_Y | TBLOCK_X | TBLOCK_Y |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_9  |    32     |    32     |    16     |    2     |    2     |   128    |    2     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*|         |           |           |           |          |          |          |          |
*|  GFX_11 |    16     |    16     |    16     |    4     |    2     |    64    |    4     |
*|_________|___________|___________|___________|__________|__________|__________|__________|
*
* __________________________________________
*|         |                                |
*|         |           WARP_SIZE            |
*|_________|________________________________|
*|         |                                |
*|  GFX_9  | Constants::AMDGCN_WAVE_SIZE_64 |
*|_________|________________________________|
*|         |                                |
*|  GFX_11 | Constants::AMDGCN_WAVE_SIZE_32 |
*|_________|________________________________|
*/

namespace gfx9Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 32u,
        ROCWMMA_N = 32u,
        ROCWMMA_K = 16u,
        BLOCKS_X  = 2u,
        BLOCKS_Y  = 2u,
        TBLOCK_X  = 128u,
        TBLOCK_Y  = 2u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_64
    };
}

namespace gfx11Params
{
    enum kernelParams : uint32_t
    {
        ROCWMMA_M = 16u,
        ROCWMMA_N = 16u,
        ROCWMMA_K = 16u,
        BLOCKS_X  = 4u,
        BLOCKS_Y  = 2u,
        TBLOCK_X  = 64u,
        TBLOCK_Y  = 4u,
        WARP_SIZE = Constants::AMDGCN_WAVE_SIZE_32
    };
}

//#if(ROCWMMA_ARCH_GFX9)
using namespace gfx9Params;
//#else
//using namespace gfx11Params;
//#endif // defined(ROCWMMA_ARCH_GFX9)

///
/// Types and Data Layouts
///

using InputT   = float16_t;
using OutputT  = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = col_major;
using DataLayoutB   = row_major;
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;

///
/// Fragment types
///

// #if (ROCWMMA_ARCH_GFX9 || ROCWMMA_ARCH_GFX11)
// Warp tile: computed by each warp
constexpr uint32_t WARP_TILE_X = BLOCKS_X * ROCWMMA_M;
constexpr uint32_t WARP_TILE_Y = BLOCKS_Y * ROCWMMA_N;

// Macro Tile: computed by each thread block (workgroup)
// Note: TBLOCK_X must be multiple of WARP_SIZE.
constexpr uint32_t WARPS_X      = TBLOCK_X / WARP_SIZE;
constexpr uint32_t WARPS_Y      = TBLOCK_Y;
constexpr uint32_t MACRO_TILE_X = WARPS_X * WARP_TILE_X;
constexpr uint32_t MACRO_TILE_Y = WARPS_Y * WARP_TILE_Y;

// Mfma frags
using MfmaFragA   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using MfmaFragB   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutB>;
using MfmaFragC   = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, OutputT, DataLayoutC>;
using MfmaFragD   = MfmaFragC;
using MfmaFragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;

// Global read (macro tile)
using GRBuffA = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using GRBuffB = fragment<matrix_b, ROCWMMA_M, MACRO_TILE_Y, ROCWMMA_K, InputT, DataLayoutB>;

// Local write of global buffers (macro tile)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LWBuffA = ApplyDataLayout_t<GRBuffA, DataLayoutLds>;
using LWBuffB = ApplyDataLayout_t<ApplyTranspose_t<GRBuffB>, DataLayoutLds>;

// Local read (mfma frags)
// - Must match Lds data layout.
// - Lds has transposed B frags.
using LRFragA = ApplyDataLayout_t<MfmaFragA, DataLayoutLds>;
using LRFragB = ApplyDataLayout_t<ApplyTranspose_t<MfmaFragB>, DataLayoutLds>;
// #endif // (ROCWMMA_ARCH_GFX9 || ROCWMMA_ARCH_GFX11)

///
/// Wrapper functions: repeat mfma tile operations across entire warp tile.
///

// Cooperative global read / local write (Macro tile data movement)
// Loads / stores a global data fragment cooperatively across warps. Each participating warp is
// responsible for only a portion of the whole fragment.
//
// The cooperative operation is split into work items (SplitCount). Work items are consumed in
// a round robin fashion by warps in the range of [0, WaveCount). The wave index determines the
// order of the current wave in the collaboration pool.
//
// WaveCount, SplitCount and waveIndex parameters must match successive coop load / store calls
// to ensure the entire fragment remains coherent.

// Global A reads in cooperative mode (macro tile)
template <uint32_t WaveCountA>
ROCWMMA_DEVICE static inline void
    globalReadCoopA(GRBuffA& grBuffA, InputT const* gAddrA, uint32_t lda, uint32_t waveIndexA)
{
    load_matrix_coop_sync<WaveCountA>(grBuffA, gAddrA, lda, waveIndexA);
}

// Global B reads in cooperative mode (macro tile)
template <uint32_t WaveCountB>
ROCWMMA_DEVICE static inline void
    globalReadCoopB(GRBuffB& grBuffB, InputT const* gAddrB, uint32_t ldb, uint32_t waveIndexB)
{
    load_matrix_coop_sync<WaveCountB>(grBuffB, gAddrB, ldb, waveIndexB);
}

// Local A writes in cooperative mode (macro tile)
template <uint32_t WaveCountA>
ROCWMMA_DEVICE static inline void
    localWriteCoopA(InputT* ldsAddr, GRBuffA const& grBuffA, uint32_t ldsld, uint32_t waveIndexA)
{
    // No transpose, but apply the lds data layout
    store_matrix_coop_sync<WaveCountA>(
        ldsAddr, applyDataLayout<DataLayoutLds, WaveCountA>(grBuffA), ldsld, waveIndexA);
}

// Local B writes in cooperative mode (macro tile)
template <uint32_t WaveCountB>
ROCWMMA_DEVICE static inline void
    localWriteCoopB(InputT* ldsAddr, GRBuffB const& grBuffB, uint32_t ldsld, uint32_t waveIndexB)
{
    // Transpose B and then apply lds data layout
    store_matrix_coop_sync<WaveCountB>(
        ldsAddr,
        applyDataLayout<DataLayoutLds, WaveCountB>(applyTranspose(grBuffB)),
        ldsld,
        waveIndexB);
}

// Global read (macro tile)
using LRBuffA = fragment<matrix_a, WARP_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutLds>;
using LRBuffB = ApplyTranspose_t<LRBuffA>;
using GRBuffC = fragment<accumulator, WARP_TILE_X, WARP_TILE_Y, ROCWMMA_K, OutputT, DataLayoutC>;
using AccumBuffInt = fragment<accumulator, WARP_TILE_X, WARP_TILE_Y, ROCWMMA_K, ComputeT>;

ROCWMMA_DEVICE static inline void
    localReadA(LRBuffA& fragsA, InputT const* ldsAddrA, uint32_t ldsld)
{
    constexpr uint32_t VW = 4;

    using Profile
        = rocwmma::LayoutProfile::ColInt<WARP_TILE_X, ROCWMMA_K, float16_t, DataLayoutLds, 16u, 1u>;

    using DataLayout   = typename Profile::DataLayout;
    using MatrixLayout = typename Profile::MatrixLayout;

    using Loader = OpaqueLoad<WARP_TILE_X, ROCWMMA_K, float16_t, DataLayout, MatrixLayout, VW>;

    // Load then implicit pack
    Loader::exec(fragsA.mAccess, ldsAddrA, ldsld);

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63)
    // {
    //     auto reg = 0u;
    //     auto x0 = fragsA.mAccess.data[0];
    //     auto x1 = fragsA.mAccess.data[1];
    //     auto x2 = fragsA.mAccess.data[2];
    //     auto x3 = fragsA.mAccess.data[3];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }
}

// Local B reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void
    localReadB(LRBuffB& fragsB, InputT const* ldsAddrB, uint32_t ldsld)
{
    // How to choose? Comes from the IOConfig?
    constexpr uint32_t VW = 4;

    using Profile
        = rocwmma::LayoutProfile::ColInt<WARP_TILE_X, ROCWMMA_K, float16_t, DataLayoutLds, 16u, 1u>;

    using MatrixLayout = typename Profile::MatrixLayout;
    using DataLayout   = typename Profile::DataLayout;

    using Loader = OpaqueLoad<WARP_TILE_X, ROCWMMA_K, float16_t, DataLayout, MatrixLayout, VW>;

    // Load then implicit pack
    Loader::exec(reinterpret_cast<LRBuffB&>(fragsB).mAccess, ldsAddrB, ldsld);

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63)
    // {
    //     auto reg = 0u;
    //     auto x0 = fragsB.mAccess.data[0];
    //     auto x1 = fragsB.mAccess.data[1];
    //     auto x2 = fragsB.mAccess.data[2];
    //     auto x3 = fragsB.mAccess.data[3];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }
}

// Global C reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void globalReadC(GRBuffC& fragsC, OutputT const* gAddrC, uint32_t ldc)
{
    // How to choose? Comes from the IOConfig?
    constexpr uint32_t VW = 4;

    using Profile
        = rocwmma::LayoutProfile::RowInt<WARP_TILE_Y, WARP_TILE_X, OutputT, DataLayoutC, 16u, 1u>;

    using MatrixLayout = typename Profile::MatrixLayout;
    using DataLayout   = typename Profile::DataLayout;

    using Loader = OpaqueLoad<WARP_TILE_Y, WARP_TILE_X, OutputT, DataLayout, MatrixLayout, VW>;

    // Load then implicit pack
    GRBuffC tmp;
    Loader::exec(tmp.mAccess, gAddrC, ldc);

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     auto reg = 0u;
    //     auto x0 = tmp.mAccess.data[0];
    //     auto x1 = tmp.mAccess.data[1];
    //     auto x2 = tmp.mAccess.data[2];
    //     auto x3 = tmp.mAccess.data[3];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }
    //MatrixLayout::debug();
    {
        // Post load to accum format

#pragma unroll
        for(int i = 0; i < 4u; i++)
        {
            fragsC.mAccess.data[0 * 16 + 0 + i] = tmp.mAccess.data[i * 16 + 0 + 0];
            fragsC.mAccess.data[1 * 16 + 0 + i] = tmp.mAccess.data[i * 16 + 0 + 1];
            fragsC.mAccess.data[2 * 16 + 0 + i] = tmp.mAccess.data[i * 16 + 0 + 2];
            fragsC.mAccess.data[3 * 16 + 0 + i] = tmp.mAccess.data[i * 16 + 0 + 3];

            fragsC.mAccess.data[0 * 16 + 4 + i] = tmp.mAccess.data[i * 16 + 4 + 0];
            fragsC.mAccess.data[1 * 16 + 4 + i] = tmp.mAccess.data[i * 16 + 4 + 1];
            fragsC.mAccess.data[2 * 16 + 4 + i] = tmp.mAccess.data[i * 16 + 4 + 2];
            fragsC.mAccess.data[3 * 16 + 4 + i] = tmp.mAccess.data[i * 16 + 4 + 3];

            fragsC.mAccess.data[0 * 16 + 8 + i] = tmp.mAccess.data[i * 16 + 8 + 0];
            fragsC.mAccess.data[1 * 16 + 8 + i] = tmp.mAccess.data[i * 16 + 8 + 1];
            fragsC.mAccess.data[2 * 16 + 8 + i] = tmp.mAccess.data[i * 16 + 8 + 2];
            fragsC.mAccess.data[3 * 16 + 8 + i] = tmp.mAccess.data[i * 16 + 8 + 3];

            fragsC.mAccess.data[0 * 16 + 12 + i] = tmp.mAccess.data[i * 16 + 12 + 0];
            fragsC.mAccess.data[1 * 16 + 12 + i] = tmp.mAccess.data[i * 16 + 12 + 1];
            fragsC.mAccess.data[2 * 16 + 12 + i] = tmp.mAccess.data[i * 16 + 12 + 2];
            fragsC.mAccess.data[3 * 16 + 12 + i] = tmp.mAccess.data[i * 16 + 12 + 3];
        }
    }

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     auto reg = 0u;
    //     auto x0 = fragsC.mAccess.data[12];
    //     auto x1 = fragsC.mAccess.data[13];
    //     auto x2 = fragsC.mAccess.data[14];
    //     auto x3 = fragsC.mAccess.data[15];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }
}

// Global D reads for warp tile gemm, non-cooperative
ROCWMMA_DEVICE static inline void globalWriteD(OutputT* gAddrD, GRBuffC const& fragsD, uint32_t ldd)
{
    // How to choose? Comes from the IOConfig?
    constexpr uint32_t VW = 4;

    using Profile
        = rocwmma::LayoutProfile::RowInt<WARP_TILE_Y, WARP_TILE_X, OutputT, DataLayoutC, 16u, 1u>;

    using MatrixLayout = typename Profile::MatrixLayout;
    using DataLayout   = typename Profile::DataLayout;

    using Storer = OpaqueStore<WARP_TILE_Y, WARP_TILE_X, OutputT, DataLayout, MatrixLayout, VW>;

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     auto reg = 0u;
    //     auto x0 = fragsD.mAccess.data[0];
    //     auto x1 = fragsD.mAccess.data[16];
    //     auto x2 = fragsD.mAccess.data[32];
    //     auto x3 = fragsD.mAccess.data[48];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }

    // Pre-store to output fmt
    GRBuffC tmp;
    // tmp.mAccess.data[0] = fragsD.mAccess.data[0];
    // tmp.mAccess.data[1] = fragsD.mAccess.data[16];
    // tmp.mAccess.data[2] = fragsD.mAccess.data[32];
    // tmp.mAccess.data[3] = fragsD.mAccess.data[48];
    // tmp.mAccess.data[4] = fragsD.mAccess.data[4];
    // tmp.mAccess.data[5] = fragsD.mAccess.data[20];
    // tmp.mAccess.data[6] = fragsD.mAccess.data[36];
    // tmp.mAccess.data[7] = fragsD.mAccess.data[52];
    // tmp.mAccess.data[8] = fragsD.mAccess.data[8];
    // tmp.mAccess.data[9] = fragsD.mAccess.data[24];
    // tmp.mAccess.data[10] = fragsD.mAccess.data[40];
    // tmp.mAccess.data[11] = fragsD.mAccess.data[56];
    // tmp.mAccess.data[12] = fragsD.mAccess.data[12];
    // tmp.mAccess.data[13] = fragsD.mAccess.data[28];
    // tmp.mAccess.data[14] = fragsD.mAccess.data[44];
    // tmp.mAccess.data[15] = fragsD.mAccess.data[60];
    // tmp.mAccess.data[16] = fragsD.mAccess.data[1];
    // tmp.mAccess.data[17] = fragsD.mAccess.data[17];
    // tmp.mAccess.data[18] = fragsD.mAccess.data[33];
    // tmp.mAccess.data[19] = fragsD.mAccess.data[49];
    // tmp.mAccess.data[20] = fragsD.mAccess.data[5];
    // tmp.mAccess.data[21] = fragsD.mAccess.data[21];
    // tmp.mAccess.data[22] = fragsD.mAccess.data[37];
    // tmp.mAccess.data[23] = fragsD.mAccess.data[53];
    // tmp.mAccess.data[24] = fragsD.mAccess.data[9];
    // tmp.mAccess.data[25] = fragsD.mAccess.data[25];
    // tmp.mAccess.data[26] = fragsD.mAccess.data[41];
    // tmp.mAccess.data[27] = fragsD.mAccess.data[57];
    // tmp.mAccess.data[28] = fragsD.mAccess.data[13];
    // tmp.mAccess.data[29] = fragsD.mAccess.data[29];
    // tmp.mAccess.data[30] = fragsD.mAccess.data[45];
    // tmp.mAccess.data[31] = fragsD.mAccess.data[61];
    // tmp.mAccess.data[32] = fragsD.mAccess.data[2];
    // tmp.mAccess.data[33] = fragsD.mAccess.data[18];
    // tmp.mAccess.data[34] = fragsD.mAccess.data[34];
    // tmp.mAccess.data[35] = fragsD.mAccess.data[50];
    // tmp.mAccess.data[36] = fragsD.mAccess.data[6];
    // tmp.mAccess.data[37] = fragsD.mAccess.data[22];
    // tmp.mAccess.data[38] = fragsD.mAccess.data[38];
    // tmp.mAccess.data[39] = fragsD.mAccess.data[54];
    // tmp.mAccess.data[40] = fragsD.mAccess.data[10];
    // tmp.mAccess.data[41] = fragsD.mAccess.data[26];
    // tmp.mAccess.data[42] = fragsD.mAccess.data[42];
    // tmp.mAccess.data[43] = fragsD.mAccess.data[58];
    // tmp.mAccess.data[44] = fragsD.mAccess.data[14];
    // tmp.mAccess.data[45] = fragsD.mAccess.data[30];
    // tmp.mAccess.data[46] = fragsD.mAccess.data[46];
    // tmp.mAccess.data[47] = fragsD.mAccess.data[62];
    // tmp.mAccess.data[48] = fragsD.mAccess.data[3];
    // tmp.mAccess.data[49] = fragsD.mAccess.data[19];
    // tmp.mAccess.data[50] = fragsD.mAccess.data[35];
    // tmp.mAccess.data[51] = fragsD.mAccess.data[51];
    // tmp.mAccess.data[52] = fragsD.mAccess.data[7];
    // tmp.mAccess.data[53] = fragsD.mAccess.data[23];
    // tmp.mAccess.data[54] = fragsD.mAccess.data[39];
    // tmp.mAccess.data[55] = fragsD.mAccess.data[55];
    // tmp.mAccess.data[56] = fragsD.mAccess.data[11];
    // tmp.mAccess.data[57] = fragsD.mAccess.data[27];
    // tmp.mAccess.data[58] = fragsD.mAccess.data[43];
    // tmp.mAccess.data[59] = fragsD.mAccess.data[59];
    // tmp.mAccess.data[60] = fragsD.mAccess.data[15];
    // tmp.mAccess.data[61] = fragsD.mAccess.data[31];
    // tmp.mAccess.data[62] = fragsD.mAccess.data[47];
    // tmp.mAccess.data[63] = fragsD.mAccess.data[63];
#pragma unroll
    for(int i = 0; i < 4u; i++)
    {
        tmp.mAccess.data[i * 16 + 0 + 0] = fragsD.mAccess.data[0 * 16 + 0 + i];
        tmp.mAccess.data[i * 16 + 0 + 1] = fragsD.mAccess.data[1 * 16 + 0 + i];
        tmp.mAccess.data[i * 16 + 0 + 2] = fragsD.mAccess.data[2 * 16 + 0 + i];
        tmp.mAccess.data[i * 16 + 0 + 3] = fragsD.mAccess.data[3 * 16 + 0 + i];

        tmp.mAccess.data[i * 16 + 4 + 0] = fragsD.mAccess.data[0 * 16 + 4 + i];
        tmp.mAccess.data[i * 16 + 4 + 1] = fragsD.mAccess.data[1 * 16 + 4 + i];
        tmp.mAccess.data[i * 16 + 4 + 2] = fragsD.mAccess.data[2 * 16 + 4 + i];
        tmp.mAccess.data[i * 16 + 4 + 3] = fragsD.mAccess.data[3 * 16 + 4 + i];

        tmp.mAccess.data[i * 16 + 8 + 0] = fragsD.mAccess.data[0 * 16 + 8 + i];
        tmp.mAccess.data[i * 16 + 8 + 1] = fragsD.mAccess.data[1 * 16 + 8 + i];
        tmp.mAccess.data[i * 16 + 8 + 2] = fragsD.mAccess.data[2 * 16 + 8 + i];
        tmp.mAccess.data[i * 16 + 8 + 3] = fragsD.mAccess.data[3 * 16 + 8 + i];

        tmp.mAccess.data[i * 16 + 12 + 0] = fragsD.mAccess.data[0 * 16 + 12 + i];
        tmp.mAccess.data[i * 16 + 12 + 1] = fragsD.mAccess.data[1 * 16 + 12 + i];
        tmp.mAccess.data[i * 16 + 12 + 2] = fragsD.mAccess.data[2 * 16 + 12 + i];
        tmp.mAccess.data[i * 16 + 12 + 3] = fragsD.mAccess.data[3 * 16 + 12 + i];
    }

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     if(threadIdx.x == 0)
    //     {
    //         printf("D Before STORE\n");
    //         printf("Count: %d\n", tmp.num_elements);
    //     }
    //     auto reg = 0u;
    //     auto x0 = tmp.mAccess.data[0];
    //     auto x1 = tmp.mAccess.data[16];
    //     auto x2 = tmp.mAccess.data[32];
    //     auto x3 = tmp.mAccess.data[48];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }

    // Load then implicit pack
    Storer::exec(gAddrD, tmp.mAccess, ldd);
}

// Performs warp tile mfma
ROCWMMA_DEVICE static inline void mfma(AccumBuffInt&       fragsAccOut,
                                       LRBuffA const&      fragsA,
                                       LRBuffB const&      fragsB,
                                       AccumBuffInt const& fragsAccIn)
{
    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     auto x0 = fragsA.mAccess.data[0];
    //     auto x1 = fragsA.mAccess.data[1];
    //     auto x2 = fragsA.mAccess.data[2];
    //     auto x3 = fragsA.mAccess.data[3];
    //     printf("(A)Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));

    //     x0 = fragsB.mAccess.data[0];
    //     x1 = fragsB.mAccess.data[1];
    //     x2 = fragsB.mAccess.data[2];
    //     x3 = fragsB.mAccess.data[3];
    //     printf("(B)Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }
    // Need to get the MFMA tile size from the IO traits somehow
    constexpr static uint32_t MFMFA_TILE = 16u;

    // From here, need to 'unpack' the interleaved data
    // Should be 16 registers, need to re-order them in groups of 4
    LRBuffA tmpA;
    LRBuffB tmpB;
#pragma unroll
    for(int i = 0; i < 4u; i++)
    {
        tmpA.mAccess.data[i * 4 + 0] = fragsA.mAccess.data[0 * 4 + i];
        tmpA.mAccess.data[i * 4 + 1] = fragsA.mAccess.data[1 * 4 + i];
        tmpA.mAccess.data[i * 4 + 2] = fragsA.mAccess.data[2 * 4 + i];
        tmpA.mAccess.data[i * 4 + 3] = fragsA.mAccess.data[3 * 4 + i];

        tmpB.mAccess.data[i * 4 + 0] = fragsB.mAccess.data[0 * 4 + i];
        tmpB.mAccess.data[i * 4 + 1] = fragsB.mAccess.data[1 * 4 + i];
        tmpB.mAccess.data[i * 4 + 2] = fragsB.mAccess.data[2 * 4 + i];
        tmpB.mAccess.data[i * 4 + 3] = fragsB.mAccess.data[3 * 4 + i];
    }

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     auto x0 = tmpA.mAccess.data[12];
    //     auto x1 = tmpA.mAccess.data[13];
    //     auto x2 = tmpA.mAccess.data[14];
    //     auto x3 = tmpA.mAccess.data[15];
    //     printf("(A)Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));

    //     x0 = tmpB.mAccess.data[12];
    //     x1 = tmpB.mAccess.data[13];
    //     x2 = tmpB.mAccess.data[14];
    //     x3 = tmpB.mAccess.data[15];
    //     printf("(B)Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }

    // Iterate over MFMA input requirements
    // A = 16 regs unpacked, 8 packed
    // B = 16 regs unpacked, 8 packed
    // Accum = 64 regs unpacked/packed
    // MFMA blocks = 16 x 4 regs
    // Iterate through A - major
    auto       bIt        = makeVectorIterator<2u>(tmpB.mStorage).begin();
    auto const accumInIt  = makeVectorIterator<4u>(fragsAccOut.mStorage).begin();
    auto       accumOutIt = makeVectorIterator<4u>(fragsAccOut.mStorage).begin();

    using MMA = Mfma<float16_t, float32_t, 16, 16, 16>;

#pragma unroll
    for(int j = 0; j < 4u; j++)
    {
        auto aIt = makeVectorIterator<2u>(tmpA.mStorage).begin();
#pragma unroll
        for(int i = 0; i < 4u; i++)
        {
            // mma functions operate on packed vectors
            *accumOutIt = MMA::exec(*aIt, *bIt, *accumInIt);
            aIt++;
            accumInIt++;
            accumOutIt++;
        }
        bIt++;
    }

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     if(threadIdx.x == 0)
    //     {
    //         printf("Count: %d\n", fragsAccOut.num_elements);
    //     }
    //     auto reg = 0u;
    //     auto x0 = fragsAccOut.mAccess.data[0];
    //     auto x1 = fragsAccOut.mAccess.data[1];
    //     auto x2 = fragsAccOut.mAccess.data[2];
    //     auto x3 = fragsAccOut.mAccess.data[3];
    //     printf("Thread %d: %#010x %#010x %#010x %#010x\n", threadIdx.x, reinterpret_cast<uint32_t&>(x0), reinterpret_cast<uint32_t&>(x1), reinterpret_cast<uint32_t&>(x2), reinterpret_cast<uint32_t&>(x3));
    // }
}

// Broadcast value to fragments in warp tile
template <typename FragT>
ROCWMMA_DEVICE static inline void fill(FragT& frags, GetDataType_t<FragT> value)
{
    fill_fragment(frags, value);
}

// Uniform multiply - add (FMA)
// Performs D = alpha * acc + beta * C, where alpha, beta are uniform scalars
ROCWMMA_DEVICE static inline void uniformFma(GRBuffC&            fragsD,
                                             ComputeT            alpha,
                                             AccumBuffInt const& fragsAcc,
                                             ComputeT            beta,
                                             GRBuffC const&      fragsC)
{

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     if(threadIdx.x == 0)
    //     {
    //         printf("Count: %d\n", fragsAcc.num_elements);
    //     }
    //     auto reg = 0u;
    //     auto x0 = fragsAcc.mAccess.data[0];
    //     auto x1 = fragsAcc.mAccess.data[1];
    //     auto x2 = fragsAcc.mAccess.data[2];
    //     auto x3 = fragsAcc.mAccess.data[3];
    //     printf("Thread %d: %#010x %#010x %#010x %#010x\n", threadIdx.x, reinterpret_cast<uint32_t&>(x0), reinterpret_cast<uint32_t&>(x1), reinterpret_cast<uint32_t&>(x2), reinterpret_cast<uint32_t&>(x3));
    // }

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     if(threadIdx.x == 0)
    //     {
    //         printf("Count: %d\n", fragsC.num_elements);
    //     }
    //     auto reg = 0u;
    //     auto x0 = fragsC.mAccess.data[0];
    //     auto x1 = fragsC.mAccess.data[1];
    //     auto x2 = fragsC.mAccess.data[2];
    //     auto x3 = fragsC.mAccess.data[3];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }

    static constexpr uint32_t ChunkFactor = 2u;
    static constexpr uint32_t ChunkSize   = 64u / ChunkFactor;
    auto                      dIt         = makeVectorIterator<ChunkSize>(fragsD.mAccess).begin();
    auto const                accumIt     = makeVectorIterator<ChunkSize>(fragsAcc.mAccess).begin();
    auto const                cIt         = makeVectorIterator<ChunkSize>(fragsC.mAccess).begin();

    for(int k = 0; k < fragsD.num_elements / ChunkFactor; k++)
    {
        // Perform computation in ComputeT and cast back to OutputT
        (*dIt).data[k] = static_cast<OutputT>(alpha * (*accumIt).data[k]
                                              + beta * static_cast<ComputeT>((*cIt).data[k]));
    }

    dIt++;
    accumIt++;
    cIt++;

    for(int k = 0; k < fragsD.num_elements / ChunkFactor; k++)
    {
        // Perform computation in ComputeT and cast back to OutputT
        (*dIt).data[k] = static_cast<OutputT>(alpha * (*accumIt).data[k]
                                              + beta * static_cast<ComputeT>((*cIt).data[k]));
    }

    // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x <= 63 && threadIdx.y == 0)
    // {
    //     if(threadIdx.x == 0)
    //     {
    //         printf("D AFTER UNIFORM FMA\n");
    //         printf("Count: %d\n", fragsD.num_elements);
    //     }
    //     auto reg = 0u;
    //     auto x0 = fragsD.mAccess.data[0];
    //     auto x1 = fragsD.mAccess.data[16];
    //     auto x2 = fragsD.mAccess.data[32];
    //     auto x3 = fragsD.mAccess.data[48];
    //     printf("Thread %d: %#06x %#06x %#06x %#06x\n", threadIdx.x, reinterpret_cast<uint16_t&>(x0), reinterpret_cast<uint16_t&>(x1), reinterpret_cast<uint16_t&>(x2), reinterpret_cast<uint16_t&>(x3));
    // }
}

//ROCWMMA_KERNEL void  gemm_rocwmma_d(uint32_t       m,
//ROCWMMA_KERNEL void __attribute__((amdgpu_num_vgpr(0)))  gemm_rocwmma_d(uint32_t       m,
ROCWMMA_KERNEL void __launch_bounds__(1024) gemm_rocwmma_d(
    uint32_t m,
    //ROCWMMA_KERNEL void __attribute__((amdgpu_waves_per_eu(1))) gemm_rocwmma_d(uint32_t       m,
    uint32_t       n,
    uint32_t       k,
    InputT const*  a,
    InputT const*  b,
    OutputT const* c,
    OutputT*       d,
    uint32_t       lda,
    uint32_t       ldb,
    uint32_t       ldc,
    uint32_t       ldd,
    ComputeT       alpha,
    ComputeT       beta)
{
    if constexpr(!ROCWMMA_ARCH_HOST)
    {
        ///
        /// 2D matrix coordinate setup
        ///

        // Tile Sizes
        constexpr auto warpTileSize  = make_coord2d(WARP_TILE_X, WARP_TILE_Y);
        constexpr auto macroTileSize = make_coord2d(MACRO_TILE_X, MACRO_TILE_Y);

        // Local warp coordinate relative to current threadblock (wg).
        constexpr auto warpDims        = make_coord2d(WARPS_X, WARPS_Y);
        auto           localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);
        auto           localWarpOffset = localWarpCoord * warpTileSize;

        // Global matrix coordinates for C/D
        auto macroTileCoord = make_coord2d(blockIdx.x, blockIdx.y) * macroTileSize;
        auto warpTileCoord  = macroTileCoord + localWarpOffset;

        // Bounds check
        auto warpTileBound = warpTileCoord + warpTileSize;
        if(get<0>(warpTileBound) > m || get<1>(warpTileBound) > n)
        {
            return;
        }

        ///
        /// 1D global read coordinate setup
        ///
        using GRBuffAMap1d = GetDataLayout_t<GRBuffA>;
        using GRBuffBMap1d = GetDataLayout_t<GRBuffB>;

        // Initial globa read address offsets
        auto globalReadOffsetA
            = GRBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(macroTileCoord), 0u), lda);
        auto globalReadOffsetB
            = GRBuffBMap1d::fromMatrixCoord(make_coord2d(0u, get<1>(macroTileCoord)), ldb);

        // Incremental global read address offsets
        auto kStepOffsetA = GRBuffAMap1d::fromMatrixCoord(make_coord2d(0u, ROCWMMA_K), lda);
        auto kStepOffsetB = GRBuffBMap1d::fromMatrixCoord(make_coord2d(ROCWMMA_K, 0u), ldb);

        ///
        /// Cooperative config for global read A / B
        ///

        // WorkItems will be split up by minimum IOCount to perform either global read or local write.
        // These are inputs to cooperative functions.
        constexpr auto warpCount = get<0>(warpDims) * get<1>(warpDims);

        // Scheduling warp order is analogous to row major priority.
        // E.g. Wg = (128, 2) = 2x2 warps
        // (0, 0)   (0, 1)   Share Schedule: w0 = (0, 0), w1 = (0, 1),
        // (1, 0)   (1, 1)                   w2 = (1, 0), w3 = (1, 1), count = 4
        const auto warpIndex = get<0>(localWarpCoord) * get<1>(warpDims) + get<1>(localWarpCoord);

        ///
        /// Perform initial global pre-fetch
        ///

        GRBuffA grBuffA;
        GRBuffB grBuffB;

        globalReadCoopA<warpCount>(grBuffA, a + globalReadOffsetA, lda, warpIndex);
        globalReadCoopB<warpCount>(grBuffB, b + globalReadOffsetB, ldb, warpIndex);

        globalReadOffsetA += kStepOffsetA;
        globalReadOffsetB += kStepOffsetB;

        ///
        /// Setup LDS addressing
        /// This kernel will use 2 separate LDS blocks for pipelining
        /// the input prefetching during the accumulation loop
        ///

        HIP_DYNAMIC_SHARED(void*, localMemPtr);
        using LWBuffAShape = GetIOShape_t<LWBuffA>;
        using LWBuffBShape = GetIOShape_t<LWBuffB>;
        using LWBuffAMap1d = GetDataLayout_t<LWBuffA>;
        using LWBuffBMap1d = GetDataLayout_t<LWBuffB>;

        constexpr uint32_t ldsWidth  = ROCWMMA_K;
        constexpr uint32_t ldsHeight = LWBuffAShape::BlockHeight + LWBuffBShape::BlockHeight;
        constexpr uint32_t sizeLds   = ldsHeight * ldsWidth;
        constexpr uint32_t ldsld = std::is_same_v<DataLayoutLds, row_major> ? ldsWidth : ldsHeight;

        auto* ldsPtrLo = reinterpret_cast<InputT*>(localMemPtr);
        auto* ldsPtrHi = ldsPtrLo + sizeLds;

        // Local write offsets to start of A / B data
        auto ldsWriteOffsetA = 0u;
        auto ldsWriteOffsetB
            = LWBuffAMap1d::fromMatrixCoord(make_coord2d(LWBuffAShape::BlockHeight, 0u), ldsld);

        // Local read offsets for mfma frags
        auto ldsReadOffsetA
            = ldsWriteOffsetA
              + LWBuffAMap1d::fromMatrixCoord(make_coord2d(get<0>(localWarpOffset), 0u), ldsld);
        auto ldsReadOffsetB
            = ldsWriteOffsetB
              + LWBuffBMap1d::fromMatrixCoord(make_coord2d(get<1>(localWarpOffset), 0u), ldsld);

        ///
        /// Write prefetch to local
        ///
        localWriteCoopA<warpCount>(ldsPtrLo + ldsWriteOffsetA, grBuffA, ldsld, warpIndex);
        localWriteCoopB<warpCount>(ldsPtrLo + ldsWriteOffsetB, grBuffB, ldsld, warpIndex);

        ///
        /// Initialize accumulation frags
        ///
        AccumBuffInt fragsAcc;
        fill(fragsAcc, 0.0f);

        ///
        /// Synchronize warps and memory
        ///
        synchronize_workgroup();

        ///
        /// Accumulate A * B for all mfma frags in warp tile
        ///
        // - LDS Triple buffer
        // - LDS no buffer-> tiny m/n large K
        // - unroll K to have more work
        // - __restrict__
        //
        for(uint32_t currentK = ROCWMMA_K; currentK < k; currentK += ROCWMMA_K)
        {
            // Make sure that all waves have finished reading / writing to lds for currentK.
            synchronize_workgroup();

            // Prefetch next round of global frags
            globalReadCoopA<warpCount>(grBuffA, a + globalReadOffsetA, lda, warpIndex);
            globalReadCoopB<warpCount>(grBuffB, b + globalReadOffsetB, ldb, warpIndex);

            LRBuffA fragsA;
            LRBuffB fragsB;

            // Local read mfma frags from first LDS buffer
            localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
            localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);

            // Advance offsets to next k step
            globalReadOffsetA += kStepOffsetA;
            globalReadOffsetB += kStepOffsetB;

            // accum(A * B)
            mfma(fragsAcc, fragsA, fragsB, fragsAcc);

            // Write prefetch to second LDS buffer
            localWriteCoopA<warpCount>(ldsPtrHi + ldsWriteOffsetA, grBuffA, ldsld, warpIndex);
            localWriteCoopB<warpCount>(ldsPtrHi + ldsWriteOffsetB, grBuffB, ldsld, warpIndex);

            // Swap Lds buffers
            auto* tmp = ldsPtrLo;
            ldsPtrLo  = ldsPtrHi;
            ldsPtrHi  = tmp;

            // Scheduling

            // // VMEM read
            // __builtin_amdgcn_sched_group_barrier(32, 2, 0);
            // // DS read
            // __builtin_amdgcn_sched_group_barrier(256, 16, 0);
            // // Non-VMEM
            // __builtin_amdgcn_sched_group_barrier(1, 16, 0);
            // // MFMA
            // __builtin_amdgcn_sched_group_barrier(8, 4, 0);
            // // DS read
            // __builtin_amdgcn_sched_group_barrier(256, 16, 1);
            // // // Non-VMEM
            // __builtin_amdgcn_sched_group_barrier(1, 16, 1);
            // // MFMA
            // __builtin_amdgcn_sched_group_barrier(8, 4, 1);
            // // DS write
            // __builtin_amdgcn_sched_group_barrier(512, 32, 0);

            ////////// Works good - 127.46
            // VMEM read
            __builtin_amdgcn_sched_group_barrier(32, 4, 0);
            // DS read
            __builtin_amdgcn_sched_group_barrier(256, 64, 0);
            // SALU
            __builtin_amdgcn_sched_group_barrier(4, 256, 0);
            // VALU
            __builtin_amdgcn_sched_group_barrier(2, 256, 0);
            // MFMA
            __builtin_amdgcn_sched_group_barrier(8, 16, 0);
            // DS write
            __builtin_amdgcn_sched_group_barrier(512, 64, 0);
            //////////////////
        }

        // Make sure that all waves have finished reading / writing to lds for currentK.
        synchronize_workgroup();

        ///
        /// Start loading C
        ///
        using MfmaFragCMap1d = GetDataLayout_t<MfmaFragC>;
        using MfmaFragDMap1d = GetDataLayout_t<MfmaFragD>;

        GRBuffC fragsC;
        globalReadC(fragsC, c + MfmaFragCMap1d::fromMatrixCoord(warpTileCoord, ldc), ldc);

        // ///
        // /// Clean up tail A * B
        // ///
        LRBuffA fragsA;
        LRBuffB fragsB;

        // // Local read mfma frags
        localReadA(fragsA, ldsPtrLo + ldsReadOffsetA, ldsld);
        localReadB(fragsB, ldsPtrLo + ldsReadOffsetB, ldsld);
        mfma(fragsAcc, fragsA, fragsB, fragsAcc);

        // ///
        // /// D = alpha * accum + beta * C
        // ///
        GRBuffC fragsD;
        uniformFma(fragsD, alpha, fragsAcc, beta, fragsC);
        //globalWriteD(d + MfmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd), reinterpret_cast<GRBuffC&>(fragsAcc), ldd);
        globalWriteD(d + MfmaFragDMap1d::fromMatrixCoord(warpTileCoord, ldd), fragsD, ldd);

        ////////// Works good - 127.46
        // DS read
        __builtin_amdgcn_sched_group_barrier(256, 64, 0);
        // VMEM read
        __builtin_amdgcn_sched_group_barrier(32, 64, 0);

        // MFMA
        __builtin_amdgcn_sched_group_barrier(8, 16, 0);
        // SALU
        __builtin_amdgcn_sched_group_barrier(4, 256, 0);
        // VALU
        __builtin_amdgcn_sched_group_barrier(2, 512, 0);

        // VMEM write
        __builtin_amdgcn_sched_group_barrier(512, 64, 0);
        //////////////////
    }
}

ROCWMMA_HOST void gemm_test(uint32_t m, uint32_t n, uint32_t k, ComputeT alpha, ComputeT beta)
{
    // Runtime checks for host parameters
    uint32_t hTBLOCK_X    = isGfx9() ? gfx9Params::TBLOCK_X : gfx11Params::TBLOCK_X;
    uint32_t hTBLOCK_Y    = isGfx9() ? gfx9Params::TBLOCK_Y : gfx11Params::TBLOCK_Y;
    uint32_t hBLOCKS_X    = isGfx9() ? gfx9Params::BLOCKS_X : gfx11Params::BLOCKS_X;
    uint32_t hBLOCKS_Y    = isGfx9() ? gfx9Params::BLOCKS_Y : gfx11Params::BLOCKS_Y;
    uint32_t hROCWMMA_M   = isGfx9() ? gfx9Params::ROCWMMA_M : gfx11Params::ROCWMMA_M;
    uint32_t hROCWMMA_N   = isGfx9() ? gfx9Params::ROCWMMA_N : gfx11Params::ROCWMMA_N;
    uint32_t hROCWMMA_K   = isGfx9() ? gfx9Params::ROCWMMA_K : gfx11Params::ROCWMMA_K;
    uint32_t hWARP_TILE_X = hBLOCKS_X * hROCWMMA_M;
    uint32_t hWARP_TILE_Y = hBLOCKS_Y * hROCWMMA_N;

    // Runtime warp calculation (host code needs to query warpsize dynamically)
    auto warpSize = getWarpSize();
    auto macroTileSize
        = rocwmma::make_coord2d(hTBLOCK_X / warpSize * hWARP_TILE_X, hTBLOCK_Y * hWARP_TILE_Y);

    // Device check for supported block and wave sizes
    if((isGfx11() || isGfx12()) && (hROCWMMA_M != 16 || hROCWMMA_N != 16))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if(isGfx9() && (hROCWMMA_M != hROCWMMA_N) || (hROCWMMA_M != 16 && hROCWMMA_M != 32))
    {
        std::cout << "Unsupported block size!\n";
        return;
    }

    if((isGfx11() || isGfx12()) && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_32)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    if(isGfx9() && getWarpSize() != Constants::AMDGCN_WAVE_SIZE_64)
    {
        std::cout << "Unsupported wave size!\n";
        return;
    }

    // Bounds check
    if((m < get<0>(macroTileSize) || n < get<1>(macroTileSize) || k < hROCWMMA_K)
       || (m % hROCWMMA_M || n % hROCWMMA_N || k % hROCWMMA_K))
    {
        std::cout << "Unsupported matrix size!\n";
        return;
    }

    // Layouts leading dims
    int lda = std::is_same_v<DataLayoutA, row_major> ? k : m;
    int ldb = std::is_same_v<DataLayoutB, row_major> ? n : k;
    int ldc = std::is_same_v<DataLayoutC, row_major> ? n : m;
    int ldd = ldc;

    std::cout << "Initializing host data..." << std::endl;

    // Initialize input matrices
    std::vector<InputT>  matrixA(m * k);
    std::vector<InputT>  matrixB(k * n);
    std::vector<OutputT> matrixC(m * n);

    // Fill outputs with NaN to catch contamination
    std::vector<OutputT> matrixD(m * n, std::numeric_limits<OutputT>::signaling_NaN());

    fillRand(matrixA.data(), m, k);
    fillRand(matrixB.data(), k, n);
    fillRand(matrixC.data(), m, n);
    //fillEnc<DataLayoutA>(matrixA.data(), m, k);
    //printEnc<DataLayoutA>(matrixA.data(), m, k);
    //fillEnc<DataLayoutB>(matrixB.data(), k, n);
    //printEnc<DataLayoutB>(matrixB.data(), k, n);
    //fillEnc<DataLayoutC>(matrixC.data(), m, n);
    //printEnc<DataLayoutC>(matrixC.data(), m, n);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    InputT*  d_a;
    InputT*  d_b;
    OutputT* d_c;
    OutputT* d_d;

    const size_t bytesA = matrixA.size() * sizeof(InputT);
    const size_t bytesB = matrixB.size() * sizeof(InputT);
    const size_t bytesC = matrixC.size() * sizeof(OutputT);
    const size_t bytesD = matrixD.size() * sizeof(OutputT);

    CHECK_HIP_ERROR(hipMalloc(&d_a, bytesA));
    CHECK_HIP_ERROR(hipMalloc(&d_b, bytesB));
    CHECK_HIP_ERROR(hipMalloc(&d_c, bytesC));
    CHECK_HIP_ERROR(hipMalloc(&d_d, bytesD));

    CHECK_HIP_ERROR(hipMemcpy(d_a, matrixA.data(), bytesA, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, matrixB.data(), bytesB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, matrixC.data(), bytesC, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_d, matrixD.data(), bytesD, hipMemcpyHostToDevice));

    auto blockDim = dim3(hTBLOCK_X, hTBLOCK_Y);
    auto gridDim  = dim3(rocwmma::ceilDiv(m, get<0>(macroTileSize)),
                        rocwmma::ceilDiv(n, get<1>(macroTileSize)));

    std::cout << "Launching GEMM kernel..." << std::endl;
    std::cout << "gridDim (" << gridDim.x << " " << gridDim.y << ")"
              << " blockdim (" << blockDim.x << " " << blockDim.y << ")" << std::endl;

    // Uses 2 lds blocks for prefetch loop (A and B)
    int ldsusage
        = 2u * sizeof(InputT) * (get<0>(macroTileSize) + get<1>(macroTileSize)) * hROCWMMA_K;

    ////
    auto rocwmmaKernel = [&]() {
        hipExtLaunchKernelGGL(gemm_rocwmma_d,
                              gridDim,
                              blockDim,
                              ldsusage,
                              0,
                              nullptr,
                              nullptr,
                              0,
                              m,
                              n,
                              k,
                              d_a,
                              d_b,
                              d_c,
                              d_d,
                              lda,
                              ldb,
                              ldc,
                              ldd,
                              alpha,
                              beta);
    };

    constexpr uint32_t warmups    = 50u;
    constexpr uint32_t recordRuns = 5u;

    // Warm-up runs, not recorded
    for(uint32_t i = 0; i < warmups; ++i)
    {
        rocwmmaKernel();
    }

    // Actual recorded runs
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));
    for(uint32_t i = 0; i < recordRuns; ++i)
    {
        rocwmmaKernel();
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

    auto gFlops = calculateGFlops(m, n, k);
    auto tFlopsPerSec
        = calculateTFlopsPerSec(m, n, k, static_cast<double>(elapsedTimeMs), recordRuns);

    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    // Echo performance
    std::cout << "TBlockX, TBlockY, "
              << "BlocksX, BlocksY, "
              << "BlkM, BlkN, BlkK, "
              << "MatM, MatN, MatK, "
              << "alpha, lda, ldb, "
              << "beta, ldc, ldd, "
              << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

    std::cout << hTBLOCK_X << ", " << hTBLOCK_Y << ", " << hBLOCKS_X << ", " << hBLOCKS_Y << ", "
              << hROCWMMA_M << ", " << hROCWMMA_N << ", " << hROCWMMA_K << ", " << m << ", " << n
              << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
              << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", "
              << tFlopsPerSec << std::endl;

#if 1

    std::cout << "Validating result with reference..." << std::endl;

    if((uint64_t)m * (uint64_t)n * (uint64_t)k > (2048ull * 2048ull * 2048ull))
    {
        std::cout << "Please wait. Large sizes can take a while!" << std::endl;
    }

    // Bring kernel result back to host
    CHECK_HIP_ERROR(hipMemcpy(matrixD.data(), d_d, bytesD, hipMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<OutputT> matrixD_ref(m * n, std::numeric_limits<OutputT>::signaling_NaN());
    gemm_cpu_h<InputT, OutputT, ComputeT, DataLayoutA, DataLayoutB, DataLayoutC>(m,
                                                                                 n,
                                                                                 k,
                                                                                 matrixA.data(),
                                                                                 matrixB.data(),
                                                                                 matrixC.data(),
                                                                                 matrixD_ref.data(),
                                                                                 lda,
                                                                                 ldb,
                                                                                 ldc,
                                                                                 ldd,
                                                                                 alpha,
                                                                                 beta);

    auto res = compareEqual(matrixD.data(), matrixD_ref.data(), m * n);

    //std::cout << "Reference: \n";
    //printData<DataLayoutC>(matrixD_ref.data(), m, n);
    //std::cout << "Actual:\n";
    //printData<DataLayoutC>(matrixD.data(), m, n);

    if(std::get<0>(res) == false)
    {
        std::cout << "FAILED\n";
    }
    else
    {
        std::cout << "PASSED\n";
    }

    std::cout << "Max relative error: " << std::get<1>(res) << std::endl;

#endif // !NDEBUG

    // Release device memory
    CHECK_HIP_ERROR(hipFree(d_a));
    CHECK_HIP_ERROR(hipFree(d_b));
    CHECK_HIP_ERROR(hipFree(d_c));
    CHECK_HIP_ERROR(hipFree(d_d));

    std::cout << "Finished!" << std::endl;
}

int main()
{
    gemm_test(7168, 7168, 7168, 2, 2);
    //gemm_test(8192, 8192, 8192, 2, 2);
    //gemm_test(128, 128, 16, 2, 2);
    return 0;
}
