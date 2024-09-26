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
#ifndef ROCWMMA_LAYOUT_PROFILE_HPP
#define ROCWMMA_LAYOUT_PROFILE_HPP

#include "layout.hpp"

namespace rocwmma
{
    // Layout profiles are high-level objects that describe fragments in three mapped spaces:
    // 1. DataLayout:     data locality in 1D memory space (row_major or col_major)
    // 2. MatrixLayout:   data locality in 2D matrix space (ColOrthoVW, ColInlineVW, etc.)
    // 3. RegisterLayout: data locality in register space (Storage, or MmaInput)
    namespace LayoutProfile
    {
        // ColNT is a layout profile that has the following properties:
        // 1. Leading dimension is aligned with column elements of fragment data:
        //    - BlockDim is assumed the column size, or BlockM dimension.
        //    - Analogous to capturing columns of 'matrix A'.
        // 2. When BlockDim is supported by mma, register elements are always in MmaInput friendly register layout.
        // 3. Register layout does NOT change whether DataLayout is col_major or row_major (free DataLayoutT change).
        // 4. MatrixLayout will capture contiguous column elements across contiguous threads.
        // 5. VectorWidth is fixed to 1 in col_major to ensure #4 (non-optimal).
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColNT
        {
            // Layouts
            using DataLayout = DataLayout::template Array1d<DataLayoutT>;

            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, 1, MaxVectorWidth>,
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, MaxVectorWidth, MaxVectorWidth>>;

            using RegisterLayout = RegisterLayout::Storage<MatrixLayout>;

            // Mapping
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayoutT>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // Sanity checks
            // ColNT enforces consistent in-register alignment of contiguous matrix column
            // elements in both row_major or col_major data layouts.
            // This layout cannot support for VW > 1 in col_major data layout otherwise the
            // ordering is broken.
            static_assert(!(is_same_v<DataLayoutT, col_major> && VectorWidth > 1),
                          "ColNT in col_major does not support VectorWidth > 1");

            // Must ensure that MaxVectorWidth fits inside the leading dimension
            static_assert(
                !(is_same_v<DataLayoutT, row_major> && (MaxVectorWidth > BlockK)),
                "MaxVectorWidth is larger than BlockK dimension. Try reducing MaxVectorWidth");
        };

        // RowNT is a layout profile that has the following properties:
        // 1. Leading dimension is aligned with row elements of fragment data:
        //    - BlockDim is assumed the row size, or BlockN dimension.
        //    - Analogous to capturing rows of 'matrix B' or 'accumulator'.
        // 2. When BlockDim is supported by mma, register elements are always MmaInput friendly register layout.
        // 3. Register layout does NOT change whether DataLayout is col_major or row_major (fast DataLayoutT change).
        // 4. MatrixLayout will capture contiguous row elements across contiguous threads.
        // 5. VectorWidth is fixed to 1 in row_major to ensure #4 (non-optimal).
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct RowNT
        {
            // Layouts
            using DataLayout = DataLayout::template Array1d<DataLayoutT>;

            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, 1, MaxVectorWidth>>;

            using RegisterLayout = RegisterLayout::Storage<MatrixLayout>;

            // Mapping
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayoutT>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // Sanity checks
            // RowNT enforces consistent in-register alignment of contiguous matrix row
            // elements in both in row_major or col_major data layouts.
            // This layout cannot support for VW > 1 in row_major data layout.
            static_assert(!(is_same_v<DataLayoutT, row_major> && VectorWidth > 1),
                          "RowNT in row_major does not support VectorWidth > 1");

            // Must ensure that MaxVectorWidth fits inside the leading dimension
            static_assert(
                !(is_same_v<DataLayoutT, col_major> && (MaxVectorWidth > BlockK)),
                "MaxVectorWidth is larger than BlockK dimension. Try reducing MaxVectorWidth");
        };

        // Col is a layout profile that has the following properties:
        // 1. Leading dimension is aligned with column elements of fragment data:
        //    - BlockDim is assumed the column size, or BlockM dimension.
        //    - Analogous to capturing columns of 'matrix A'.
        // 2. Register layout is dynamic:
        //    - col_major data is stored in AOS register layout (non-MmaInput friendly), and
        //    - row_major data is stored in SOA register layout (MmaInput friendly).
        // 3. Register layout DOES change whether DataLayout is col_major or row_major (cost for DataLayoutT change).
        // 4. VectorWidth is NOT fixed to 1 in either data layout (optimal).
        // 5. Must convert to SOA if using with MFMA.
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth = VectorWidth>
        struct Col
        {
            // Layouts
            using DataLayout = DataLayout::template Array1d<DataLayoutT>;

            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>;

            using RegisterLayout = RegisterLayout::Storage<MatrixLayout>;

            // Mapping
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayoutT>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // Sanity checks
            // Must ensure that MaxVectorWidth fits inside the leading dimension
            static_assert(
                !(is_same_v<DataLayoutT, row_major> && (MaxVectorWidth > BlockK)),
                "MaxVectorWidth is larger than BlockK dimension. Try reducing MaxVectorWidth");
        };

        // Row is a layout profile that has the following properties:
        // 1. Leading dimension is aligned with row elements of fragment data:
        //    - BlockDim is assumed the row size, or BlockN dimension.
        //    - Analogous to capturing rows of 'matrix B' or 'accumulator'.
        // 2. Register layout is dynamic:
        //    - row_major data is stored in AOS register layout (non-MFMA friendly), and
        //    - col_major data is stored in SOA register layout (MFMA friendly).
        //    - Both data layouts cover the same geometric elements (transform friendly).
        // 3. Register layout DOES change whether DataLayout is col_major or row_major (cost for DataLayoutT change).
        // 4. VectorWidth is NOT fixed to 1 in either data layout (optimal).
        // 5. User must convert to SOA if using with MFMA.
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth = VectorWidth>
        struct Row
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, row_major>,
                MatrixLayout::RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>;

            using RegisterLayout = RegisterLayout::Storage<MatrixLayout>;

            // Mapping
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayoutT>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // Sanity checks
            // Must ensure that MaxVectorWidth fits inside the leading dimension
            static_assert(
                !(is_same_v<DataLayoutT, col_major> && (MaxVectorWidth > BlockK)),
                "MaxVectorWidth is larger than BlockK dimension. Try reducing MaxVectorWidth");
        };

        //////////////// Interleaved layouts /////////////

        // Col is a layout profile that has the following properties:
        // 1. Leading dimension is aligned with column elements of fragment data:
        //    - BlockDim is assumed the column size, or BlockM dimension.
        //    - Analogous to capturing columns of 'matrix A'.
        // 2. Register layout is dynamic:
        //    - col_major data is stored in AOS register layout (non-MFMA friendly), and
        //    - row_major data is stored in SOA register layout (MFMA friendly).
        //    - Both data layouts cover the same geometric elements (transform friendly).
        // 3. Register layout DOES change whether DataLayout is col_major or row_major (cost for DataLayoutT change).
        // 4. VectorWidth is NOT fixed to 1 in either data layout (optimal).
        // 5. User must convert to SOA if using with MFMA.
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t MfmaDim = 16u,
                  uint32_t SplitK  = 1u>
        struct ColInt
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::ColInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>,
                MatrixLayout::ColOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>;

            using RegisterLayout = RegisterLayout::Storage<MatrixLayout>;

            // Mapping
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayoutT>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // Sanity checks
            // Must ensure that MaxVectorWidth fits inside the leading dimension
            // TODO: fix
            // static_assert(
            //     !(is_same_v<DataLayoutT, row_major> && (MaxVectorWidth > BlockK)),
            //     "MaxVectorWidth is larger than BlockK dimension. Try reducing MaxVectorWidth");
        };

        // Row is a layout profile that has the following properties:
        // 1. Leading dimension is aligned with row elements of fragment data:
        //    - BlockDim is assumed the row size, or BlockN dimension.
        //    - Analogous to capturing rows of 'matrix B' or 'accumulator'.
        // 2. Register layout is dynamic:
        //    - row_major data is stored in AOS register layout (non-MFMA friendly), and
        //    - col_major data is stored in SOA register layout (MFMA friendly).
        //    - Both data layouts cover the same geometric elements (transform friendly).
        // 3. Register layout DOES change whether DataLayout is col_major or row_major (cost for DataLayoutT change).
        // 4. VectorWidth is NOT fixed to 1 in either data layout (optimal).
        // 5. User must convert to SOA if using with MFMA.
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t MfmaDim = 16u,
                  uint32_t SplitK  = 1u>
        struct RowInt
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, row_major>,
                MatrixLayout::RowInlineInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>,
                MatrixLayout::RowOrthoInt<BlockDim, BlockK, DataT, MfmaDim, SplitK>>;

            using RegisterLayout = RegisterLayout::Storage<MatrixLayout>;

            // Mapping
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayoutT>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // Sanity checks
            // Must ensure that MaxVectorWidth fits inside the leading dimension
            // TODO: fix
            // static_assert(
            //     !(is_same_v<DataLayoutT, col_major> && (MaxVectorWidth > BlockK)),
            //     "MaxVectorWidth is larger than BlockK dimension. Try reducing MaxVectorWidth");
        };

    } // namespace LayoutProfile

} // namespace rocwmma

#endif // ROCWMMA_LAYOUT_PROFILE_HPP
