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
#ifndef ROCWMMA_LAYOUT_HPP
#define ROCWMMA_LAYOUT_HPP

#include "mapping_util.hpp"

namespace rocwmma
{
    // In relation to matrix space, DataLayouts describe whether consecutive elements in 1D data arrays are:
    // 1. Contiguous rows (row_major)
    // 2. Contiguous columns (col_major)
    namespace DataLayout
    {
        template <typename DataLayoutT>
        using Array1d = typename ::rocwmma::detail::template DataSpace<DataLayoutT>;

        using RowMajor = Array1d<row_major>;
        using ColMajor = Array1d<col_major>;

    } // namespace DataLayout

    // In 2D space, Matrix Layouts describe per-thread offset coordinates and iterative spaces
    // 1. Base thread offsets
    // 2. Stride offsets
    // 3. Stride spaces (counts)
    // 4. Per-iteration offsets (stride step based on iteration)
    // 5. Cumulative offsets (cumulative stride steps based on iteration)
    namespace MatrixLayout
    {
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColOrthoVW;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColInlineVW;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct RowOrthoVW;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct RowInlineVW;

        /////////////////// Interleaved patterns //////////////////
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth, //
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK = 1> // # of splits
        struct ColInlineInt;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth, //
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK = 1> // # of splits
        struct ColOrthoInt;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth, //
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK = 1> // # of splits
        struct RowInlineInt;

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth, //
                  uint32_t MfmaDim, // MFMA instruction size
                  uint32_t SplitK = 1> // # of splits
        struct RowOrthoInt;

        /////////////////// ////////////////////////////// //////////////////

    } // namespace MatrixLayout

    // Register layouts describe whether contiguous BlockDim elements are:
    // 1. Captured in the same register lane as if the input were in Array-Of-Structures (AOS)
    // 2. Captured across multiple register lanes as if the input were in Structure-Of-Arrays (SOA)
    namespace RegisterLayout
    {
        template <uint32_t BlockDim, uint32_t VW>
        struct Aos
        {
        };
        template <uint32_t BlockDim, uint32_t VW>
        struct Soa
        {
        };
    }

    // Layout profiles describe fragments in three mapped spaces:
    // 1. DataLayout:     data locality in memory space (row_major or col_major)
    // 2. MatrixLayout:   data locality in matrix space (ColOrthoVW, ColInlineVW, etc.)
    // 3. RegisterLayout: data locality in register space (AOS or SOA)
    namespace LayoutProfile
    {
        // ColNT is a layout profile that has the following properties:
        // 1. Leading dimension is aligned with column elements of fragment data:
        //    - BlockDim is assumed the column size, or BlockM dimension.
        //    - Analogous to capturing columns of 'matrix A'.
        // 2. Register elements are in MFMA friendly, or SOA register layout.
        // 3. Register layout does NOT change whether DataLayout is col_major or row_major (fast DataLayoutT change).
        // 4. MatrixLayout will capture contiguous column elements across multiple register lanes.
        // 5. VectorWidth is fixed to 1 in col_major to ensure #3 (non-optimal).
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct ColNT
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, 1, MaxVectorWidth>,
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>;
            using RegisterLayout = RegisterLayout::template Soa<BlockDim, MaxVectorWidth>;

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
        // 2. Register elements are in MFMA friendly, or SOA register layout.
        // 3. Register layout does NOT change whether DataLayout is col_major or row_major (fast DataLayoutT change).
        // 4. MatrixLayout will capture contiguous row elements across multiple register lanes.
        // 5. VectorWidth is fixed to 1 in row_major to ensure #3 (non-optimal).
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  typename DataLayoutT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct RowNT
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, 1, MaxVectorWidth>>;
            using RegisterLayout = RegisterLayout::template Soa<BlockDim, MaxVectorWidth>;

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
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth = VectorWidth>
        struct Col
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<
                is_same_v<DataLayoutT, col_major>,
                MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>,
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>;
            using RegisterLayout
                = conditional_t<is_same_v<DataLayoutT, col_major>,
                                RegisterLayout::template Aos<BlockDim, MaxVectorWidth>,
                                RegisterLayout::template Soa<BlockDim, MaxVectorWidth>>;

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
            using RegisterLayout
                = conditional_t<is_same_v<DataLayoutT, row_major>,
                                RegisterLayout::template Aos<BlockDim, MaxVectorWidth>,
                                RegisterLayout::template Soa<BlockDim, MaxVectorWidth>>;

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
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth = VectorWidth,
                  uint32_t MfmaDim        = 16u,
                  uint32_t SplitK         = 1u>
        struct ColInt
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<is_same_v<DataLayoutT, col_major>,
                                               MatrixLayout::ColInlineInt<BlockDim,
                                                                          BlockK,
                                                                          DataT,
                                                                          VectorWidth,
                                                                          MaxVectorWidth,
                                                                          MfmaDim,
                                                                          SplitK>,
                                               MatrixLayout::ColOrthoInt<BlockDim,
                                                                         BlockK,
                                                                         DataT,
                                                                         VectorWidth,
                                                                         MaxVectorWidth,
                                                                         MfmaDim,
                                                                         SplitK>>;
            using RegisterLayout
                = conditional_t<is_same_v<DataLayoutT, col_major>,
                                RegisterLayout::template Aos<BlockDim, MaxVectorWidth>,
                                RegisterLayout::template Soa<BlockDim, MaxVectorWidth>>;

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
                  uint32_t MaxVectorWidth = VectorWidth,
                  uint32_t MfmaDim        = 16u,
                  uint32_t SplitK         = 1u>
        struct RowInt
        {
            // Layouts
            using DataLayout   = DataLayout::template Array1d<DataLayoutT>;
            using MatrixLayout = conditional_t<is_same_v<DataLayoutT, row_major>,
                                               MatrixLayout::RowInlineInt<BlockDim,
                                                                          BlockK,
                                                                          DataT,
                                                                          VectorWidth,
                                                                          MaxVectorWidth,
                                                                          MfmaDim,
                                                                          SplitK>,
                                               MatrixLayout::RowOrthoInt<BlockDim,
                                                                         BlockK,
                                                                         DataT,
                                                                         VectorWidth,
                                                                         MaxVectorWidth,
                                                                         MfmaDim,
                                                                         SplitK>>;
            using RegisterLayout
                = conditional_t<is_same_v<DataLayoutT, row_major>,
                                RegisterLayout::template Aos<BlockDim, MaxVectorWidth>,
                                RegisterLayout::template Soa<BlockDim, MaxVectorWidth>>;

            // Mapping
            using MappingUtil  = MappingUtil<BlockDim, BlockK, DataT, DataLayoutT>;
            using MatrixCoordT = typename MappingUtil::MatrixCoordT;

            // Sanity checks
            // Must ensure that MaxVectorWidth fits inside the leading dimension
            static_assert(
                !(is_same_v<DataLayoutT, col_major> && (MaxVectorWidth > BlockK)),
                "MaxVectorWidth is larger than BlockK dimension. Try reducing MaxVectorWidth");
        };

    } // namespace FragmentLayout

    ///
    /// Helper to ensure layout types are consistent (same, or equivalent)
    ///
    template <typename LhsLayout, typename RhsLayout>
    struct ConsistencyCheck : public false_type
    {
    };

    ///
    /// Check for layout orthogonality
    ///
    template <typename LhsMatrixLayout, typename RhsMatrixLayout>
    struct OrthogonalCheck : public false_type
    {
    };

    template <typename LayoutT>
    struct OrthogonalLayout;

    template <typename Layout>
    using orthogonal_layout_t = typename OrthogonalLayout<Layout>::Type;

    template <typename LhsDataLayout, typename RhsDataLayout>
    struct is_orthogonal;

    template <typename LhsDataLayout, typename RhsDataLayout>
    inline constexpr bool is_orthogonal_v = is_orthogonal<LhsDataLayout, RhsDataLayout>::value;

} // namespace rocwmma

#include "layout_impl.hpp"

#endif // ROCWMMA_LAYOUT_HPP
