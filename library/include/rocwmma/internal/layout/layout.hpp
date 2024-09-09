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
    // DataLayout objects map 2D matrix coordinate space to 1D data arrays offsets.
    // DataLayoutT tags describe whether consecutive elements are:
    // 1. Contiguous rows (row_major)
    // 2. Contiguous columns (col_major)
    namespace DataLayout
    {
        /*! \class Array1d
        *  \brief  A class to help map from 2D matrix space to 1D data space.
        *  @tparam DataLayoutT Meta-tag indicating whether data is stored in
        * row_major or col_major order.
        */
        template <typename DataLayoutT>
        using Array1d = typename ::rocwmma::detail::template DataSpace<DataLayoutT>;

        /*! \class RowMajor
        *  \brief  Maps 2D matrix space to row_major 1D data space
        */
        using RowMajor = Array1d<row_major>;

        /*! \class ColMajor
        *  \brief  Maps 2D matrix space to col_major 1D data space
        */
        using ColMajor = Array1d<col_major>;

    } // namespace DataLayout

    // Matrix Layouts map thread offsets into 2D matrix coordinate space:
    // 1. Base thread offsets
    // 2. Stride offsets
    // 3. Stride counts
    // 4. Per-iteration offsets (stride step based on iteration)
    // 5. Cumulative offsets (cumulative stride steps based on iteration)
    namespace MatrixLayout
    {
        /*! \class ColOrthoVW
        *  \brief  A matrix layout that maps contiguous threads to contiguous column elements, in the BlockDim direction.
        * VectorWidth elements are mapped orthogonal to the column, in the BlockK Direction.
        *  @tparam BlockDim The height of the column
        *  @tparam BlockK The number of columns
        *  @tparam DataT The datatype
        *  @tparam VectorWidth The iterative vector width
        *  @tparam MaxVectorWidth The total vector width
        */
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

    // Register layouts describe in-register layout and serve as transform states, or endpoints.
    // These are mnemonics which provide:
    // 1. A relationship between in-register layouts and combinations of matrix / data layouts.
    // 2. Useful parameters that may be used in transformations between endpoints.
    // 3. With indications from layout traits, can determine likeness or orthogonality between states.
    // Note: For these mnemonics to be useful, there must exist a transformable path between layouts.
    // Example:
    // Suppose we associate associate fragment register data with Storage<MatrixLayout::RowInline> upon loading.
    // To use the fragment register data with mma functions, we may attempt to transform the data from
    // Storage<MatrixLayout::RowInline> to MmaInput<16> to serve as input to a 16x16xk mma builtin.
    namespace RegisterLayout
    {
        // A mnemonic used to describe the register layout is suitable for input/output
        template <class MatrixLayout>
        struct Storage
        {
        };

        // A mnemonic used to describe the register layout is suitable for mma input for A/B
        template <uint32_t MmaSize>
        struct MmaInput
        {
        };

        // A mnemonic used to describe the register layout is suitable for mma input for accumulator input/output
        template <uint32_t MmaSize>
        struct MmaAcc
        {
        };

    } // namespace RegisterLayout

} // namespace rocwmma

#include "data_layout_impl.hpp"
#include "matrix_layout_impl.hpp"
#include "matrix_layout_interleaved_impl.hpp"
#include "register_layout_impl.hpp"

#endif // ROCWMMA_LAYOUT_HPP
