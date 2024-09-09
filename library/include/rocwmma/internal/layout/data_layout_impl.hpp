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
#ifndef ROCWMMA_DATA_LAYOUT_IMPL_HPP
#define ROCWMMA_DATA_LAYOUT_IMPL_HPP

#include "layout.hpp"
#include "layout_traits.hpp"
#include "utility/type_traits.hpp"

namespace rocwmma
{
    // Data layout trait tags are transposes
    template <>
    struct is_layout_transpose<row_major, col_major> : public true_type
    {
    };

    template <>
    struct is_layout_transpose<col_major, row_major> : public true_type
    {
    };

    // Data layout objects are transposes
    template <>
    struct is_layout_transpose<DataLayout::RowMajor, DataLayout::ColMajor> : public true_type
    {
    };

    template <>
    struct is_layout_transpose<DataLayout::ColMajor, DataLayout::RowMajor> : public true_type
    {
    };

    // Data layout trait tag transpose
    template <>
    struct layout_transpose<row_major>
    {
        using type = col_major;
    };

    template <>
    struct layout_transpose<col_major>
    {
        using type = row_major;
    };

    // Data layout object type transpose
    template <typename DataLayoutT>
    struct layout_transpose<DataLayout::template Array1d<DataLayoutT>>
    {
        using Type = DataLayout::template Array1d<layout_transpose_t<DataLayoutT>>;
    };

} // namespace rocwmma

#endif // ROCWMMA_DATA_LAYOUT_IMPL_HPP
