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
#ifndef ROCWMMA_DATA_LAYOUT_TRAITS_HPP
#define ROCWMMA_DATA_LAYOUT_TRAITS_HPP

#include "layout.hpp"
#include "layout_traits.hpp"

namespace rocwmma
{
    namespace LayoutTraits_impl
    {
        // Sameness classifier
        template <>
        struct is_layout_same<row_major, row_major, void> : public true_type
        {
        };

        template <>
        struct is_layout_same<col_major, col_major, void> : public true_type
        {
        };

        template <>
        struct is_layout_same<DataLayout::RowMajor, DataLayout::RowMajor, void> : public true_type
        {
        };

        template <>
        struct is_layout_same<DataLayout::ColMajor, DataLayout::ColMajor, void> : public true_type
        {
        };

        // Orthogonality classifier
        template <>
        struct is_layout_orthogonal<row_major, col_major, void> : public true_type
        {
        };

        template <>
        struct is_layout_orthogonal<col_major, row_major, void> : public true_type
        {
        };

        template <>
        struct is_layout_orthogonal<DataLayout::RowMajor, DataLayout::ColMajor, void>
            : public true_type
        {
        };

        template <>
        struct is_layout_orthogonal<DataLayout::ColMajor, DataLayout::RowMajor, void>
            : public true_type
        {
        };

        // Orthogonal layout guides
        template <>
        struct orthogonal_layout<row_major>
        {
            using type = col_major;
        };

        template <>
        struct orthogonal_layout<col_major>
        {
            using type = row_major;
        };

        template <typename DataLayoutT>
        struct orthogonal_layout<DataLayout::template Array1d<DataLayoutT>>
        {
            using Type
                = DataLayout::template Array1d<typename orthogonal_layout<DataLayoutT>::type>;
        };

    } // namespace LayoutTraits_impl

} // namespace rocwmma

#endif // ROCWMMA_DATA_LAYOUT_TRAITS_HPP
