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
#ifndef ROCWMMA_LAYOUT_TRAITS_HPP
#define ROCWMMA_LAYOUT_TRAITS_HPP

#include "utility/type_traits.hpp"

namespace rocwmma
{
    /*! \class is_layout_same
    *  \brief  Compares layout types are the same, or are equivalent. Similar to is_same,
    * however layouts can have an equivalency with small variations input parameters such that they
    * are still technically the same. This should be used when comparing any layout types:
    * DataLayout, MatrixLayout and RegisterLayout
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    struct is_layout_same : public false_type
    {
    };

    /*! \class is_layout_transpose
    *  \brief  Compares layout types if they are transposed with each other.
    *  @tparam LhsLayout Comparative left hand side
    *  @tparam RhsLayout Comparative right hand side
    */
    template <typename LhsLayout, typename RhsLayout>
    struct is_layout_transpose : public false_type
    {
    };

    /*! \class layout_transpose
    *  \brief  Transforms the layout type into its direct transpose.
    *  @tparam Layout the layout to transpose from
    */
    template <typename Layout>
    struct layout_transpose
    {
        // using type = ...
    };

    /*! \class layout_transpose_t
    *  \brief  Transforms the layout type into its direct transpose.
    *  @tparam Layout the layout to transpose from
    */
    template <typename Layout>
    using layout_transpose_t = typename layout_transpose<Layout>::type;

} // namespace rocwmma

#include "layout_traits_impl.hpp"

#endif // ROCWMMA_LAYOUT_TRAITS_HPP
