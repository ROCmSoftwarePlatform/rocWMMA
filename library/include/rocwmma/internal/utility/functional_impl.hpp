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

#ifndef ROCWMMA_UTILITY_FUNCTIONAL_IMPL_HPP
#define ROCWMMA_UTILITY_FUNCTIONAL_IMPL_HPP

#include "type_traits.hpp"

namespace rocwmma
{
    namespace detail
    {
        // Logical ops
        template <typename... Bs>
        struct logical_or;

        template <>
        struct logical_or<> : public false_type
        {
        };

        template <typename T>
        struct logical_or<T> : public T
        {
        };

        template <typename B1, typename B2>
        struct logical_or<B1, B2> : public conditional_t<B1::value, B1, B2>
        {
        };

        template <typename B1, typename B2, typename B3, typename... Bs>
        struct logical_or<B1, B2, B3, Bs...>
            : public conditional_t<B1::value, B1, logical_or<B2, B3, Bs...>>
        {
        };

        template <typename... Bs>
        using logical_or_t = typename logical_or<Bs...>::type;

        template <typename...>
        struct logical_and;

        template <>
        struct logical_and<> : public true_type
        {
        };

        template <typename B1>
        struct logical_and<B1> : public B1
        {
        };

        template <typename B1, typename B2>
        struct logical_and<B1, B2> : public conditional_t<B1::value, B2, B1>
        {
        };

        template <typename B1, typename B2, typename B3, typename... Bs>
        struct logical_and<B1, B2, B3, Bs...>
            : public conditional_t<B1::value, logical_and<B2, B3, Bs...>, B1>
        {
        };

        template <typename... Bs>
        using logical_and_t = typename logical_and<Bs...>::type;

        template <typename B>
        struct logical_not : public bool_constant<!bool(B::value)>
        {
        };

        template <typename B>
        using logical_not_t = typename logical_not<B>::type;

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_FUNCTIONAL_IMPL_HPP
