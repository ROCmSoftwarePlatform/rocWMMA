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

#ifndef ROCWMMA_UTILITY_VECTOR_IMPL_HPP
#define ROCWMMA_UTILITY_VECTOR_IMPL_HPP

#include <rocwmma/internal/utility/get.hpp>
#include <rocwmma/internal/utility/type_traits.hpp>

namespace rocwmma
{
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr inline auto vector_size(VecT const& v)
    {
        return VecTraits<VecT>::size();
    }

    namespace detail
    {
        template <typename... Ts>
        struct first_type;

        template <typename T, typename... Ts>
        struct first_type<T, Ts...>
        {
            using type = T;
        };

        template <typename... Ts>
        using first_type_t = typename first_type<Ts...>::type;

        template <typename... Ts>
        struct is_same_type;

        template <typename T>
        struct is_same_type<T> : true_type
        {
        };

        template <typename T, typename U, typename... Ts>
        struct is_same_type<T, U, Ts...>
            : conditional_t<is_same<T, U>{}, is_same_type<U, Ts...>, false_type>
        {
        };

        template <typename... Ts>
        constexpr bool is_same_type_v = is_same_type<Ts...>::value;

        template <uint32_t N>
        using Number = integral_constant<int32_t, N>;

        // Can be used to build any vector class of <DataT, VecSize>
        // Either VecT or non_native_vector_vase.
        // Class acts as a static for_each style generator:
        // Incoming functor F will be called with each index + args in sequence.
        // Results of functor calls are used to construct a new vector.
        template <template <typename, uint32_t> class VecT, typename DataT, uint32_t VecSize>
        struct vector_generator
        {
            static_assert(VecSize > 0, "VectorSize must be at least 1");

            ROCWMMA_HOST_DEVICE constexpr vector_generator() {}

            // F signature: F(Number<Iter>, args...)
            template <class F, typename... ArgsT>
            ROCWMMA_HOST_DEVICE constexpr auto operator()(F f, ArgsT&&... args) const
            {
                // Build the number sequence to be expanded below.
                return operator()(f, detail::Seq<VecSize>{}, forward<ArgsT>(args)...);
            }

        private:
            template <class F, uint32_t... Indices, typename... ArgsT>
            ROCWMMA_HOST_DEVICE constexpr auto
                operator()(F f, detail::SeqT<Indices...>, ArgsT&&... args) const
            {
                // Execute incoming functor f with each index, as well as forwarded args.
                // The resulting vector is constructed with the results of each functor call.
                return VecT<DataT, VecSize>{(f(Number<Indices>{}, forward<ArgsT>(args)...))...};
            }
        };
    }

    template <typename DataT>
    ROCWMMA_HOST_DEVICE constexpr inline auto swap(HIP_vector_type<DataT, 2> const& v)
    {
        return HIP_vector_type<DataT, 2>{get<1>(v), get<0>(v)};
    }

    template <typename... Ts>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) make_vector(Ts&&... ts)
    {
        // TODO: When HIP_vector_type becomes constexpr replace with non_native_vector type.

        // Ensure that all the arguments are the same type
        static_assert(detail::is_same_type_v<decay_t<Ts>...>,
                      "Vector arguments must all be the same type");

        using DataT = typename detail::first_type_t<decay_t<Ts>...>;
        return non_native_vector_base<DataT, sizeof...(Ts)>{forward<Ts>(ts)...};
    }

    namespace detail
    {
        template <typename DataT0,
                  uint32_t Rank0,
                  size_t... Is0,
                  typename DataT1,
                  uint32_t Rank1,
                  size_t... Is1>
        constexpr static inline decltype(auto)
            vector_cat_impl(non_native_vector_base<DataT0, Rank0> const& lhs,
                            index_sequence<Is0...>,
                            non_native_vector_base<DataT1, Rank1> const& rhs,
                            index_sequence<Is1...>)
        {
            return make_vector(get<Is0>(lhs)..., get<Is1>(rhs)...);
        }

    } // namespace detail

    template <typename Lhs, typename Rhs>
    ROCWMMA_HOST_DEVICE constexpr decltype(auto) vector_cat(Lhs&& lhs, Rhs&& rhs)
    {
        constexpr size_t Size0 = VecTraits<decay_t<decltype(lhs)>>::size();
        constexpr size_t Size1 = VecTraits<decay_t<decltype(rhs)>>::size();

        return detail::vector_cat_impl(forward<Lhs>(lhs),
                                       detail::make_index_sequence<Size0>(),
                                       forward<Rhs>(rhs),
                                       detail::make_index_sequence<Size1>());
    }

    namespace detail
    {
        template <typename DataT0, typename DataT1, uint32_t Rank, size_t... Is>
        constexpr static inline decltype(auto)
            mult_poly_vec_impl(non_native_vector_base<DataT0, Rank> const& lhs,
                               non_native_vector_base<DataT1, Rank> const& rhs,
                               index_sequence<Is...>)
        {
            return make_vector((get<Is>(lhs) * get<Is>(rhs))...);
        }

    } // namespace detail

    template <typename DataT0, typename DataT1, uint32_t Rank>
    constexpr decltype(auto) operator*(non_native_vector_base<DataT0, Rank> const& lhs,
                                       non_native_vector_base<DataT1, Rank> const& rhs)
    {
        return detail::mult_poly_vec_impl(lhs, rhs, detail::make_index_sequence<Rank>());
    }

    namespace detail
    {
        template <class BinOp, typename T, typename... Ts>
        ROCWMMA_HOST_DEVICE constexpr static inline decay_t<T> reduceOp_impl(T&& t,
                                                                             Ts&&... ts) noexcept
        {
            using CastT = decay_t<T>;
            if constexpr(sizeof...(Ts) >= 1)
            {
                return BinOp::exec(static_cast<CastT>(t), reduceOp_impl<BinOp>(forward<Ts>(ts)...));
            }
            else
            {
                return static_cast<CastT>(t);
            }
        }

        template <class BinOp, typename VecT, size_t... Is>
        ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
            vector_reduce_impl(VecT&& v, index_sequence<Is...>) noexcept
        {
            return reduceOp_impl<BinOp>(get<Is>(forward<VecT>(v))...);
        }

        // Use with operations that have 1 operands
        template <class BinOp, typename VecT>
        ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
            vector_reduce(VecT&& lhs) noexcept
        {
            return vector_reduce_impl<BinOp>(
                forward<VecT>(lhs),
                detail::make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
        }
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto)
        vector_reduce_and(VecT&& lhs) noexcept
    {
        return detail::vector_reduce<detail::BitwiseOp::And>(forward<VecT>(lhs));
    }
} // namespace rocwmma

#endif // ROCWMMA_UTILITY_VECTOR_IMPL_HPP
