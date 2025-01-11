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
#include <rocwmma/internal/vector_iterator.hpp>

namespace rocwmma
{
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_size(VecT&& v)
    {
        return VecTraits<decay_t<VecT>>::size();
    }

    namespace detail
    {
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
            ROCWMMA_HOST_DEVICE constexpr inline auto operator()(F f, ArgsT&&... args) const
            {
                // Build the number sequence to be expanded below.
                return operator()(f, detail::Seq<VecSize>{}, forward<ArgsT>(args)...);
            }

        private:
            template <class F, uint32_t... Indices, typename... ArgsT>
            ROCWMMA_HOST_DEVICE constexpr inline auto
                operator()(F f, detail::SeqT<Indices...>, ArgsT&&... args) const
            {
                // Execute incoming functor f with each index, as well as forwarded args.
                // The resulting vector is constructed with the results of each functor call.
                return VecT<DataT, VecSize>{(f(I<Indices>{}, forward<ArgsT>(args)...))...};
            }
        };
    }

    template <template<typename, uint32_t> class VecT, typename DataT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto swap(VecT<DataT, 2> const& v)
    {
        return VecT<DataT, 2>{get<1>(v), get<0>(v)};
    }

    namespace detail
    {
        template <template<typename, uint32_t> class VecT, typename... Ts>
        ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector_impl(Ts&&... ts)
        {
            // TODO: When HIP_vector_type becomes constexpr replace with non_native_vector type.

            // Ensure that all the arguments are the same type
            static_assert(detail::is_same_v<decay_t<Ts>...>,
                        "Vector arguments must all be the same type");

            using DataT = first_type_t<decay_t<Ts>...>;
            return VecT<DataT, sizeof...(Ts)>{forward<Ts>(ts)...};
        }

    } // namespace detail

    template <typename... Ts>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector(Ts&&... ts)
    {
        // TODO: When HIP_vector_type becomes constexpr replace
        // non_native_vector_base to VecT.
        return detail::make_vector_impl<non_native_vector_base>(forward<Ts>(ts)...);
    }

    namespace detail
    {
        template <typename Lhs,
                  size_t... Is0,
                  typename Rhs,
                  size_t... Is1>
        constexpr static inline auto
            vector_cat_impl(Lhs&& lhs,
                            index_sequence<Is0...>,
                            Rhs&& rhs,
                            index_sequence<Is1...>)
        {
            return make_vector(get<Is0>(forward<Lhs>(lhs))...,
                               get<Is1>(forward<Rhs>(rhs))...);
        }

    } // namespace detail

    template <typename Lhs, typename Rhs>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_cat(Lhs&& lhs, Rhs&& rhs)
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
        constexpr static inline auto
            mult_poly_vec_impl(non_native_vector_base<DataT0, Rank> const& lhs,
                               non_native_vector_base<DataT1, Rank> const& rhs,
                               index_sequence<Is...>)
        {
            return make_vector((get<Is>(lhs) * get<Is>(rhs))...);
        }

    } // namespace detail

    template <typename DataT0, typename DataT1, uint32_t Rank>
    constexpr static inline auto operator*(non_native_vector_base<DataT0, Rank> const& lhs,
                                       non_native_vector_base<DataT1, Rank> const& rhs)
    {
        return detail::mult_poly_vec_impl(lhs, rhs, detail::make_index_sequence<Rank>());
    }

    namespace detail
    {
        template <class BinOp, typename T, typename... Ts>
        ROCWMMA_HOST_DEVICE constexpr static inline auto reduceOp_impl(T&& t,
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
        ROCWMMA_HOST_DEVICE constexpr static inline auto
            vector_reduce_impl(VecT&& v, index_sequence<Is...>) noexcept
        {
            return reduceOp_impl<BinOp>(get<Is>(forward<VecT>(v))...);
        }

        // Use with operations that have 1 operands
        template <class BinOp, typename VecT>
        ROCWMMA_HOST_DEVICE constexpr static inline auto
            vector_reduce(VecT&& lhs) noexcept
        {
            return vector_reduce_impl<BinOp>(
                forward<VecT>(lhs),
                detail::make_index_sequence<VecTraits<decay_t<VecT>>::size()>{});
        }
    }

    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce_and(VecT&& lhs) noexcept
    {
        return detail::vector_reduce<detail::BitwiseOp::And>(forward<VecT>(lhs));
    }

    namespace detail
    {
        template<typename WriteIt, typename ReadIt, typename IndexT, typename Func, typename... ArgsT>
        ROCWMMA_HOST_DEVICE constexpr static inline auto vector_for_each_impl(WriteIt&& writeIt, ReadIt&& readIt, IndexT&& idx, Func&& func, ArgsT&&... args)
        {
            using WriteItT = decay_t<WriteIt>;
            using ReadItT = decay_t<ReadIt>;
            using NumberT = decay_t<IndexT>;

            static_assert(WriteItT::range() == ReadItT::range(), "Mismatch in iterator range");
            static_assert(WriteItT::range() > NumberT::value, "Invalid index");
            static_assert(ReadItT::range() > NumberT::value, "Invalid index");

            // De-reference read iterator and feed sub-vector to function.
            // Feed constexpr index value to function argument.
            // Write back sub-vector to writing iterator with function result
            *writeIt = func(*readIt, NumberT::value, args...);
        }

        // Internal variant that will mutate the original input vector in place
        template<typename VecT, class Func, uint32_t... Indices, typename... ArgsT>
        ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto) vector_mutate_for_each_impl(VecT&& v, Func&& func, detail::SeqT<Indices...>, ArgsT&&... args)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;
            constexpr uint32_t SubVecSize = VecTraits::size() / sizeof...(Indices);

            // Fold over each subvector
            // Iterators attach directly to input for read / write mutation
            (
                vector_for_each_impl(makeVectorIterator<SubVecSize>(forward<VecT>(v)).it(Indices),
                                        makeVectorIterator<SubVecSize>(forward<VecT>(v)).it(Indices),
                                        I<Indices>{},
                                        forward<Func>(func),
                                        forward<ArgsT>(args)...),
                ...
            );
            return v;
        }

        // Internal variant that will not mutate the original input vector and produce a separate result
        template<class Func, typename VecT, uint32_t... Indices, typename... ArgsT>
        ROCWMMA_HOST_DEVICE constexpr static inline auto vector_for_each_impl(VecT&& v, Func&& func, detail::SeqT<Indices...>, ArgsT&&... args)
        {
            using VecTraits = VecTraits<decay_t<VecT>>;
            constexpr uint32_t SubVecSize = VecTraits::size() / sizeof...(Indices);

            // Initialize a result
            auto result = decay_t<VecT>{};

            // Fold over each subvector
            // Write iterators attach to new result, not mutating input
            (
                vector_for_each_impl(makeVectorIterator<SubVecSize>(result).it(Indices),
                                        makeVectorIterator<SubVecSize>(forward<VecT>(v)).it(Indices),
                                        I<Indices>{},
                                        forward<Func>(func),
                                        forward<ArgsT>(args)...),
                ...
            );
            return result;
        }
    } // namespace detail

    // Func signature: Func(VecT<DataT, SubVecSize>, uint32_t idx, args...)
    template<uint32_t SubVecSize /*= 1u*/, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_for_each(VecT&& v, Func&& func, ArgsT&&... args)
    {
        using VecTraits = VecTraits<decay_t<VecT>>;
        constexpr uint32_t VecSize = VecTraits::size();

        // Sanity checks
        static_assert(VecSize >= SubVecSize, "SubVecSize exceeds VecSize");
        static_assert(VecSize % SubVecSize == 0u, "VecSize must be a multiple of SubVecSize");

        // Feed-fwd with index sequence
        return detail::vector_for_each_impl(forward<VecT>(v),
                                        forward<Func>(func),
                                        detail::Seq<VecSize / SubVecSize>{},
                                        forward<ArgsT>(args)...);
    }

    template<uint32_t SubVecSize /*= 1u*/, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto) vector_mutate_for_each(VecT&& v, Func&& func,  ArgsT&&... args)
    {
        using VecTraits = VecTraits<decay_t<VecT>>;
        constexpr uint32_t VecSize = VecTraits::size();

        // Sanity checks
        static_assert(VecSize >= SubVecSize, "SubVecSize exceeds VecSize");
        static_assert(VecSize % SubVecSize == 0u, "VecSize must be a multiple of SubVecSize");

        // Feed-fwd with index sequence
        return detail::vector_mutate_for_each_impl(forward<VecT>(v),
                                               forward<Func>(func),
                                               detail::Seq<VecSize / SubVecSize>{},
                                               forward<ArgsT>(args)...);
    }

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_VECTOR_IMPL_HPP
