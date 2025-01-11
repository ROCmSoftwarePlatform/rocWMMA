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

#ifndef ROCWMMA_UTILITY_VECTOR_HPP
#define ROCWMMA_UTILITY_VECTOR_HPP

#include <rocwmma/internal/types.hpp>
#include <rocwmma/internal/utility/vector_impl.hpp>

namespace rocwmma
{
    //! Returns the element count, or size of the input vector
    //! @param v Input vector
    //! @tparam Input vector type
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_size(VecT&& v);

    //! Creates a vector with the given input values
    //! @param ts Variadic list of inputs values
    //! @tparam Types of incoming input values
    template <typename... Ts>
    ROCWMMA_HOST_DEVICE constexpr static inline auto make_vector(Ts&&... ts);

    /*! \class vector_generator
    *  \brief A flexible vector generator class that calls a functor to generate vector element.
    * Functor signature: F(Number<Idx>, args...), where Idx is the vector element number. The functor
    * may accept any number of arguments and generates a single value to be used as element Idx in the
    * result.
    *
    * @tparam Data type of the vector
    * @tparam The number of vector elements
    * @tparam BoundCtrl - OOB thread indices write 0 to output element
    */
    template <typename DataT, uint32_t VecSize>
    struct vector_generator : public detail::vector_generator<VecT, DataT, VecSize>
    {
    };

    //! Returns a concatenated vector of (lhs, rhs)
    //! @param lhs Input vector, lower elements
    //! @param rhs Input vector, upper elements
    //! @tparam Input vector type of lhs
    //! @tparam Input vector type of rhs
    template <typename Lhs, typename Rhs>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_cat(Lhs&& lhs, Rhs&& rhs);

    //! Returns the reduction result of bit-wise and between all elements in the input vector
    //! @param v Input vector
    //! @tparam Input vector type
    template <typename VecT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_reduce_and(VecT&& v) noexcept;

    //! Swaps elements in vector size of 2
    //! @param v Input vector
    //! @tparam Input vector type
    template <template<typename, uint32_t> class VecT,
    typename DataT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto swap(VecT<DataT, 2> const& v);

    //! Splits input vector into sub-vectors. Func is applied to each sub-vector, which are then
    //! concatenated and returned as a result. Does not change input vector v.
    //! @param v Input vector
    //! @param func Functor with signature Func(VecT<DataT, SubVecSize>, Number<I> idx, args...)
    //! @param args Arguments that are forwarded to the functor
    //! @tparam Sub-vector size (defaults to 1)
    //! @tparam Type of input vector
    //! @tparam Type of functor
    //! @tparam Type of forwarded arguments to functor
    template<uint32_t SubVecSize = 1u, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline auto vector_for_each(VecT&& v, Func&& func, ArgsT&&... args);

    //! Splits input vector into sub-vectors. Func is applied to each sub-vector in-place. Returns
    //! a reference to modified input vector.
    //! @param v Input vector
    //! @param func Functor with signature Func(VecT<DataT, SubVecSize>, Number<I> idx, args...)
    //! @param args Arguments that are forwarded to the functor
    //! @tparam Sub-vector size (defaults to 1)
    //! @tparam Type of input vector
    //! @tparam Type of functor
    //! @tparam Type of forwarded arguments to functor
    template<uint32_t SubVecSize = 1u, typename VecT, class Func, typename... ArgsT>
    ROCWMMA_HOST_DEVICE constexpr static inline decltype(auto) vector_mutate_for_each(VecT&& v, Func&& func,  ArgsT&&... args);

} // namespace rocwmma

#endif // ROCWMMA_UTILITY_VECTOR_HPP
