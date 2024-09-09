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
#ifndef ROCWMMA_TRANSFORMS_HPP
#define ROCWMMA_TRANSFORMS_HPP

#include "transforms_impl.hpp"
#include "vector.hpp"

namespace rocwmma
{
    namespace Transforms
    {
        template <class Op>
        struct Driver
        {
            using Func = Op;

            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                auto result = VecT<DataT, VecSize>{};

                auto rIt = makeVectorIterator<Func::VecSize>(v).begin();
                auto wIt = makeVectorIterator<Func::VecSize>(result).begin();

#pragma unroll
                for(uint32_t i = 0; i < decltype(rIt)::range(); i++)
                {
                    *wIt = Func::exec(*rIt);
                    rIt++;
                    wIt++;
                }
                return result;
            }
        };

        template <uint32_t BlockDim, uint32_t MaxVW>
        using AosToSoa = Driver<TransformsImpl::Ops::AosToSoa<BlockDim, MaxVW>>;

        template <uint32_t BlockDim, uint32_t MaxVW>
        using SoaToAos = Driver<TransformsImpl::Ops::SoaToAos<BlockDim, MaxVW>>;

        // Note: If you arrive at an undefined register_transform error, it is likely
        // the layout transformation is not currently supported. Need to either implement
        // the transform or ensure your layout transform mapping is correct.
        template <typename SrcLayout,
                  typename DstLayout,
                  typename Match = is_same<SrcLayout, DstLayout>>
        struct register_transform;

        // Layouts that are identical do not require register transformations
        template <typename SrcLayout, typename DstLayout>
        struct register_transform<SrcLayout, DstLayout, true_type>
        {
            template <typename DataT, uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline decltype(auto)
                exec(VecT<DataT, VecSize> const& v)
            {
                return v;
            }
        };

        /////// To MmaInput ///////

        // ColInlineVW and RowInlineVW layouts are not mma friendly and require Aos->Soa transform.
        // Only valid for BlockDims that supported by mma
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct register_transform<
            RegisterLayout::Storage<
                MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>,
            RegisterLayout::MmaInput<BlockDim>,
            false_type>
        {
            // TODO: Remove DataT from the transform
            template <uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                static_assert(RegisterLayout::detail::testSupportedMmaDim(BlockDim),
                              "Unsupported mma dim");

                // ColInlineVW -> ColOrthoVW (mma friendly) = AOS -> SOA transform
                return AosToSoa<BlockDim, MaxVectorWidth>::exec(v);
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidth,
                  uint32_t MaxVectorWidth>
        struct register_transform<
            RegisterLayout::Storage<
                MatrixLayout::RowInlineVW<BlockDim, BlockK, DataT, VectorWidth, MaxVectorWidth>>,
            RegisterLayout::MmaInput<BlockDim>,
            false_type>
        {
            // TODO: Remove DataT from the transform
            template <uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                static_assert(RegisterLayout::detail::testSupportedMmaDim(BlockDim),
                              "Unsupported mma dim");

                // RowInlineVW -> RowOrthoVW (mma friendly) = AOS -> SOA transform
                return AosToSoa<BlockDim, MaxVectorWidth>::exec(v);
            }
        };

        /////// To Other Layouts ///////

        // In-register layouts for (RowInlineVW and RowOrthoVW), and (ColInlineVW and ColOrthoVW) are orthgonal
        // and need specific transforms to transition between either representation.
        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidthLhs,
                  uint32_t VectorWidthRhs,
                  uint32_t MaxVectorWidth>
        struct register_transform<
            RegisterLayout::Storage<
                MatrixLayout::RowInlineVW<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth>>,
            RegisterLayout::Storage<
                MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth>>,
            false_type>
        {
            // TODO: Remove DataT from the transform
            template <uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                static_assert(RegisterLayout::detail::testSupportedVW(
                                  MaxVectorWidth, VectorWidthLhs, VectorWidthRhs),
                              "Invalid VW");

                // RowInlineVW -> RowOrthoVW = AOS -> SOA transform
                return AosToSoa<BlockDim, MaxVectorWidth>::exec(v);
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidthLhs,
                  uint32_t VectorWidthRhs,
                  uint32_t MaxVectorWidth>
        struct register_transform<
            RegisterLayout::Storage<
                MatrixLayout::RowOrthoVW<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth>>,
            RegisterLayout::Storage<
                MatrixLayout::RowInlineVW<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth>>,
            false_type>
        {
            // TODO: Remove DataT from the transform
            template <uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                static_assert(RegisterLayout::detail::testSupportedVW(
                                  MaxVectorWidth, VectorWidthLhs, VectorWidthRhs),
                              "Invalid VW");

                // RowOrthoVW -> RowInlineVW = SOA -> AOS transform
                return SoaToAos<BlockDim, MaxVectorWidth>::exec(v);
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidthLhs,
                  uint32_t VectorWidthRhs,
                  uint32_t MaxVectorWidth>
        struct register_transform<
            RegisterLayout::Storage<
                MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth>>,
            RegisterLayout::Storage<
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth>>,
            false_type>
        {
            // TODO: Remove DataT from the transform
            template <uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                static_assert(RegisterLayout::detail::testSupportedVW(
                                  MaxVectorWidth, VectorWidthLhs, VectorWidthRhs),
                              "Invalid VW");

                // ColInlineVW -> ColOrthoVW = AOS -> SOA transform
                return AosToSoa<BlockDim, MaxVectorWidth>::exec(v);
            }
        };

        template <uint32_t BlockDim,
                  uint32_t BlockK,
                  typename DataT,
                  uint32_t VectorWidthLhs,
                  uint32_t VectorWidthRhs,
                  uint32_t MaxVectorWidth>
        struct register_transform<
            RegisterLayout::Storage<
                MatrixLayout::ColOrthoVW<BlockDim, BlockK, DataT, VectorWidthRhs, MaxVectorWidth>>,
            RegisterLayout::Storage<
                MatrixLayout::ColInlineVW<BlockDim, BlockK, DataT, VectorWidthLhs, MaxVectorWidth>>,
            false_type>
        {
            // TODO: Remove DataT from the transform
            template <uint32_t VecSize>
            ROCWMMA_DEVICE constexpr static inline auto exec(VecT<DataT, VecSize> const& v)
            {
                static_assert(0, "Nope");
                static_assert(RegisterLayout::detail::testSupportedVW(
                                  MaxVectorWidth, VectorWidthLhs, VectorWidthRhs),
                              "Invalid VW");

                // ColOrthoVW -> ColInlineVW = SOA -> AOS transform
                return SoaToAos<BlockDim, MaxVectorWidth>::exec(v);
            }
        };

    } // namespace Transforms

} // namespace rocwmma

#endif // ROCWMMA_TRANSFORMS_HPP
