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
#ifndef ROCWMMA_OPAQUE_STORE_HPP
#define ROCWMMA_OPAQUE_STORE_HPP

#include "io_traits.hpp"
#include "types.hpp"
#include "vector_iterator.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename DataT, uint32_t VectorWidth>
        struct amdgcn_opaque_store
        {
            static_assert(VectorWidth > 0, "Vector width must be greater than 0");
            static_assert(sizeof(DataT[VectorWidth]) == sizeof(VecT<DataT, VectorWidth>),
                          "Cannot vectorize output");

            using BufferT = VecT<DataT, VectorWidth>;

            ROCWMMA_HOST_DEVICE constexpr static inline uint32_t size()
            {
                return VectorWidth;
            }

            ROCWMMA_DEVICE static inline void
                exec(DataT* dataPtr, BufferT const& data, index_t offset = 0)
            {
                *(BufferT*)(&(dataPtr[offset])) = data;
            }
        };

        // Wrapper to adapt to the IOBearer interface (bufferT, PtrT)
        template <typename DataT, uint32_t VectorWidth>
        struct OpaqueStoreBearer : public detail::amdgcn_opaque_store<DataT, VectorWidth>
        {
        private:
            using Base = detail::amdgcn_opaque_store<DataT, VectorWidth>;

        public:
            template <typename BufferT, typename DataPtrT>
            ROCWMMA_DEVICE static inline void
                exec(BufferT&& data, DataPtrT&& dataPtr, index_t offset = 0)
            {
                Base::exec(forward<DataPtrT>(dataPtr), forward<BufferT>(data), offset);
            }
        };

    } // namespace detail

    template <class DataLayout, class MatrixLayout>
    struct OpaqueStore : public IOBearer<DataLayout, MatrixLayout, detail::OpaqueStoreBearer>
    {
    private:
        using Base = IOBearer<DataLayout, MatrixLayout, detail::OpaqueStoreBearer>;

    public:
        template <typename DataPtrT, typename BufferT>
        ROCWMMA_DEVICE static void exec(DataPtrT&& dataPtr, BufferT&& buff, uint32_t ldm)
        {
            Base::exec(forward<BufferT>(buff), forward<DataPtrT>(dataPtr), ldm);
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_OPAQUE_STORE_HPP
