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
#ifndef ROCWMMA_WMMA_HPP
#define ROCWMMA_WMMA_HPP

#include "mma.hpp"
#include "wmma_impl.hpp"

namespace rocwmma
{
    // Expose WMMA implementation backend
    template<typename InputTA,
             typename InputTB,
             typename ComputeT,
             uint32_t BlockM,
             uint32_t BlockN>
    using Wmma_impl = detail::amdgcn_wmma<InputTA, InputTB, ComputeT, BlockM, BlockN>;

    // Wmma interface through Mma
    template <uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              typename InputTA,
              typename InputTB,
              typename ComputeT,
              uint32_t BlockM,
              uint32_t BlockN = BlockM,
              MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR>
    struct Wmma : public Mma<FragM, FragN, FragK, Wmma_impl<InputTA, InputTB, ComputeT, BlockM, BlockN>, AccumPolicy>
    {
        // Interface:
        // template <typename VecTA, typename VecTB, typename VecTC>
        // ROCWMMA_DEVICE static inline decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC& accum);
    };

    template<typename InputTA_In,
             typename InputTB_In,
             typename ComputeT_In,
             uint32_t BlockM_In,
             uint32_t BlockN_In>
    struct MmaTraits<Wmma_impl<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In>>
    {
        // Base implementation
        using Base = Wmma_impl<InputTA_In, InputTB_In, ComputeT_In, BlockM_In, BlockN_In>;

        // Operand types
        using InputTA = InputTA_In;
        using InputTB = InputTB_In;
        using ComputeT = ComputeT_In;

        // Raw input / output types
        using ARegsT = typename Base::ARegsT;
        using BRegsT = typename Base::BRegsT;
        using CRegsT = typename Base::CRegsT;
        using DRegsT = typename Base::DRegsT;

        // Geometric block sizes
        constexpr static uint32_t BlockM = BlockM_In;
        constexpr static uint32_t BlockN = BlockN_In;
        constexpr static uint32_t BlockK = Base::KPerMma;

        // Vector sizes per block (packed)
        constexpr static uint32_t BlockSizeA = VecTraits<ARegsT>::size();
        constexpr static uint32_t BlockSizeB = VecTraits<BRegsT>::size();
        constexpr static uint32_t BlockSizeC = VecTraits<CRegsT>::size();

        // Backend flag
        constexpr static bool is_wmma = true;
        constexpr static bool is_mfma = false;
    };

} // namespace rocwmma

#endif // ROCWMMA_WMMA_HPP
