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
#ifndef ROCWMMA_WMMA_IMPL_HPP
#define ROCWMMA_WMMA_IMPL_HPP

#include "types.hpp"
#include "utility/type_traits.hpp"

namespace rocwmma
{
    namespace detail
    {
        template <typename InputTA,
                 typename InputTB,
                 typename ComputeT,
                 uint32_t BlockM,
                 uint32_t BlockN,
                 typename Dummy = true_type,
                 typename Enabler = void>
        struct amdgcn_wmma
        {
            constexpr static uint32_t KPerMma = 16u;
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            template <typename RegsA, typename RegsB, typename RegsC>
            ROCWMMA_DEVICE static inline decltype(auto) exec(RegsA&& regsA, RegsB&& regsB, RegsC&& regsC)
            {
                return regsC;
            }
        };

        enum struct WmmaCtrlFlags: bool
        {
            // Output register selection of WMMA.
            // Low = bits [15:0]
            // High = bits[31:16]
            LOW  = false,
            HIGH = true,

            // Signage indicator of inputs / accum
            UNSIGNED = false,
            SIGNED   = true
        };

        // gfx11 implementations
        template <typename Dummy>
        struct amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX11>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX11>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32(regsA.data, regsB.data, regsC.data, (bool)AccumBits)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX11>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX11>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x8;
            using BRegsT = VRegF32x8;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(regsA.data, regsB.data, regsC.data, (bool)AccumBits)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX11>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x4;
            using BRegsT = VRegI32x4;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32((bool)InputSign,
                                                                          regsA.data,
                                                                          (bool)InputSign,
                                                                          regsB.data,
                                                                          regsC.data,
                                                                          (bool)AccumSign)};
                return result;
            }
        };

        // gfx12 implementations
        template <typename Dummy>
        struct amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX12, int>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX12, int>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX12, int>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<bfloat16_t, bfloat16_t, bfloat16_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX12, int>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x4;
            using BRegsT = VRegF32x4;
            using CRegsT = VRegF32x4;
            using DRegsT = VRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(regsA.data, regsB.data, regsC.data)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<int8_t, int8_t, int32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX12, int>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x8;
            using DRegsT = AccRegI32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12((bool)InputSign,
                                                                                regsA.data,
                                                                                (bool)InputSign,
                                                                                regsB.data,
                                                                                regsC.data,
                                                                                (bool)AccumSign)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<float8_t, float8_t, float32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX12>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                    reinterpret_cast<TypeIn const&>(regsA).data,
                    reinterpret_cast<TypeIn const&>(regsB).data,
                    regsC.data)};
                return result;
            }
        };

        template <typename Dummy>
        struct amdgcn_wmma<bfloat8_t, bfloat8_t, float32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value && (bool)ROCWMMA_ARCH_GFX12>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static WmmaCtrlFlags InputSign = WmmaCtrlFlags::SIGNED;
            constexpr static WmmaCtrlFlags AccumBits = WmmaCtrlFlags::LOW;
            constexpr static WmmaCtrlFlags AccumSign = WmmaCtrlFlags::SIGNED;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x8;
            using DRegsT = AccRegF32x8;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects vector of int.
                using TypeIn = VecT<int, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data = {__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
                    reinterpret_cast<TypeIn const&>(regsA).data,
                    reinterpret_cast<TypeIn const&>(regsB).data,
                    regsC.data)};
                return result;
            }
        };

        // Derivative implementations
        template <typename Dummy>
        struct amdgcn_wmma<hfloat16_t, hfloat16_t, float32_t, 16u, 16u, Dummy, enable_if_t<Dummy::value
                                                                       &&((bool)ROCWMMA_ARCH_GFX11 || (bool)ROCWMMA_ARCH_GFX12)
                                                                       && !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_wmma<float16_t, float16_t, float32_t, 16u, 16u>
        {
        };

        template <typename Dummy>
        struct amdgcn_wmma<hfloat16_t, hfloat16_t, hfloat16_t, 16u, 16u, Dummy, enable_if_t<Dummy::value
                                                                       &&((bool)ROCWMMA_ARCH_GFX11 || (bool)ROCWMMA_ARCH_GFX12)
                                                                       && !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_wmma<float16_t, float16_t, float16_t, 16u, 16u>
        {
        };

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_WMMA_IMPL_HPP
