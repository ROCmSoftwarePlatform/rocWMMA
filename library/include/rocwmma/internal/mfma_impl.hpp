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
#ifndef ROCWMMA_MFMA_IMPL_HPP
#define ROCWMMA_MFMA_IMPL_HPP

#include "convert.hpp"
#include "io_traits.hpp"
#include "types.hpp"
#include "vector.hpp"

namespace rocwmma
{

    namespace detail
    {

        template <typename InputT, typename ComputeT, uint32_t BlockM, uint32_t BlockN>
        struct amdgcn_mfma
        {
            template <typename RegsA, typename RegsB, typename RegsC>
            ROCWMMA_DEVICE static inline auto exec(RegsA&& regsA, RegsB&& regsB, RegsC& regsC)
            {
                return regsC;
            }
        };

// MFMA is MI architecture specific
#if ROCWMMA_ARCH_GFX9

        template <>
        struct amdgcn_mfma<float16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x16f16(
                    regsA.data, regsB.data, regsC.data, 0, 0, 0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<float16_t, float16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x2;
                using DRegsT = AccRegF32x2;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<float16_t, float32_t, 16, 16>;
                using Pack16            = PackUtil<float16_t>;
                using Convert_fp16_fp32 = Convert<float16_t, float32_t>;
                using Convert_fp32_fp16 = Convert<float32_t, float16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D
                // to fp16 as 'simulated' fp16 computation
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_fp16_fp32::exec(Pack16::unpack(regsC)));
                return Pack16::pack(Convert_fp32_fp16::exec(Dfp32));
            }
        };

        template <>
        struct amdgcn_mfma<float16_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x8f16(
                    regsA.data, regsB.data, regsC.data, 0, 0, 0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<float16_t, float16_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<float16_t, float32_t, 32, 32>;
                using PackCD            = PackUtil<float16_t>;
                using Convert_fp16_fp32 = Convert<float16_t, float32_t>;
                using Convert_fp32_fp16 = Convert<float32_t, float16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to fp16 result;
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_fp16_fp32::exec(PackCD::unpack(regsC)));
                return PackCD::pack(Convert_fp32_fp16::exec(Dfp32));
            }
        };

#if !ROCWMMA_NO_HALF
        template <>
        struct amdgcn_mfma<hfloat16_t, float32_t, 16, 16>
            : public amdgcn_mfma<float16_t, float32_t, 16, 16>
        {
        };

        template <>
        struct amdgcn_mfma<hfloat16_t, hfloat16_t, 16, 16>
            : public amdgcn_mfma<float16_t, float16_t, 16, 16>
        {
        };

        template <>
        struct amdgcn_mfma<hfloat16_t, float32_t, 32, 32>
            : public amdgcn_mfma<float16_t, float32_t, 32, 32>
        {
        };

        template <>
        struct amdgcn_mfma<hfloat16_t, hfloat16_t, 32, 32>
            : public amdgcn_mfma<float16_t, float16_t, 32, 32>
        {
        };
#endif // !ROCWMMA_NO_HALF

#if !ROCWMMA_ARCH_GFX908

        // NOTE: Successors to gfx908 have upgraded bf16 instructions
        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    regsA.data, regsB.data, regsC.data, 0, 0, 0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x2;
                using DRegsT = AccRegF32x2;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 16, 16>;
                using PackCD            = PackUtil<bfloat16_t>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(PackCD::unpack(regsC)));
                return PackCD::pack(Convert_fp32_bf16::exec(Dfp32));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(
                    regsA.data, regsB.data, regsC.data, 0, 0, 0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 32, 32>;
                using PackCD            = PackUtil<bfloat16_t>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(PackCD::unpack(regsC)));
                return PackCD::pack(Convert_fp32_bf16::exec(Dfp32));
            }
        };

#else // ROCWMMA_ARCH_GFX908

        // NOTE: gfx908 architecture supports only subset of bf16 instructions
        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decltype(regsA)),
                              "Inconsistent data formats");

                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x8bf16(
                    reinterpret_cast<TypeIn const&>(regsA).data,
                    reinterpret_cast<TypeIn const&>(regsB).data,
                    regsC.data,
                    0,
                    0,
                    0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8,
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x2;
                using DRegsT = AccRegF32x2;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 16, 16>;
                using PackCD            = PackUtil<bfloat16_t>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(PackCD::unpack(regsC)));
                return PackCD::pack(Convert_fp32_bf16::exec(Dfp32));
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decltype(regsA)),
                              "Inconsistent data formats");

                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x4bf16(
                    reinterpret_cast<TypeIn const&>(regsA).data,
                    reinterpret_cast<TypeIn const&>(regsB).data,
                    regsC.data,
                    0,
                    0,
                    0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4,
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x8;
                using DRegsT = AccRegF32x8;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                using Mfma              = amdgcn_mfma<bfloat16_t, float32_t, 32, 32>;
                using PackCD            = PackUtil<bfloat16_t>;
                using Convert_bf16_fp32 = Convert<bfloat16_t, float32_t>;
                using Convert_fp32_bf16 = Convert<float32_t, bfloat16_t>;

                // MFMA unit compute type is always fp32.
                // Upconvert C to fp32, do MFMA, then down convert D to bf16 result
                auto Dfp32
                    = Mfma::exec(regsA, regsB, Convert_bf16_fp32::exec(PackCD::unpack(regsC)));
                return PackCD::pack(Convert_fp32_bf16::exec(Dfp32));
            }
        };

#endif // !ROCWMMA_ARCH_GFX908
#if(!ROCWMMA_ARCH_GFX940) && (!ROCWMMA_ARCH_GFX941) && (!ROCWMMA_ARCH_GFX942)

        template <>
        struct amdgcn_mfma<int8_t, int32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegI32x1;
                using BRegsT = VRegI32x1;
                using CRegsT = AccRegI32x16;
                using DRegsT = AccRegI32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_i32_32x32x8i8(
                    regsA.data[0], regsB.data[0], regsC.data, 0, 0, 0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<int8_t, int32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegI32x1;
                using BRegsT = VRegI32x1;
                using CRegsT = AccRegI32x4;
                using DRegsT = AccRegI32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_i32_16x16x16i8(
                    regsA.data[0], regsB.data[0], regsC.data, 0, 0, 0)};
                return result;
            }
        };

#else // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942

        template <>
        struct amdgcn_mfma<int8_t, int32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegI32x2;
                using BRegsT = VRegI32x2;
                using CRegsT = AccRegI32x16;
                using DRegsT = AccRegI32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                using inputType = VRegI64x1;
                result.data
                    = {__builtin_amdgcn_mfma_i32_32x32x16_i8(((inputType const&)(regsA)).data[0],
                                                             ((inputType const&)(regsB)).data[0],
                                                             regsC.data,
                                                             0,
                                                             0,
                                                             0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<int8_t, int32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 32
                };
                using ARegsT = VRegI32x2;
                using BRegsT = VRegI32x2;
                using CRegsT = AccRegI32x4;
                using DRegsT = AccRegI32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                using inputType = VRegI64x1;
                result.data
                    = {__builtin_amdgcn_mfma_i32_16x16x32_i8(((inputType const&)(regsA)).data[0],
                                                             ((inputType const&)(regsB)).data[0],
                                                             regsC.data,
                                                             0,
                                                             0,
                                                             0)};
                return result;
            }
        };

#endif // (!ROCWMMA_ARCH_GFX940) && (!ROCWMMA_ARCH_GFX941) && (!ROCWMMA_ARCH_GFX942)

        template <>
        struct amdgcn_mfma<float32_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x4f32(
                    regsA.data[0], regsB.data[0], regsC.data, 0, 0, 0)};
                return result;
            }
        };

        // Single 32 x 32 block mfma
        template <>
        struct amdgcn_mfma<float32_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 2
                };
                using ARegsT = VRegF32x1;
                using BRegsT = VRegF32x1;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x2f32(
                    regsA.data[0], regsB.data[0], regsC.data, 0, 0, 0)};
                return result;
            }
        };

#if !ROCWMMA_ARCH_GFX908

        // NOTE: Successors to gfx908 support fp64 mfma
        template <>
        struct amdgcn_mfma<float64_t, float64_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF64x1;
                using BRegsT = VRegF64x1;
                using CRegsT = AccRegF64x4;
                using DRegsT = AccRegF64x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f64_16x16x4f64(
                    regsA.data[0], regsB.data[0], regsC.data, 0, 0, 0)};
                return result;
            }
        };

#else // !ROCWMMA_ARCH_GFX908

        // Required for general fp64 support
        template <>
        struct amdgcn_mfma<float64_t, float64_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF64x1;
                using BRegsT = VRegF64x1;
                using CRegsT = AccRegF64x4;
                using DRegsT = AccRegF64x4;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 lacks support for fp64 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("fp64 mfma not supported on gfx908")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC)

                -> typename Traits::DRegsT const&
            {
                return regsC;
            }
        };

#endif // !ROCWMMA_ARCH_GFX908

#if ROCWMMA_ARCH_GFX94X

        template <>
        struct amdgcn_mfma<float8_fnuz_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 32
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                using inputType = VRegI64x1;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(((inputType const&)(regsA)).data[0],
                                                               ((inputType const&)(regsB)).data[0],
                                                               regsC.data,
                                                               0,
                                                               0,
                                                               0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<float8_fnuz_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                using inputType = VRegI64x1;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(((inputType const&)(regsA)).data[0],
                                                               ((inputType const&)(regsB)).data[0],
                                                               regsC.data,
                                                               0,
                                                               0,
                                                               0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat8_fnuz_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 32
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                using inputType = VRegI64x1;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(((inputType const&)(regsA)).data[0],
                                                               ((inputType const&)(regsB)).data[0],
                                                               regsC.data,
                                                               0,
                                                               0,
                                                               0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat8_fnuz_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                using inputType = VRegI64x1;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(((inputType const&)(regsA)).data[0],
                                                               ((inputType const&)(regsB)).data[0],
                                                               regsC.data,
                                                               0,
                                                               0,
                                                               0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<xfloat32_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x8_xf32(
                    regsA.data, regsB.data, regsC.data, 0, 0, 0)};
                return result;
            }
        };

        template <>
        struct amdgcn_mfma<xfloat32_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                typename Traits::DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x4_xf32(
                    regsA.data, regsB.data, regsC.data, 0, 0, 0)};
                return result;
            }
        };

#else // (!ROCWMMA_ARCH_GFX940) && (!ROCWMMA_ARCH_GFX941) && (!ROCWMMA_ARCH_GFX942)

        template <>
        struct amdgcn_mfma<float8_fnuz_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 32
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for fp8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("fp8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC)

                -> typename Traits::DRegsT const&
            {
                return regsC;
            }
        };

        template <>
        struct amdgcn_mfma<float8_fnuz_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for fp8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("fp8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return regsC;
            }
        };

        // Required for general bf8 support
        template <>
        struct amdgcn_mfma<bfloat8_fnuz_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 32
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for bf8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("bf8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC)

                -> typename Traits::DRegsT const&
            {
                return regsC;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat8_fnuz_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for bf8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("bf8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return regsC;
            }
        };

        template <>
        struct amdgcn_mfma<xfloat32_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 8
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for xf32 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("xf32 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC)

                -> typename Traits::DRegsT const&
            {
                return regsC;
            }
        };

        template <>
        struct amdgcn_mfma<xfloat32_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 4
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for xf32 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("xf32 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return regsC;
            }
        };

#endif // ROCWMMA_ARCH_GFX940 || ROCWMMA_ARCH_GFX941 || ROCWMMA_ARCH_GFX942

        // Required for general fp8 support
        template <>
        struct amdgcn_mfma<float8_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 32
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for fp8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("fp8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC)

                -> typename Traits::DRegsT const&
            {
                return regsC;
            }
        };

        template <>
        struct amdgcn_mfma<float8_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for fp8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("fp8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return regsC;
            }
        };

        // Required for general bf8 support
        template <>
        struct amdgcn_mfma<bfloat8_t, float32_t, 16, 16>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 32
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x4;
                using DRegsT = AccRegF32x4;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for bf8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("bf8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC)

                -> typename Traits::DRegsT const&
            {
                return regsC;
            }
        };

        template <>
        struct amdgcn_mfma<bfloat8_t, float32_t, 32, 32>
        {
            // Packed register traits
            struct Traits
            {
                enum : uint32_t
                {
                    KPerMfma = 16
                };
                using ARegsT = VRegF32x2;
                using BRegsT = VRegF32x2;
                using CRegsT = AccRegF32x16;
                using DRegsT = AccRegF32x16;
            };

            // This implementation is needed to satisfy the MmaSyncTest interface,
            // and WILL not function as intended.
            // gfx908 and gfx90a lacks support for bf8 MFMA instructions.
            ROCWMMA_UNSUPPORTED_IMPL("bf8 mfma not supported on gfx908/gfx90a")
            ROCWMMA_DEVICE static inline auto exec(typename Traits::ARegsT const& regsA,
                                                   typename Traits::BRegsT const& regsB,
                                                   typename Traits::CRegsT const& regsC) ->
                typename Traits::DRegsT
            {
                return regsC;
            }
        };

#endif // ROCWMMA_ARCH_GFX9

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_MFMA_IMPL_HPP
