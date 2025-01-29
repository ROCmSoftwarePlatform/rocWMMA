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
        enum struct MfmaCtrlFlags: uint32_t
        {
            DEFAULT = 0u,
        };

        struct Unsupported;
        struct Gfx9;

        // SFINAE target enabler for gfx9 with conditional check
        template <typename TestTarget, bool Cond = true>
        using enable_gfx9_t = enable_if_t<(bool)ROCWMMA_ARCH_GFX9 && is_same_v<TestTarget, Gfx9> && Cond, Gfx9>;

        /*! \class amdgcn_mfma
        *  \brief  Builtin wrapper for mfma instructions
        *  @tparam InputTA Datatype of input A
        *  @tparam InputTB Datatype of input B
        *  @tparam ComputeT Datatype of accumulator
        *  @tparam BlockM M-dimension of wmma block
        *  @tparam BlockN N-dimension of wmma block
        *  @tparam GfxTarget The current gfx family target of interest being compiled
        *  @tparam TargetEnable Enabler for the current target if supported
        */
        template <typename InputTA,
                 typename InputTB,
                 typename ComputeT,
                 uint32_t BlockM,
                 uint32_t BlockN,
                 typename GfxTarget = conditional_t<(bool)ROCWMMA_ARCH_GFX9, Gfx9, Unsupported>,
                 typename TargetEnable = GfxTarget>
        struct amdgcn_mfma
        {
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Choose reasonable minimal default params to satisfy static checks
            constexpr static uint32_t KPerMma = 8u;

        private:
            using PackTraitsA = PackTraits<InputTA>;
            using PackTraitsB = PackTraits<InputTB>;
            using PackTraitsAcc = PackTraits<ComputeT>;

            constexpr static uint32_t InputASize = BlockM * KPerMma / (Constants::AMDGCN_WAVE_SIZE * PackTraitsA::PackRatio);
            constexpr static uint32_t InputBSize = BlockN * KPerMma / (Constants::AMDGCN_WAVE_SIZE * PackTraitsB::PackRatio);
            constexpr static uint32_t AccumSize = BlockM * BlockM / (Constants::AMDGCN_WAVE_SIZE * PackTraitsAcc::PackRatio);

        public:

            using ARegsT = VecT<typename PackTraitsA::PackedT, InputASize>;
            using BRegsT = VecT<typename PackTraitsB::PackedT, InputBSize>;
            using CRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;
            using DRegsT = VecT<typename PackTraitsAcc::PackedT, AccumSize>;

            template <typename RegsA, typename RegsB, typename RegsC>
            ROCWMMA_DEVICE static inline decltype(auto) exec(RegsA&& regsA, RegsB&& regsB, RegsC&& regsC)
            {
                return forward<RegsC>(regsC);
            }
        };

#if ROCWMMA_ARCH_GFX9

        // Non-B32 compute types
        // Note: MFMA unit accum type is always b32 size.
        // Since we cannot natively accumulate in the desired type,
        // we must convert to native accum type, perform mfma and convert
        // the accum result back to the desired type.
        // Warning: This can be very slow!
        template <typename InputTA,
                 typename InputTB,
                 typename ComputeT,
                 uint32_t BlockM,
                 uint32_t BlockN,
                 typename GfxTarget>
        struct amdgcn_mfma<InputTA, InputTB, ComputeT, BlockM, BlockN, GfxTarget, enable_gfx9_t<GfxTarget, (sizeof(ComputeT) < 4u)>>
        {
        private:
            using PackTraits = PackTraits<ComputeT>;
            using PackUtil = PackUtil<ComputeT>;

            // B32 mfma traits
            using AccumDataT = typename PackTraits::PackedT;
            using MfmaB32 = amdgcn_mfma<InputTA, InputTB, AccumDataT, BlockM, BlockN>;
            using AccumTraitsB32 = VecTraits<typename MfmaB32::CRegsT>;

            // ComputeT mfma traits
            // Scale accum registers by pack ratio, due to ComputeT
            using AccumRegsT = typename AccumTraitsB32::template VecT<AccumDataT, AccumTraitsB32::size() / PackTraits::PackRatio>;

        public:
            constexpr static uint32_t KPerMma = MfmaB32::KPerMma;
            constexpr static MfmaCtrlFlags Cbsz = MfmaB32::Cbsz;
            constexpr static MfmaCtrlFlags Abid = MfmaB32::Abid;
            constexpr static MfmaCtrlFlags Blgp = MfmaB32::Blgp;

            // Packed register types
            using ARegsT = typename MfmaB32::ARegsT;
            using BRegsT = typename MfmaB32::BRegsT;
            using CRegsT = AccumRegsT;
            using DRegsT = AccumRegsT;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // 'Packing' dichotomy:
                // - Registers arguments going in or out are assumed to
                //   be in 'packed' form (e.g., the physical space they consume).
                // - B32 native compute type is the 'packed' datatype;
                //   Desired compute type is the 'unpacked' datatype.
                using ConvertUp = Convert<typename PackTraits::UnpackedT, typename PackTraits::PackedT>;
                using ConvertDown = Convert<typename PackTraits::PackedT, typename PackTraits::UnpackedT>;

                auto unpacked_result
                    = MfmaB32::exec(regsA, regsB, ConvertUp::exec(PackUtil::unpack(regsC)));
                return PackUtil::pack(ConvertDown::exec(unpacked_result));
            }
        };

        // fp16
        template <typename GfxTarget>
        struct amdgcn_mfma<float16_t, float16_t, float32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x16f16(regsA.data, regsB.data, regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<float16_t, float16_t, float32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget>>
        {
            constexpr static uint32_t KPerMma = 8u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x8f16(regsA.data, regsB.data, regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        // hfloat16 derivative
        template <uint32_t BlockM, uint32_t BlockN, typename GfxTarget>
        struct amdgcn_mfma<hfloat16_t, hfloat16_t, float32_t, BlockM, BlockN, GfxTarget, enable_gfx9_t<GfxTarget, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_mfma<float16_t, float16_t, float32_t, BlockM, BlockN>
        {
        };

        template <uint32_t BlockM, uint32_t BlockN, typename GfxTarget>
        struct amdgcn_mfma<hfloat16_t, hfloat16_t, hfloat16_t, BlockM, BlockN, GfxTarget, enable_gfx9_t<GfxTarget, !(bool)ROCWMMA_NO_HALF>>
            : public amdgcn_mfma<float16_t, float16_t, float16_t, BlockM, BlockN>
        {
        };

        // bf16
        template <typename GfxTarget>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget, (bool)ROCWMMA_ARCH_GFX908>>
        {
            constexpr static uint32_t KPerMma = 8u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x8bf16(
                    ((TypeIn const&)(regsA)).data,
                    ((TypeIn const&)(regsB)).data,
                    regsC.data,
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, float32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget, (bool)ROCWMMA_ARCH_GFX908>>
        {
            constexpr static uint32_t KPerMma = 4u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                // Built-in expects unpacked vector of short.
                // Strange, but OK we can do that here.
                using TypeIn = VecT<short, 2>;

                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x4bf16(
                    ((TypeIn const&)(regsA)).data,
                    ((TypeIn const&)(regsB)).data,
                    regsC.data,
                    (int)Cbsz,
                    (int)Abid,
                    (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, float32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                ((bool)ROCWMMA_ARCH_GFX90A
                                                                                                || (bool)ROCWMMA_ARCH_GFX94X)>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(regsA.data, regsB.data, regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<bfloat16_t, bfloat16_t, float32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                ((bool)ROCWMMA_ARCH_GFX90A
                                                                                                 || (bool)ROCWMMA_ARCH_GFX94X)>>
        {
            constexpr static uint32_t KPerMma = 8u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x8bf16_1k(regsA.data, regsB.data, regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        // fp32
        template <typename GfxTarget>
        struct amdgcn_mfma<float32_t, float32_t, float32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget>>
        {
            constexpr static uint32_t KPerMma = 4u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x4f32(regsA.data[0], regsB.data[0], regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<float32_t, float32_t, float32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget>>
        {
            constexpr static uint32_t KPerMma = 2u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x1;
            using BRegsT = VRegF32x1;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x2f32(regsA.data[0], regsB.data[0], regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        // fp64
        template <typename GfxTarget>
        struct amdgcn_mfma<float64_t, float64_t, float64_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                               ((bool)ROCWMMA_ARCH_GFX90A
                                                                                               || (bool)ROCWMMA_ARCH_GFX94X)>>
        {
            constexpr static uint32_t KPerMma = 4u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF64x1;
            using BRegsT = VRegF64x1;
            using CRegsT = AccRegF64x4;
            using DRegsT = AccRegF64x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f64_16x16x4f64(regsA.data[0], regsB.data[0], regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        // int8
        template <typename GfxTarget>
        struct amdgcn_mfma<int8_t, int8_t, int32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                      ((bool)ROCWMMA_ARCH_GFX908
                                                                                      || (bool)ROCWMMA_ARCH_GFX90A)>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x4;
            using DRegsT = AccRegI32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_i32_16x16x16i8(regsA.data[0], regsB.data[0], regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<int8_t, int8_t, int32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                      ((bool)ROCWMMA_ARCH_GFX908
                                                                                      || (bool)ROCWMMA_ARCH_GFX90A)>>
        {
            constexpr static uint32_t KPerMma = 8u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x1;
            using BRegsT = VRegI32x1;
            using CRegsT = AccRegI32x16;
            using DRegsT = AccRegI32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_i32_32x32x8i8(regsA.data[0], regsB.data[0], regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<int8_t, int8_t, int32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                       (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 32u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x4;
            using DRegsT = AccRegI32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data
                    = {__builtin_amdgcn_mfma_i32_16x16x32_i8(((TypeIn const&)(regsA)).data[0],
                                                             ((TypeIn const&)(regsB)).data[0],
                                                             regsC.data,
                                                             (int)Cbsz,
                                                             (int)Abid,
                                                             (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<int8_t, int8_t, int32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                    (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegI32x2;
            using BRegsT = VRegI32x2;
            using CRegsT = AccRegI32x16;
            using DRegsT = AccRegI32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data
                    = {__builtin_amdgcn_mfma_i32_32x32x16_i8(((TypeIn const&)(regsA)).data[0],
                                                             ((TypeIn const&)(regsB)).data[0],
                                                             regsC.data,
                                                             (int)Cbsz,
                                                             (int)Abid,
                                                             (int)Blgp)};
                return result;
            }
        };

        // f8_fnuz
        template <typename GfxTarget>
        struct amdgcn_mfma<float8_fnuz_t, float8_fnuz_t, float32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                      (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 32u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(((TypeIn const&)(regsA)).data[0],
                                                               ((TypeIn const&)(regsB)).data[0],
                                                               regsC.data,
                                                               (int)Cbsz,
                                                               (int)Abid,
                                                               (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<float8_fnuz_t, float8_fnuz_t, float32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                        (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(((TypeIn const&)(regsA)).data[0],
                                                               ((TypeIn const&)(regsB)).data[0],
                                                               regsC.data,
                                                               (int)Cbsz,
                                                               (int)Abid,
                                                               (int)Blgp)};
                return result;
            }
        };

        // bf8_fnuz
        template <typename GfxTarget>
        struct amdgcn_mfma<bfloat8_fnuz_t, bfloat8_fnuz_t, float32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                        (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 32u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(((TypeIn const&)(regsA)).data[0],
                                                               ((TypeIn const&)(regsB)).data[0],
                                                               regsC.data,
                                                               (int)Cbsz,
                                                               (int)Abid,
                                                               (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<bfloat8_fnuz_t, bfloat8_fnuz_t, float32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                        (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 16u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                using TypeIn = VRegI64x1;
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsA)>), "Inconsistent data formats");
                static_assert(sizeof(TypeIn) == sizeof(decay_t<decltype(regsB)>), "Inconsistent data formats");

                DRegsT result;
                result.data     = {
                    __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(((TypeIn const&)(regsA)).data[0],
                                                               ((TypeIn const&)(regsB)).data[0],
                                                               regsC.data,
                                                               (int)Cbsz,
                                                               (int)Abid,
                                                               (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<xfloat32_t, xfloat32_t, float32_t, 16u, 16u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                 (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 8u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x4;
            using DRegsT = AccRegF32x4;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_16x16x8_xf32(regsA.data, regsB.data, regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

        template <typename GfxTarget>
        struct amdgcn_mfma<xfloat32_t, xfloat32_t, float32_t, 32u, 32u, GfxTarget, enable_gfx9_t<GfxTarget,
                                                                                                (bool)ROCWMMA_ARCH_GFX94X>>
        {
            constexpr static uint32_t KPerMma = 4u;
            constexpr static MfmaCtrlFlags Cbsz = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Abid = MfmaCtrlFlags::DEFAULT;
            constexpr static MfmaCtrlFlags Blgp = MfmaCtrlFlags::DEFAULT;

            // Packed register types
            using ARegsT = VRegF32x2;
            using BRegsT = VRegF32x2;
            using CRegsT = AccRegF32x16;
            using DRegsT = AccRegF32x16;

            ROCWMMA_DEVICE static inline auto exec(ARegsT const& regsA, BRegsT const& regsB, CRegsT const& regsC) -> DRegsT
            {
                DRegsT result;
                result.data = {__builtin_amdgcn_mfma_f32_32x32x4_xf32(regsA.data, regsB.data, regsC.data, (int)Cbsz, (int)Abid, (int)Blgp)};
                return result;
            }
        };

#endif // ROCWMMA_ARCH_GFX9

    } // namespace detail

} // namespace rocwmma

#endif // ROCWMMA_MFMA_IMPL_HPP
