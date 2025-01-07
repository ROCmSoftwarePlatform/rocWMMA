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
#ifndef ROCWMMA_FLOW_CONTROL_HPP
#define ROCWMMA_FLOW_CONTROL_HPP

namespace rocwmma
{

    namespace detail
    {
        // Perform synchronization across fragments(wavefronts) in a workgroup
        struct amdgcn_barrier
        {
            ROCWMMA_DEVICE static inline auto exec()
            {
                return __builtin_amdgcn_s_barrier();
            }
        };

        // Fine tune scheduler behavior
        template <int32_t mask = 0>
        struct amdgcn_sched_barrier
        {
            ROCWMMA_DEVICE static inline auto exec()
            {
                return __builtin_amdgcn_sched_barrier(mask);
            }
        };

        // Modifies the priority of the current wavefront
        template <int32_t priority = 0>
        struct amdgcn_setprio
        {
            enum : const int16_t
            {
                priority16 = priority
            };

            ROCWMMA_DEVICE static inline auto exec()
            {
                static_assert(priority16 >= 0 && priority16 <= 3, "Priority must be from 0 to 3");

                return __builtin_amdgcn_s_setprio(priority16);
            }
        };

        template <int32_t vmcnt, int32_t lgkmcnt>
        struct amdgcn_s_waitcnt
        {
            enum : const int16_t
            {
                vmcnt16   = (((0xF) & vmcnt) | (((0x30) & vmcnt) << 10)),
                lgkmcnt16 = (((0xF) & lgkmcnt) << 8),
                cnt       = vmcnt16 | lgkmcnt16
            };

            ROCWMMA_DEVICE static inline auto exec()
            {
                static_assert(vmcnt >= 0 && vmcnt < 64,
                              "Vector memory operations allocated a maximum of 6 bits");
                static_assert(lgkmcnt >= 0 && lgkmcnt < 16,
                              "Scalar mem/LDS/GDS allocated a maximum of 4 bits");

                return __builtin_amdgcn_s_waitcnt(cnt);
            }
        };

        template <int32_t vmcnt>
        struct amdgcn_s_vmcnt : public amdgcn_s_waitcnt<vmcnt, 0>
        {
        };

        template <int32_t lgkmcnt>
        struct amdgcn_s_lgkmcnt : public amdgcn_s_waitcnt<0, lgkmcnt>
        {
        };

    } // namespace detail

    using Barrier = detail::amdgcn_barrier;

    template <int32_t mask>
    using SchedBarrier = detail::amdgcn_sched_barrier<mask>;

    template <int32_t priority>
    using SetPrio = detail::amdgcn_setprio<priority>;

    template <int32_t vmcnt, int32_t lgkmcnt>
    using Waitcnt = detail::amdgcn_s_waitcnt<vmcnt, lgkmcnt>;

    template <int32_t vmcnt>
    using WaitVmcnt = detail::amdgcn_s_vmcnt<vmcnt>;

    template <int32_t lgkmcnt>
    using WaitLgkmcnt = detail::amdgcn_s_lgkmcnt<lgkmcnt>;

} // namespace rocwmma

#endif // ROCWMMA_FLOW_CONTROL_HPP
