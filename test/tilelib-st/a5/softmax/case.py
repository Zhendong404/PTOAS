#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""PTODSL migration of the legacy online softmax update case.

The legacy kernel combines public tile movement with a raw vector-scope
online update.  This case keeps that composite structure, while using the
public ``pto.tile.load``/``pto.tile.store`` boundaries and native Python
control flow in the authored kernel.

The legacy launcher used three eight-row blocks.  The TileLib runner currently
launches one block, so this first migration binds the same 24-row workload to
one vector kernel and records the scheduling difference in MIGRATION_STATUS.md.
The numerical contract and the valid ``seq=73`` region are unchanged.
"""

from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import assert_close, auto_main
from ptodsl import pto


ROWS = 24
COLS = 128
SEQ = 73
LANES = 64
EPS = 1e-4

# Explicit mode is required by vecscope and therefore uses level3 emission.
# Keep the UB buffers disjoint and 32-byte aligned, as in the legacy kernel.
OLDMAX_ADDR = 0
OLDSUM_ADDR = 128
QK_ADDR = 256
OUT_ADDR = QK_ADDR + ROWS * COLS * 4
NEWMAX_ADDR = OUT_ADDR + ROWS * COLS * 4
NEWSUM_ADDR = NEWMAX_ADDR + 128
EXPMAX_ADDR = NEWSUM_ADDR + 128


@pto.jit(
    name="softmax_f32_rows24_seq73",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def _kernel(
    oldmax_ptr: pto.ptr(pto.f32, "gm"),
    oldsum_ptr: pto.ptr(pto.f32, "gm"),
    qk_ptr: pto.ptr(pto.f32, "gm"),
    newmax_ptr: pto.ptr(pto.f32, "gm"),
    newsum_ptr: pto.ptr(pto.f32, "gm"),
    expmax_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
):
    oldmax_view = pto.make_tensor_view(oldmax_ptr, shape=[ROWS, 1], strides=[1, 1])
    oldsum_view = pto.make_tensor_view(oldsum_ptr, shape=[ROWS, 1], strides=[1, 1])
    qk_view = pto.make_tensor_view(qk_ptr, shape=[ROWS, COLS], strides=[COLS, 1])
    newmax_view = pto.make_tensor_view(newmax_ptr, shape=[ROWS, 1], strides=[1, 1])
    newsum_view = pto.make_tensor_view(newsum_ptr, shape=[ROWS, 1], strides=[1, 1])
    expmax_view = pto.make_tensor_view(expmax_ptr, shape=[ROWS, 1], strides=[1, 1])
    out_view = pto.make_tensor_view(out_ptr, shape=[ROWS, COLS], strides=[COLS, 1])

    oldmax_tile = pto.alloc_tile(
        shape=[ROWS, 1], dtype=pto.f32, valid_shape=[ROWS, 1], blayout="ColMajor", addr=OLDMAX_ADDR
    )
    oldsum_tile = pto.alloc_tile(
        shape=[ROWS, 1], dtype=pto.f32, valid_shape=[ROWS, 1], blayout="ColMajor", addr=OLDSUM_ADDR
    )
    qk_tile = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.f32, valid_shape=[ROWS, SEQ], addr=QK_ADDR
    )
    out_tile = pto.alloc_tile(
        shape=[ROWS, COLS], dtype=pto.f32, valid_shape=[ROWS, SEQ], addr=OUT_ADDR
    )
    newmax_tile = pto.alloc_tile(
        shape=[ROWS, 1], dtype=pto.f32, valid_shape=[ROWS, 1], blayout="ColMajor", addr=NEWMAX_ADDR
    )
    newsum_tile = pto.alloc_tile(
        shape=[ROWS, 1], dtype=pto.f32, valid_shape=[ROWS, 1], blayout="ColMajor", addr=NEWSUM_ADDR
    )
    expmax_tile = pto.alloc_tile(
        shape=[ROWS, 1], dtype=pto.f32, valid_shape=[ROWS, 1], blayout="ColMajor", addr=EXPMAX_ADDR
    )

    pto.tile.load(oldmax_view, oldmax_tile)
    pto.tile.load(oldsum_view, oldsum_tile)
    pto.tile.load(qk_view, qk_tile)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    qk_ub = qk_tile.as_ptr()
    oldmax_ub = oldmax_tile.as_ptr()
    oldsum_ub = oldsum_tile.as_ptr()
    out_ub = out_tile.as_ptr()
    newmax_ub = newmax_tile.as_ptr()
    newsum_ub = newsum_tile.as_ptr()
    expmax_ub = expmax_tile.as_ptr()
    vreg_f32 = pto.vreg_type(LANES, pto.f32)

    with pto.vecscope():
        active = pto.pset_b32(pto.MaskPattern.ALL)
        one_mask, _ = pto.make_mask(pto.f32, 1)
        for row in range(ROWS):
            oldmax = pto.vlds(oldmax_ub, row, vreg_f32, dist="BRC_B32")
            oldsum = pto.vlds(oldsum_ub, row, vreg_f32, dist="BRC_B32")
            running_max = oldmax
            running_sum = oldsum

            for chunk in range(0, SEQ, LANES):
                valid_cols = pto.const(LANES)
                if chunk != 0:
                    valid_cols = pto.const(SEQ - LANES)
                chunk_mask, _ = pto.make_mask(pto.f32, valid_cols)
                chunk_base = row * COLS + chunk
                values = pto.vlds(qk_ub, chunk_base, vreg_f32)
                chunk_max = pto.vcmax(values, chunk_mask)
                chunk_max_broadcast = pto.vdup(chunk_max, active, pto.PositionMode.LOWEST)
                merged_max = pto.vmax(running_max, chunk_max_broadcast, active)
                scaled_running = pto.vexpdif(running_max, merged_max, active, "ODD")
                running_sum_scaled = pto.vmul(scaled_running, running_sum, active)
                chunk_exp = pto.vexpdif(values, merged_max, chunk_mask, "ODD")
                chunk_sum = pto.vcadd(chunk_exp, chunk_mask)
                chunk_sum_broadcast = pto.vdup(chunk_sum, active, pto.PositionMode.LOWEST)
                running_sum = pto.vadd(running_sum_scaled, chunk_sum_broadcast, active)
                running_max = merged_max

            raw_expmax = pto.vexpdif(oldmax, running_max, active, "ODD")
            scaled_oldsum = pto.vmul(raw_expmax, oldsum, active)
            expmax = pto.vdiv(scaled_oldsum, running_sum, active)
            pto.vsts(running_max, newmax_ub, row, one_mask, dist="1PT_B32")
            pto.vsts(running_sum, newsum_ub, row, one_mask, dist="1PT_B32")
            pto.vsts(expmax, expmax_ub, row, one_mask, dist="1PT_B32")

            for chunk in range(0, SEQ, LANES):
                valid_cols = pto.const(LANES)
                if chunk != 0:
                    valid_cols = pto.const(SEQ - LANES)
                chunk_mask, _ = pto.make_mask(pto.f32, valid_cols)
                chunk_base = row * COLS + chunk
                values = pto.vlds(qk_ub, chunk_base, vreg_f32)
                exp_values = pto.vexpdif(values, running_max, chunk_mask, "ODD")
                output = pto.vdiv(exp_values, running_sum, chunk_mask)
                pto.vsts(output, out_ub, chunk_base, chunk_mask)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(newmax_tile, newmax_view)
    pto.tile.store(newsum_tile, newsum_view)
    pto.tile.store(expmax_tile, expmax_view)
    pto.tile.store(out_tile, out_view)
    pto.pipe_barrier(pto.Pipe.ALL)


def _inputs():
    rng = np.random.default_rng(19)
    oldmax = rng.uniform(-3.0, 1.5, size=(ROWS, 1)).astype(np.float32)
    oldsum = rng.uniform(0.5, 4.0, size=(ROWS, 1)).astype(np.float32)
    qk = rng.normal(0.0, 1.5, size=(ROWS, COLS)).astype(np.float32)
    zeros_state = np.zeros((ROWS, 1), dtype=np.float32)
    zeros_out = np.zeros((ROWS, COLS), dtype=np.float32)
    return [oldmax, oldsum, qk, zeros_state.copy(), zeros_state.copy(), zeros_state.copy(), zeros_out]


def _expected(oldmax, oldsum, qk, *_):
    qk_active = qk[:, :SEQ]
    row_max = np.max(qk_active, axis=1, keepdims=True)
    newmax = np.maximum(row_max, oldmax)
    tmp_active = np.exp(qk_active - newmax, dtype=np.float32)
    cur_sum = np.sum(tmp_active, axis=1, keepdims=True, dtype=np.float32)
    raw_expmax = np.exp(oldmax - newmax, dtype=np.float32)
    newsum = raw_expmax * oldsum + cur_sum
    expmax = (raw_expmax * oldsum) / newsum
    out = np.zeros((ROWS, COLS), dtype=np.float32)
    out[:, :SEQ] = tmp_active / newsum
    return newmax, newsum, expmax, out


def _make_case():
    host_inputs = _inputs()
    expected = _expected(*host_inputs)
    return host_inputs, expected


def _check_case(device_inputs, expected):
    for output_index, golden in zip((3, 4, 5, 6), expected):
        actual = device_inputs[output_index].cpu().numpy()
        assert_close(actual, golden, rtol=EPS, atol=EPS)


CASES = [{
    "name": "softmax_f32_rows24_seq73",
    "kernel": _kernel,
    "make_case": _make_case,
    "check": _check_case,
}]


auto_main(globals())
