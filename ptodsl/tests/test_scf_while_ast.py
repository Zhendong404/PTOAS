#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import re
import textwrap

from ptodsl import pto
from ptodsl._ast_rewrite import PTODSLAstRewriteError, _ControlFlowRewriter


def _while_iter_arg_count(mlir_text: str) -> int:
    """Count the loop-carried state slots of the top-level scf.while op."""
    match = re.search(r"scf\.while\s*\([^)]*\)\s*:\s*\(([^)]*)\)", mlir_text)
    assert match is not None, "scf.while signature not found in MLIR"
    return len([part for part in match.group(1).split(",") if part.strip()])


def _rewrite_source(source: str):
    """Rewrite a small source probe without requiring the MLIR runtime."""
    tree = ast.parse(textwrap.dedent(source))
    function_def = tree.body[0]
    function_def.body = _ControlFlowRewriter({}).rewrite_block(
        function_def.body,
        live_after=set(),
    )
    ast.fix_missing_locations(tree)
    return tree


def _is_attr_call(node, owner: str, attribute: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
        and node.func.attr == attribute
    )


def _is_active_guard(node) -> bool:
    if not _is_attr_call(node, "pto", "if_"):
        return False
    return bool(node.args) and isinstance(node.args[0], ast.Name) and "_active_" in node.args[0].id


def _has_constant_add(node, value: int) -> bool:
    return any(
        isinstance(candidate, ast.BinOp)
        and isinstance(candidate.op, ast.Add)
        and any(
            isinstance(descendant, ast.Constant) and descendant.value == value
            for descendant in ast.walk(candidate)
        )
        for candidate in ast.walk(node)
    )


def _has_name_store(node, name: str) -> bool:
    return any(
        isinstance(candidate, ast.Name)
        and isinstance(candidate.ctx, ast.Store)
        and candidate.id == name
        for candidate in ast.walk(node)
    )


@pto.jit(target="a5")
def runtime_while_probe(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def runtime_while_break_continue(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(1, dtype=pto.i32):
            continue
        if value == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(10, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def runtime_while_else(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
    else:
        value = value + pto.const(1, dtype=pto.i32)
    _ = value + pto.const(1, dtype=pto.i32)


@pto.jit(target="a5")
def runtime_for_break_continue(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    for i in range(limit):
        if i == pto.const(1, dtype=pto.i32):
            continue
        if i == pto.const(3, dtype=pto.i32):
            break
        value = value + pto.const(1, dtype=pto.i32)
    else:
        value = value + pto.const(2, dtype=pto.i32)
    _ = value + pto.const(1, dtype=pto.i32)


@pto.jit(target="a5")
def runtime_while_static_break(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        for _ in pto.static_range(2):
            break
        value = value + pto.const(1, dtype=pto.i32)
        if value == pto.const(2, dtype=pto.i32):
            break
    _ = value + pto.const(1, dtype=pto.i32)


# ---------------------------------------------------------------------------
# Issue #1256: _rewrite_while must not carry loop-local temporaries.
#
# Before: every body load was treated as loop-carried state, so a temporary
# written before being read each iteration (e.g. ``col = base + index``) was
# read while still unbound at the pto._while(...) setup, raising
# UnboundLocalError during compile.  After: only names read by the test, read
# before assignment inside the body, or live after the loop are carried.
# ---------------------------------------------------------------------------


@pto.jit(target="a5")
def issue_1256_while_local_temp(limit: pto.i32):
    base = pto.const(2, dtype=pto.i32)
    index = pto.const(0, dtype=pto.i32)
    total = pto.const(0, dtype=pto.i32)
    while index < limit:
        col = base + index
        total = total + col
        index = index + 1
    _ = total


@pto.jit(target="a5")
def issue_1256_while_break_flag(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        value = value + pto.const(1, dtype=pto.i32)
        should_break = value == pto.const(3, dtype=pto.i32)
        if should_break:
            break
    _ = value


@pto.jit(target="a5")
def issue_1256_while_branch_temp(limit: pto.i32):
    low = pto.const(0, dtype=pto.i32)
    high = limit
    while low < high:
        mid = low + pto.const(2, dtype=pto.i32)
        take_upper = mid < pto.const(4, dtype=pto.i32)
        if take_upper:
            low = mid + pto.const(1, dtype=pto.i32)
        else:
            high = mid + pto.const(0, dtype=pto.i32)
    _ = low
    _ = high


@pto.jit(target="a5")
def issue_1256_while_conditional_carry(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        if value < pto.const(2, dtype=pto.i32):
            value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def issue_1256_while_break_flags_only(limit: pto.i32, running: pto.i32, probe: pto.i32):
    # Controlled loop with no user-level carry: the test only reads
    # loop-invariant names and the body store is a loop-local temporary,
    # so the active/did_break control flags provide the loop-carried state.
    while running < limit:
        t = probe
        if t < pto.const(3, dtype=pto.i32):
            break
    _ = running


@pto.jit(target="a5")
def issue_1256_while_continue_else_flags_only(
    limit: pto.i32, running: pto.i32, probe: pto.i32
):
    # The loop test is invariant and all authored stores are loop-local.
    # continue and else therefore rely exclusively on the generated control
    # flags for their loop-carried state.
    while running < limit:
        t = probe
        if t < pto.const(3, dtype=pto.i32):
            continue
    else:
        t = probe + pto.const(1, dtype=pto.i32)
    _ = running


# Issue #1256 exact repros (https://github.com/hw-native-sys/PTOAS/issues/1256).
# Case 2 uses a literal ``while True:`` test; the generated condition must
# materialize the literal as an i1 constant so the break/continue guard
# (``active/did_break``) can combine with it through the runtime ``and`` op.


@pto.jit(target="a5")
def issue_1256_exact_while_local_temp(limit: pto.i32, base: pto.i32):
    index = pto.const(0, dtype=pto.i32)
    total = pto.const(0, dtype=pto.i32)
    while index < limit:
        col = base + index
        total = total + col
        index = index + pto.const(1, dtype=pto.i32)
    _ = total


@pto.jit(target="a5")
def issue_1256_exact_while_true_break(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while True:
        should_break = value >= limit
        if should_break:
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def issue_1256_exact_while_branch_cond(limit: pto.i32, pivot: pto.i32):
    low = pto.const(0, dtype=pto.i32)
    high = limit
    mid = pto.const(0, dtype=pto.i32)
    while low < high:
        mid = (low + high) // pto.const(2, dtype=pto.i32)
        take_upper = mid < pivot
        if take_upper:
            low = mid + pto.const(1, dtype=pto.i32)
        else:
            high = mid
    _ = low


# Issue #1259: a leading runtime break must skip the remaining body in the
# current iteration, not only terminate the following iteration.
@pto.jit(target="a5")
def issue_1259_leading_break(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while True:
        should_break = value >= limit
        if should_break:
            break
        value = value + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def issue_1259_cross_segment_temp(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    while True:
        if value >= limit:
            break
        temp = value + pto.const(1, dtype=pto.i32)
        if temp >= limit:
            break
        value = temp + pto.const(1, dtype=pto.i32)
    _ = value


@pto.jit(target="a5")
def issue_1259_cross_segment_continue(limit: pto.i32):
    value = pto.const(0, dtype=pto.i32)
    one = pto.const(1, dtype=pto.i32)
    two = pto.const(2, dtype=pto.i32)
    while value < limit:
        if value == one:
            continue
        temp = value + one
        if temp == two:
            continue
        value = temp + one
    _ = value


def unsupported_while_subscript(limit: pto.i32):
    values = [0]
    value = pto.const(0, dtype=pto.i32)
    while value < limit:
        values[0] = value
        value = value + pto.const(1, dtype=pto.i32)


def main():
    text = runtime_while_probe.compile().mlir_text()
    assert "scf.while" in text
    assert "scf.condition" in text
    assert "scf.yield" in text

    for fn in (runtime_while_break_continue, runtime_while_else, runtime_for_break_continue,
               runtime_while_static_break):
        loop_text = fn.compile().mlir_text()
        assert "scf.while" in loop_text
        assert "scf.condition" in loop_text
        assert "scf.yield" in loop_text

    # Issue #1256 regressions: loop-local temporaries must not be carried.
    # issue_1256_while_local_temp / break_flag / branch_temp all used to raise
    # UnboundLocalError at the pto._while(...) setup.  Besides compiling, each
    # case also locks the exact number of loop-carried slots of the emitted
    # scf.while, so a future regression that re-adds a loop-local temporary to
    # the carry state (the #1256 defect) fails this contract.
    carry_contract = {
        issue_1256_while_local_temp: 2,               # index, total
        issue_1256_while_break_flag: 3,               # value + control flags
        issue_1256_while_branch_temp: 2,              # low, high
        issue_1256_while_conditional_carry: 1,        # value
        issue_1256_while_break_flags_only: 2,         # active, did_break
        issue_1256_while_continue_else_flags_only: 2, # active, did_break
        issue_1256_exact_while_local_temp: 2,         # index, total
        issue_1256_exact_while_true_break: 3,         # value + control flags
        issue_1256_exact_while_branch_cond: 2,        # low, high
        issue_1259_cross_segment_temp: 3,             # value + control flags
        issue_1259_cross_segment_continue: 3,         # value + control flags
    }
    for fn, expected_carries in carry_contract.items():
        loop_text = fn.compile().mlir_text()
        assert "scf.while" in loop_text
        assert "scf.condition" in loop_text
        actual = _while_iter_arg_count(loop_text)
        assert actual == expected_carries, (
            f"{fn.__name__}: expected {expected_carries} loop-carried slots, got {actual}; "
            "loop-local temporaries must not enter the carry state")

    leading_break_tree = _rewrite_source(
        """
        def leading_break(limit):
            value = pto.const(0, dtype=pto.i32)
            while True:
                if value >= limit:
                    break
                value = value + pto.const(1, dtype=pto.i32)
            return value
        """
    )
    assert any(
        _is_active_guard(node.items[0].context_expr) and _has_constant_add(node, 1)
        for node in ast.walk(leading_break_tree)
        if isinstance(node, ast.With) and node.items
    ), "the body after a leading break must be inside an active guard"

    cross_segment_tree = _rewrite_source(
        """
        def cross_segment_continue(limit):
            value = pto.const(0, dtype=pto.i32)
            one = pto.const(1, dtype=pto.i32)
            two = pto.const(2, dtype=pto.i32)
            while value < limit:
                if value == one:
                    continue
                temp = value + one
                if temp == two:
                    continue
                value = temp + one
            return value
        """
    )
    assert any(
        isinstance(node, ast.With)
        and node.items
        and _is_active_guard(node.items[0].context_expr)
        and _has_name_store(node, "temp")
        and any(
            child is not node
            and isinstance(child, ast.With)
            and child.items
            and _is_active_guard(child.items[0].context_expr)
            and any(
                isinstance(use, ast.Name)
                and isinstance(use.ctx, ast.Load)
                and use.id == "temp"
                for use in ast.walk(child)
            )
            for child in ast.walk(node)
        )
        for node in ast.walk(cross_segment_tree)
    ), "cross-segment temporary must stay dominated by nested active guards"

    with_tree = _rewrite_source(
        """
        def nested_with(limit):
            value = pto.const(0, dtype=pto.i32)
            while value < limit:
                with marker():
                    if value >= limit:
                        break
                    value = value + pto.const(1, dtype=pto.i32)
                value = value + pto.const(2, dtype=pto.i32)
            return value
        """
    )
    marker_with = next(
        node
        for node in ast.walk(with_tree)
        if isinstance(node, ast.With)
        and node.items
        and isinstance(node.items[0].context_expr, ast.Call)
        and isinstance(node.items[0].context_expr.func, ast.Name)
        and node.items[0].context_expr.func.id == "marker"
    )
    assert any(
        isinstance(node, ast.With)
        and node.items
        and _is_attr_call(node.items[0].context_expr, "pto", "if_")
        and _has_constant_add(node, 1)
        for node in marker_with.body
    ), "with-body statements after break must be active-guarded"

    try:
        _rewrite_source(
            """
            def nested_try(limit):
                value = pto.const(0, dtype=pto.i32)
                while value < limit:
                    try:
                        break
                    finally:
                        value = value + pto.const(1, dtype=pto.i32)
                return value
            """
        )
    except PTODSLAstRewriteError as exc:
        assert "inside try blocks" in str(exc)
    else:
        raise AssertionError("break inside try must be diagnosed explicitly")

    def unsupported_break(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            break

    try:
        pto.jit(target="a5")(unsupported_break).compile()
    except Exception as exc:
        assert "control-state lowering" in str(exc)
    else:
        raise AssertionError("runtime break must not be silently traced")

    try:
        pto.jit(target="a5")(unsupported_while_subscript).compile()
    except Exception as exc:
        assert "static subscript carries" in str(exc)
    else:
        raise AssertionError("while subscript carry must be diagnosed")

    # Issue #1256 reverse regression: a loop-local name used after the loop
    # without an outer initialization must still fail to trace (no
    # over-exclusion).  Native Python would report the same unbound name.
    def issue_1256_unbound_live_after(limit: pto.i32):
        value = pto.const(0, dtype=pto.i32)
        while value < limit:
            col = value + pto.const(1, dtype=pto.i32)
            value = col
        _ = col

    try:
        pto.jit(target="a5")(issue_1256_unbound_live_after).compile()
    except UnboundLocalError:
        pass
    else:
        raise AssertionError("unbound loop-local used after while must not compile")


if __name__ == "__main__":
    main()
