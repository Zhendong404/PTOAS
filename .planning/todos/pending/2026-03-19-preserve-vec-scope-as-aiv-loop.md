---
created: 2026-03-19T02:20:28.027Z
title: Preserve VEC scope as AIV loop
area: planning
files:
  - /usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp:21
  - /usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h:472
  - /data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp
  - /data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/A5VMTextEmitter.cpp
---

## Problem

Phase 2 currently models the PTO A5 `__VEC_SCOPE__` structure for `TABS` as ordinary software `scf.for` nesting. That loses the actual Bisheng/CCE meaning of `__VEC_SCOPE__`, which the user confirmed is `__attribute__((cce_aiv_loop_hint))` and lowers to `llvm.loop.aivector_scope` metadata.

Because this semantic marker is lost, the current A5VM path does not preserve the distinction between:

- a normal software loop used to iterate tiles or vector chunks
- the loop that establishes AIV vector-scope semantics for vector register ops like `vlds`, `vabs`, and `vsts`

This affects both Phase 2 PTO lowering fidelity and Phase 3 textual HIVM emission, because the backend currently has no explicit place to carry the AIV loop hint through to near-final LLVM-like text.

## Solution

Revise the current Phase 2/3 contract so the loop corresponding to `__VEC_SCOPE__` is represented explicitly instead of being recoverable only from plain `scf.for`.

Preferred direction:

- introduce an explicit A5VM loop form or equivalent loop marker for AIV scope semantics
- keep the real software loop structure inside that scope, rather than replacing all loops with a synthetic region
- teach the textual HIVM emitter to lower that explicit representation into an LLVM-like loop form with `llvm.loop.aivector_scope` metadata or a clear metadata placeholder/comment when final assembly is still textual-only

Acceptance target:

- `TABS` lowering still mirrors PTO A5 control structure
- `__VEC_SCOPE__` is no longer silently degraded to a plain loop
- Phase 3 emitted text has a stable hook for AIV loop metadata emission
