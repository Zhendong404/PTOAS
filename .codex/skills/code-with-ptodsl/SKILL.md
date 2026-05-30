---
name: code-with-ptodsl
description: Write or revise PTODSL kernels, sub-kernels, examples, or docs-facing snippets using the current PTODSL user guide. Use when generating PTODSL code, translating kernels into PTODSL, choosing between auto and explicit mode, authoring `@pto.jit` / `@pto.simd` / `@pto.simt`, handling masks and control flow, or deciding where synchronization is required.
---

# Code With PTODSL

Use this skill when the task is to write, edit, review, or translate PTODSL code. The goal is not just to emit valid syntax, but to produce PTODSL that matches the current public surface and the team's preferred authoring style.

## First Principles

Treat the PTODSL user guide as the source of truth for public authoring style and API shape. Start from the simplest surface that fits the task, and only drop to lower-level operations when the required behavior is not expressible cleanly at the higher level.

Default style rules:

1. Prefer plain Python literals over `pto.const(...)` unless an API specifically needs a PTO value or explicit dtype materialization.
2. If a condition or loop bound is known at compile time, prefer native Python `if` / `else` / `for` instead of `pto.if_` / `pto.for_`.
3. Prefer concise string forms for enum-like op arguments when the string is already unambiguous.
4. Remember that load/store and vector compute need synchronization boundaries. In `mode="explicit"`, author them manually.
5. `@pto.jit` is the only host-visible kernel entry. `@pto.simd`, `@pto.simt`, and `@pto.cube` are sub-kernels only.

## Workflow

### 1. Choose the right programming model

Pick the highest-level surface that can express the requested kernel:

- Use `@pto.jit(target="a5")` with default `mode="auto"` for tile-first kernels that can be written with `make_tensor_view`, `alloc_tile`, `partition_view`, `pto.tile.load`, `pto.tile.store`, and tile compute ops.
- Use `@pto.jit(target="a5", mode="explicit")` only when the user needs micro-instruction control such as direct `mte_*`, pointer sequencing, explicit pipeline staging, or manual synchronization.
- Use `@pto.simd` for vector-register computation on tiles or tile slices.
- Use `@pto.simt` for scalar-path device code and scalar-heavy control logic.
- Use `@pto.cube` only for cube-unit matrix work.

Bias toward `mode="auto"` unless the task explicitly needs instruction-level orchestration.

### 2. Author the kernel boundary correctly

At the `@pto.jit` boundary, only use these parameter kinds:

- Device buffers as positional `pto.ptr(dtype, "gm")`
- Runtime scalars as positional PTO scalar annotations such as `pto.i32`, `pto.f32`, `pto.i1`
- Compile-time knobs as keyword-only `pto.constexpr`

Do not expose `Tile`, `PartitionTensorView`, `TensorView`, `VReg`, or legacy host tensor annotations at the host-visible entry.

Inside the kernel, reconstruct GM descriptors explicitly with `pto.make_tensor_view(...)`.

### 3. Decide which control flow surface to use

Use this rule consistently:

- Python `for` / `if` means trace-time control flow.
- `pto.for_` / `pto.if_` means device-side runtime control flow.

Practical guidance:

- If a bound is a `pto.constexpr` or other compile-time-known Python value, use native Python control flow.
- If a bound or condition depends on runtime tensor metadata, block indices, loaded scalars, or other device values, use `pto.for_` or `pto.if_`.
- For loop-carried state such as online softmax accumulators or chunked tail tracking, use `pto.for_(...).carry(...)`.
- For data-dependent branch results that flow outward, use `with pto.if_(cond) as br:` and merge with `br.assign(...)`.

Avoid writing Python `range(dynamic_pto_value)` or Python `if dynamic_pto_value:` in kernels.

### 4. Prefer the idiomatic data movement surface

Start with the highest-level memory movement API that fits:

- In `mode="auto"`, prefer `pto.tile.load` and `pto.tile.store`.
- In `mode="explicit"`, use `pto.mte_*` only when you truly need DMA-level scheduling.
- Within SIMD code, use `pto.vlds(...)` / `pto.vsts(...)` on tile slices or typed pointers for vector-width accesses.

Prefer shape-driven tile code over pointer arithmetic when both are possible. Drop to `as_ptr()`, `addptr`, and explicit offsets only when the requested behavior is inherently pointer-oriented.

### 5. Handle vector tails with masks, not ad hoc branching

For chunked SIMD loops:

- Get vector width with `pto.elements_per_vreg(dtype)`.
- Use a `pto.for_(0, cols, step=VEC).carry(remained=cols)` loop.
- Build the tail predicate with `mask, remained = pto.make_mask(dtype, remained)`.
- Gate vector ops with that mask and write back the updated remainder via `loop.update(...)`.

Prefer `pto.make_mask(...)` over lower-level `plt_b*` / `pset_b*` unless the task specifically needs manual mask construction.

### 6. Synchronize explicitly when you author explicit-mode staging

In `mode="auto"`, assume the tile-first abstraction is the default and do not manually sprinkle explicit synchronization unless the code already requires it.

In `mode="explicit"`:

- Assume DMA, vector, and cube pipelines are independent.
- Insert synchronization between producer and consumer stages.
- For simple full-fence staging, `pto.pipe_barrier(pto.Pipe.ALL)` is acceptable.
- For pipeline-pair sequencing, use matching `pto.set_flag(...)` and `pto.wait_flag(...)`.
- Use `pto.mem_bar(...)` only for same-pipeline memory ordering hazards such as aliasing vector load/store ordering.

Minimum mental model:

- GM/UB load completes before vector compute uses the tile.
- Vector or cube compute completes before DMA stores the result.
- If you mix load/store ops with vector ops in explicit mode, assume you need an ordering edge and then verify the exact primitive.

### 7. Prefer concise public-surface spellings

When PTODSL accepts both enum objects and simpler public spellings:

- Prefer the simpler string/token form when it is already clear.
- Keep the enum object form when the string would hide meaning or when the API requires a concrete enum value.

Also prefer surface inference when PTODSL already provides it:

- Let `vlds(tile[row, col:])` infer the vreg type.
- Let `make_tensor_view(...)` infer the view rank/type from the pointer and shape.
- Let `partition_view(...)` infer the partition type from the source view.

Do not add explicit type noise unless inference would be ambiguous.

## Default Code Shape

For most new kernels, start from this structure and then specialize:

```python
from ptodsl import pto, scalar


@pto.simd
def row_kernel(src_tile: pto.Tile, dst_tile: pto.Tile, cols: pto.i32):
    VEC = pto.elements_per_vreg(pto.f32)
    col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
    with col_loop:
        c = col_loop.iv
        remained = col_loop.remained
        mask, remained = pto.make_mask(pto.f32, remained)
        vec = pto.vlds(src_tile[0, c:])
        # ... vector math under mask ...
        pto.vsts(vec, dst_tile[0, c:], mask)
        col_loop.update(remained=remained)


@pto.jit(target="a5")
def kernel(
    X_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.constexpr = 128,
):
    x_view = pto.make_tensor_view(X_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])

    x_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    with pto.for_(0, rows, step=1) as r:
        x_part = pto.partition_view(x_view, offsets=[r, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[r, 0], sizes=[1, cols])
        pto.tile.load(x_part, x_tile)
        row_kernel(x_tile, o_tile, cols)
        pto.tile.store(o_tile, o_part)
```

Adapt this template instead of starting from raw pointers or explicit-mode DMA unless the task requires that lower level.

## Review Checklist

When writing or reviewing PTODSL code, check these first:

1. Is `@pto.jit` the only kernel entry, with sub-kernels called from inside it?
2. Are `constexpr` values used for compile-time structure and Python-native control flow?
3. Are runtime-dependent loops/branches written with `pto.for_` / `pto.if_`?
4. Is the code using `tile.load` / `tile.store` by default, only dropping to `mte_*` when necessary?
5. Are vector tails handled with masks instead of partial-width special cases?
6. If the kernel is explicit-mode, are DMA/compute ordering edges clearly synchronized?
7. Is the code using public PTODSL spellings and inference instead of unnecessary low-level verbosity?

## When To Read References

Read the quick map in [references/user-guide-map.md](references/user-guide-map.md) when you need to confirm a public API family or choose the right chapter to inspect more closely.

If you are blocked on a concrete API detail, open only the relevant user-guide chapter instead of loading everything:

- Entry and sub-kernels: `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`
- Control flow: `ptodsl/docs/user_guide/05-control-flow.md`
- Scalars and pointers: `ptodsl/docs/user_guide/06-scalar-and-pointer-ops.md`
- Data movement: `ptodsl/docs/user_guide/07-data-movement-ops.md`
- Compute ops: `ptodsl/docs/user_guide/08-compute-operations.md`
- Masks: `ptodsl/docs/user_guide/09-predicate-and-mask-ops.md`
- Synchronization: `ptodsl/docs/user_guide/10-sync-ops.md`
- End-to-end explicit orchestration example: `ptodsl/docs/user_guide/11-flash-attention-walkthrough.md`

## Anti-Patterns

Avoid these common PTODSL mistakes:

- Using `pto.const(...)` for ordinary compile-time literals that can stay plain Python.
- Using Python `for` / `if` with runtime PTO values.
- Passing host-invisible PTODSL objects directly as `@pto.jit` arguments.
- Starting in `mode="explicit"` for kernels that are naturally tile-first.
- Writing manual mask bit-manipulation when `pto.make_mask(...)` already matches the tail pattern.
- Mixing explicit DMA and vector compute without an obvious synchronization edge.
- Over-specifying result types or enum objects when the public API already infers them.
