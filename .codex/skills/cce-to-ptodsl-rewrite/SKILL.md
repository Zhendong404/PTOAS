---
name: cce-to-ptodsl-rewrite
description: Translate Bisheng CCE / AICORE C++ kernels, macro helpers, and staged pipelines into PTODSL while preserving 1:1 structure, naming, constexpr-derived locals, synchronization, and helper layering. Use when porting files such as `*.cpp` / `*.hpp` that contain `TLOAD`, `TSTORE`, `TMOV`, `TEXTRACT`, `TMATMUL`, `set_flag` / `wait_flag`, custom hook structs, or FlashAttention-style QK/P/PV/GU stages, especially when the user wants structure-first parity instead of an idiomatic PTODSL rewrite.
---

# CCE To PTODSL Rewrite

## Goal

Port Bisheng CCE code to PTODSL in a structure-first way.

Treat compile success as secondary unless the user explicitly asks to make it compile. Preserve:

- function interface shape
- local variable names
- `constexpr`-style derived locals
- `static_assert` intent
- helper / hook layering
- synchronization points
- loop and branch skeleton

## Workflow

### 1. Lock the source of truth

Read only the specific C++ files the user names, plus directly referenced helper files if the user later points to them.

Do not silently “learn” the style from unrelated PTODSL examples. When the user says “only look at file A and file B”, obey that constraint even if another file looks tempting.

### 2. Port in this order

Translate the kernel in layers instead of top-to-bottom prose conversion:

1. global constants, enums, and flags
2. tiny helper structs and hook objects
3. reusable macro helpers such as `pto_macro_matmul`
4. stage functions such as `compute_qk`, `compute_p`, `compute_pv`, `compute_gu`
5. top-level orchestration loops and ping-pong scheduling

For FA-like kernels, preserve the original stage order exactly: QK, P, PV, GU, then the prologue / steady-state / epilogue loops.

### 3. Match names before optimizing

Prefer the original C++ names even when they feel verbose. Keep:

- interface parameter order
- local `Cube_*`, `Tile_*`, `Vec_*`, `kTileFactor`, `sync_iter`, `buf_idx` names
- `logical_block_idx`, `sub_tile_id`, `row_slice`, `accTileEvtID` naming
- helper names such as `PReadyHook`, `Sm2PvFreeHook`, `PreBTExtOpReadyHook`

Translate `static_assert(...)` to Python `assert ...` near the same local context.

### 4. Separate compile-time and runtime control flow

Use Python `if` / `for` only for true compile-time structure.

Use `pto.if_` / `pto.for_` for anything derived from runtime values such as:

- `tile_id`
- `sub_tile_id`
- `logical_block_idx`
- dynamic tile counts
- dynamic synchronization conditions

Do not mix compile-time booleans and runtime values in one Python branch. If C++ has:

```cpp
if constexpr (CAUSAL_MASK) {
    if (s1_index > s0_index) { ... }
}
```

write it as one outer Python `if CAUSAL_MASK:` and one inner `pto.if_(...)`.

### 5. Preserve return semantics carefully

When C++ returns early from inside a runtime branch, do not leave a `pass` in the `else_` path and then keep executing the main path afterward.

Instead:

1. extract the normal path into a local helper such as `emit_qk_main_path()` or `emit_pv_main_path()`
2. keep the skip / early-return logic in the `then_` branch
3. call the normal-path helper only from `else_`

Use this pattern whenever the PTODSL control-flow surface cannot express a direct function-level early return from inside `scf.if`.

### 6. Prefer demo-local helpers before framework changes

If one C++ kernel uses a custom synchronization or hook abstraction, add it locally in the example first.

Examples:

- `TSync_Custom` can live in one demo-local helper file
- hook structs can become tiny dataclasses with `__call__`
- stage-specific macro helpers can live beside the demo

Only extend the PTODSL framework when the abstraction is clearly general-purpose.

### 7. Translate data movement by intent

Use PTODSL surfaces that preserve the original intent:

- `TLOAD` / `TSTORE` -> `pto.tile.load` / `pto.tile.store`
- `TINSERT` -> `pto.tile.insert`
- `TMOV<..., DualModeSplitN>` -> `pto.tile.mov(..., mode="split_n")`
- `TMOV<..., DualModeSplitM>` -> `pto.tile.mov(..., mode="split_m")`
- `wait_intra_block` / `set_intra_block` -> `pto.wait_intra_flag` / `pto.set_intra_flag`

When C++ aliases a subtile with `TASSIGN(tile, addr)`, prefer `pto.alloc_tile(..., addr=...)` in PTODSL.

Important pointer rule: `pto.addptr(ptr, offset)` uses element offsets, not byte offsets.

### 8. Keep macro-call shape visible

When C++ calls a reusable helper like:

```cpp
pto_macro_matmul<Cube_S1, Cube_HEAD, Cube_S0, true>(
    kMatTile, qMatTile, qkAccTile, AccMode::Init, preATExtOpReadyHook, qReadyHook);
```

preserve that call shape in PTODSL, even if the helper body is still a structural approximation.

If the PTODSL stage helper uses a different operand order internally, add one thin adapter lambda or wrapper rather than flattening the macro call at every callsite.

### 9. Validate lightly unless asked for more

Default validation:

- `python3 -m py_compile ...`
- quick scans for unresolved names or broken callsites

Do not stall the translation just because the whole kernel is not compile-clean yet if the user explicitly said structure matters more than compilation.

## Translation Rules

Use these defaults during porting:

- Convert `constexpr`-derived locals into nearby Python locals.
- Keep comments only when they explain non-obvious pipeline intent.
- Keep placeholder `_ = (...)` bundles when needed to preserve structural context without faking full semantics.
- Prefer short PTODSL mode strings such as `split_n` and `split_m` when the wrapper supports them.
- Keep top-level orchestration loops dynamic when the C++ loop bound is runtime-derived.

## Review Checklist

Before finishing a CCE-to-PTODSL port, check:

1. Is the function interface aligned to the C++ source?
2. Are local derived constants and asserts still recognizable?
3. Are runtime loops and branches written with `pto.for_` / `pto.if_`?
4. Are hook names and synchronization points preserved?
5. Are `split_n` / `split_m` TMOV cases still explicit?
6. Are subtile aliases using `addr=` or other PTODSL-native forms instead of ad hoc placeholders?
7. Did any fake `pass` accidentally replace a C++ early return?

## References

Read [references/patterns.md](references/patterns.md) for:

- CCE-to-PTODSL construct mapping
- common FlashAttention porting patterns
- pitfalls discovered while porting `fa_dn.cpp`, `fa_dn_matmul.cpp`, and related helpers
