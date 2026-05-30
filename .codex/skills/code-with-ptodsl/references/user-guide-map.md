# PTODSL User Guide Map

Use this file as a small routing table before opening a full PTODSL user-guide chapter.

## Which chapter answers what

- `02-quick-start.md`
  Best for first-kernel shape, `make_tensor_view`, `alloc_tile`, `partition_view`, `tile.load`, `tile.store`, and a minimal explicit-mode example.

- `03-kernel-entry-and-subkernels.md`
  Best for `@pto.jit` boundary rules, `mode="auto"` vs `mode="explicit"`, and when to use `@pto.simd` / `@pto.simt` / `@pto.cube`.

- `05-control-flow.md`
  Best for deciding between Python control flow and `pto.for_` / `pto.if_`, plus `carry(...)` loops and branch merges.

- `06-scalar-and-pointer-ops.md`
  Best for runtime scalars, pointer math, `scalar.*`, `addptr`, `as_ptr()`, and when a value is still a device-side scalar versus a Python integer.

- `07-data-movement-ops.md`
  Best for `tile.load` / `tile.store`, `mte_*`, vector loads/stores, and the expected abstraction ladder for memory movement.

- `08-compute-operations.md`
  Best for tile ops, vector math, reductions, and choosing a higher-level compute op before hand-assembling a lower-level sequence.

- `09-predicate-and-mask-ops.md`
  Best for `pto.make_mask`, tail predicates, mask types, and mask manipulation.

- `10-sync-ops.md`
  Best for `pipe_barrier`, `set_flag`, `wait_flag`, `mem_bar`, and explicit-mode pipeline ordering.

- `11-flash-attention-walkthrough.md`
  Best for a large realistic explicit-mode kernel that combines tensor views, partitioning, loop-carried state, DMA staging, synchronization, and SIMD sub-kernels.

## Grep shortcuts

Use these when you need to jump to an API quickly:

```bash
rg -n "@pto\\.jit|@pto\\.simd|mode=\"explicit\"|mode=\"auto\"" ptodsl/docs/user_guide
rg -n "pto\\.for_|pto\\.if_|carry\\(|br\\.assign|constexpr" ptodsl/docs/user_guide
rg -n "tile\\.load|tile\\.store|mte_|vlds|vsts" ptodsl/docs/user_guide
rg -n "make_mask|plt_b|pset_b|mask_" ptodsl/docs/user_guide
rg -n "pipe_barrier|set_flag|wait_flag|mem_bar" ptodsl/docs/user_guide
```

## Authoring defaults

If the task does not force a lower-level choice:

1. Start with `@pto.jit(target="a5")` in auto mode.
2. Use `make_tensor_view` + `alloc_tile` + `partition_view`.
3. Use `tile.load` / `tile.store` for GM <-> UB movement.
4. Use Python control flow for compile-time-known structure.
5. Use `pto.for_` / `pto.if_` for runtime control flow.
6. Use `pto.make_mask(...)` for vector tails.
7. Drop to explicit-mode DMA and sync only when the task truly needs staging control.
