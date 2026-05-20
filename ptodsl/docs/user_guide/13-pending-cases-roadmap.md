# 13. Pending Cases Roadmap

This document is a working map of every `ptodsl-doc-pending` marker currently
present in the PTODSL user guide. The goal is not to restate the manual. The
goal is to classify each pending case by root cause and place it in the order
that should be addressed first.

Resolved in this pass:

- `pto.bytewidth(...)` is now exported on the public `pto` surface.
- `pto.elements_per_vreg(...)` is now exported on the public `pto` surface.
- Eager scalar constructors such as `pto.i32(...)`, `pto.f16(...)`, `pto.si32(...)`, and `pto.ui8(...)` are now accepted on the public `pto` surface.
- Tile-level public APIs now live under the `pto.tile.*` namespace, and the documented L1 arithmetic / row-expand helpers are exposed through that surface.
- The type-chapter constructor and `elements_per_vreg(...)` snippets are now covered by docs-as-test fragments, in addition to `test/python/ptodsl_jit_compile.py`.

## Ordering rule

The implementation order below is dependency-driven:

1. Public surface gaps that block many chapters at once.
2. Semantic or verifier mismatches that make documented code fail even when the
   symbol exists.
3. Data-movement and synchronization surfaces that sit below the higher-level
   examples.
4. Docs-as-test coverage gaps that are already close to the implementation
   boundary.
5. Deliberate documentation placeholders that are not meant to become a
   compile-only contract yet.

## 13.1 P0 - Public surface gaps

These are the highest-priority items because they block several chapters and
because the implementation already contains nearby internal helpers or
lower-level primitives.

| Priority | Location | Root cause | Why it belongs here |
|---|---|---|---|
| P0 | `04-type-system-and-buffer.md:32`, `04-type-system-and-buffer.md:55`, `04-type-system-and-buffer.md:64`, `04-type-system-and-buffer.md:88`, `06-scalar-and-pointer-ops.md:323` | Resolved: eager constructor calls such as `pto.i32(...)` / `pto.f16(...)`, plus `pto.elements_per_vreg(...)` and `pto.bytewidth(...)`, are now available on the public surface and covered by regression tests | These were foundational type-chapter and scalar helper contract gaps |
| P0 | `04-type-system-and-buffer.md:97` | Resolved: `vlds(ptr, offset)` now infers `result_vreg_type` from the pointer element type, and `pto.vbitcast(...)` is exposed on the public surface with regression coverage | The documented vector bitcast example now matches the public call surface |
| P0 | `04-type-system-and-buffer.md:118`, `09-predicate-and-mask-ops.md:206` | Resolved: `pto.pbitcast(...)` is exposed on the public surface, with `pto.mask_b8` / `pto.mask_b16` / `pto.mask_b32` type aliases and regression coverage | The documented mask reinterpretation examples now match the public surface |
| P0 | `09-predicate-and-mask-ops.md:65`, `09-predicate-and-mask-ops.md:87` | Resolved: public `MaskPattern` tokens plus `pset_b8/b16/b32`, `pge_b8/b16/b32`, and `make_mask(..., pto.MaskPattern.*)` are now exposed and regression-covered | The documented enum-driven mask-construction surface now matches PTODSL |
| P0 | `09-predicate-and-mask-ops.md:155` | Resolved: mask logical ops (`pand`, `por`, `pxor`, `pnot`, `psel`) are now exposed on the public surface | The documented mask-composition building blocks now compile through PTODSL |
| P0 | `09-predicate-and-mask-ops.md:269` | Resolved: `vcmp` / `vcmps` compare-to-mask surfaces are now exposed on the public surface | Comparison-driven masking is now available through the documented PTODSL interface |
| P0 | `06-scalar-and-pointer-ops.md:213` | Resolved: PTO runtime scalars now use Python native comparison operators directly, and the legacy `scalar.cmpi*` comparison helpers were removed from the public surface | The documented comparison syntax now matches the actual PTODSL contract |
| P0 | `10-sync-ops.md:197`, `10-sync-ops.md:308`, `10-sync-ops.md:329`, `10-sync-ops.md:356`, `10-sync-ops.md:377` | Resolved: `get_buf`, `rls_buf`, `set_cross_core`, `wait_flag_dev`, `set_intra_block`, and `wait_intra_core` are now exposed on the public PTODSL surface with regression coverage | The sync chapter's buffer-sync and pipe-scoped sync facades now compile through PTODSL |
| P0 | `07-data-movement-ops.md:99`, `07-data-movement-ops.md:108`, `07-data-movement-ops.md:119`, `07-data-movement-ops.md:130`, `07-data-movement-ops.md:161`, `07-data-movement-ops.md:169`, `07-data-movement-ops.md:200`, `07-data-movement-ops.md:397`, `07-data-movement-ops.md:784` | Grouped MTE surfaces and alignment-threading surfaces are not exposed on the public surface | These are lower-level DMA ops that the current shorthand layer does not provide |
| P0 | `07-data-movement-ops.md:290` | The 1D `vlds(tile[start:])` form is not supported by the current tile-slice tracing surface | The tile-slice sugar is narrower than the prose example claims |
| P0 | `07-data-movement-ops.md:257` | The ukernel DMA example depends on grouped MTE ops plus `pto.bytewidth(...)`, neither of which is a stable public docs-as-test surface | This is a combined public-surface and contract gap |
| P0 | `12-additional-examples.md:104` | The tile-tail example still relies on mutating `tile.valid_shape` directly, which is not yet a stable public contract | The `pto.tile.*` surface now exists, but this example still depends on unstable tile-metadata mutation |

### P0 implementation sequence

Implement the P0 items in this order:

1. `MaskPattern`-driven creation helpers and mask composition helpers.
2. Native Python scalar comparison operators and any remaining scalar helper surface cleanup.
3. The missing grouped MTE and synchronization aliases.
4. Compare-and-mask and mask reorganization helpers.

## 13.2 P1 - Semantic and verifier mismatches

These items are not just missing exports. The documented code shape does not
match what the current implementation accepts.

| Priority | Location | Root cause | Why it belongs here |
|---|---|---|---|
| P1 | `05-control-flow.md:164` | The documented `pto.if_` / `pto.else_` block form does not match the current implementation surface, which requires explicit result types and `pto.yield_` for merged values | This is a syntax / lowering contract mismatch |
| P1 | `05-control-flow.md:198` | The expression form `pto.if_(cond, then_value, else_value)` is not supported by the current implementation | The docs describe a convenience form that does not exist |
| P1 | `05-control-flow.md:209` | The constexpr tracing example uses native `range(num_blocks)` over runtime tensor metadata, which the current implementation rejects | Trace-time Python control flow is being fed a runtime value |
| P1 | `06-scalar-and-pointer-ops.md:268` | The `addptr` example depends on implementation-specific index-typed offsets to compile | The operator exists, but the example is not yet a stable contract |
| P1 | `09-predicate-and-mask-ops.md:218` | The underlying `pto.punpack` verifier currently accepts only `PredicatePart.LOWER`; the documented `HIGHER` form is not compile-valid yet | This is a semantics gap inside the newly exposed mask-reorganization surface |
| P1 | `07-data-movement-ops.md:1015` | The standalone `@pto.cube` data-movement example is documented, but the sub-kernel form is not yet covered by the current compile-only docs contract | The symbol is real, but the docs-as-test contract is missing |
| P1 | `11-flash-attention-walkthrough.md:149`, `11-flash-attention-walkthrough.md:173` | `alloc_tile(shape=...)` currently requires static physical shapes, but the walkthrough still uses runtime dimensions as physical extents | This is a real implementation constraint, not a docs-only issue |
| P1 | `11-flash-attention-walkthrough.md:521` | Inline L3 context-manager syntax is documented but not implemented yet | The docs describe a future surface |
| P1 | `12-additional-examples.md:156`, `12-additional-examples.md:175` | The cube verifier rejects the authored `mte_l1_l0a` / `mte_l1_l0b` path and the dependent `gemm_tile` path is not compile-valid yet | This is an implementation / verifier gap, not a coverage gap |
| P1 | `12-additional-examples.md:337` | The example depends on unsupported `tnormalize` and unstable scalar-carry idioms | This is a real semantics gap in the implementation |

### P1 implementation sequence

Implement the P1 items after P0:

1. Fix `pto.if_` semantics and remove the unsupported expression form from the
   docs or implement it.
2. Decide whether `addptr` needs a public coercion helper or whether the docs
   should be rewritten to use the current accepted operand types.
3. Close the `alloc_tile(shape=...)` shape-contract mismatch used by the
   flash-attention walkthrough.
4. Decide whether inline L3 context managers are a real feature or only a
   documentation placeholder.
5. Fix the cube verifier gap around the authored GEMM examples.
6. Resolve `tnormalize` and any other scalar-carry primitives before the online
   normalization example is promoted.

## 13.3 P2 - Docs-as-test and fixture coverage gaps

These cases are usually close to the implementation boundary. The symbol often
exists, but the example is not yet supported by the compile-only docs contract
or by a dedicated fixture.

| Priority | Location | Root cause | Why it belongs here |
|---|---|---|---|
| P2 | `01-introduction.md:71`, `02-quick-start.md:140`, `03-kernel-entry-and-subkernels.md:50`, `11-flash-attention-walkthrough.md:37`, `12-additional-examples.md:50`, `12-additional-examples.md:244` | Host-side compile-and-launch or wrapper behavior is outside the current compile-only docs contract | These examples are runtime orchestration, not pure compile fragments |
| P2 | `02-quick-start.md:179` | The layered `vec_add_micro` example is documented, but not supported by the current compile-only docs contract | This is a cross-layer example that needs explicit fixture support |
| P2 | `03-kernel-entry-and-subkernels.md:110`, `03-kernel-entry-and-subkernels.md:163`, `03-kernel-entry-and-subkernels.md:179`, `03-kernel-entry-and-subkernels.md:210`, `03-kernel-entry-and-subkernels.md:226`, `03-kernel-entry-and-subkernels.md:262`, `03-kernel-entry-and-subkernels.md:277`, `03-kernel-entry-and-subkernels.md:311`, `03-kernel-entry-and-subkernels.md:323` | The decorator signatures and typical bodies are documented, but the compile-only docs contract does not yet cover them | These need dedicated fixture-backed coverage |
| P2 | `03-kernel-entry-and-subkernels.md:355`, `03-kernel-entry-and-subkernels.md:365`, `03-kernel-entry-and-subkernels.md:374` | Inline L3 context-manager syntax is documented but not implemented yet | This is a documentation-specified future surface |
| P2 | `12-additional-examples.md:104` | The tail-handling example still depends on direct `tile.valid_shape` mutation and has not been promoted to a stable compile-backed fixture | The `pto.tile.*` helper surface now exists, but this example still needs a contract rewrite |
| P2 | `05-control-flow.md:138`, `09-predicate-and-mask-ops.md:45`, `09-predicate-and-mask-ops.md:361`, `12-additional-examples.md:71`, `12-additional-examples.md:277` | Tail-handling examples now have their common helper (`pto.elements_per_vreg(...)`) available, but still need fixture-backed compile coverage or dependent mask/vector helpers | These have moved out of pure public-surface gap territory and into docs-as-test backlog |
| P2 | `06-scalar-and-pointer-ops.md:110`, `06-scalar-and-pointer-ops.md:370`, `06-scalar-and-pointer-ops.md:391` | Standalone `@pto.simt` examples are documented, but the sub-kernel form is not yet covered by the current compile-only docs contract | This is a fixture / contract gap |
| P2 | `07-data-movement-ops.md:134`, `07-data-movement-ops.md:255` | The ukernel DMA examples need dedicated compile fixture coverage | The doc examples are close, but not yet stable as authored tests |
| P2 | `07-data-movement-ops.md:397`, `07-data-movement-ops.md:784` | Alignment-state examples for vector/predicate load-store helpers are not yet public or not yet covered by fixtures | These are low-level examples that need targeted coverage |
| P2 | `08-compute-operations.md:631` | The standalone `@pto.cube` matmul pattern is documented, but not yet covered by the compile-only docs contract | The pattern is close to the public surface but still needs a fixture |
| P2 | `09-predicate-and-mask-ops.md:87`, `09-predicate-and-mask-ops.md:269` | The doc examples point at mask constructors and compare-to-mask forms that need either a public export or a fixture-backed rewrite | They straddle the line between surface gap and coverage gap |
| P2 | `10-sync-ops.md:134`, `10-sync-ops.md:255` | The ukernel sync examples reference undefined helper kernels / partitions and need dedicated compile fixture coverage | This is a fixture-debt item, not a core API blocker |
| P2 | `11-flash-attention-walkthrough.md:206`, `11-flash-attention-walkthrough.md:278`, `11-flash-attention-walkthrough.md:359`, `11-flash-attention-walkthrough.md:384` | The flash-attention walkthrough still lacks stable compile coverage for the full helper stack and the standalone cube helpers | The chapter is structurally correct, but still fragmented in coverage |
| P2 | `11-flash-attention-walkthrough.md:402` | Signature overview only; detailed behavior is covered by the tested loop/body fragments below | This is a deliberate documentation split, not a missing feature |
| P2 | `12-additional-examples.md:318` | The `mte_load`-based ukernel orchestration example lacks stable compile coverage | The shape is right, but the fixture is not yet stable |

### P2 implementation sequence

Implement the P2 items after the surface and semantics layers are stable:

1. Convert host-side wrappers into separate runtime docs, or keep them pending
   until there is a non-compile-only test harness.
2. Add fixture-backed coverage for the layered decorator examples.
3. Promote the SIMD and SIMT examples once their helper surface is public.
4. Stabilize the flash-attention walkthrough in smaller compile-backed phases.
5. Reclassify examples that are actually documentation-only splits rather than
   pending implementation work.

## 13.4 Read this as a work queue

The recommended way to consume this roadmap is:

1. Clear P0 items first.
2. Clear P1 items next.
3. Use P2 as the regression and fixture backlog.
4. Leave deliberate documentation placeholders marked as pending until the
   implementation plan explicitly moves them.

That order keeps the implementation aligned with the public PTODSL contract
instead of widening the gap between prose and code.
