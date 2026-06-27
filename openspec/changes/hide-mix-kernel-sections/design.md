## Context

PTODSL is moving toward a single logical mixed-kernel surface where users do
not explicitly partition code into Vector and Cube sections. The current PTOAS
stack already has the main backend pieces for this model:

- `PTONormalizeUncoveredTileSections` can infer uncovered section ownership and
  materialize `pto.section.vector` / `pto.section.cube`.
- `PTOSplitCVModule` can consume sectioned kernels and form physical
  Vector/Cube compile units.
- VPTO emission, host stub generation, and fatobj linking already assume that
  physical split happens before backend lowering.

What remains unclear is the canonical ownership boundary. Today some behavior is
still expressed through PTODSL surface constructs, while some validation
implicitly expects materialized sections early enough to classify
`aic_initialize` / `aiv_initialize` and similar frontend pipe setup ops.

## Goals / Non-Goals

**Goals:**

- Let PTODSL users author mixed kernels without explicit Vector/Cube section
  decorators.
- Keep one logical kernel surface through PTODSL frontend verification.
- Make PTOAS section inference the single source of truth for uncovered
  mixed-kernel ownership, including frontend pipe initialization ops.
- Ensure the VPTO path still produces separate physical Vector/Cube compile
  units before backend emission.
- Preserve strict failure modes for ambiguous ownership or illegal cross-section
  value flow.

**Non-Goals:**

- Changing EmitC-only kernel behavior.
- Reworking the VPTO backend object emission or host stub ABI model.
- Adding cross-section SSA transport semantics beyond the existing explicit sync
  and data movement model.
- Supporting arbitrary mixed ownership inference for ops that still lack a
  stable section classification rule.

## Decisions

### 1. PTOAS owns hidden-section materialization

The canonical design is that PTODSL emits one logical mixed-kernel function
without requiring `@pto.simd` / `@pto.cube` decoration for ownership. PTOAS
normalization infers uncovered Vector/Cube regions and materializes
`pto.section.vector` / `pto.section.cube` in IR.

Rationale:

- PTODSL should not duplicate backend section classification logic.
- The same inferred-section model then applies to PTODSL-generated IR and any
  other PTO input using the same logical mixed-kernel form.
- It keeps the user-facing surface independent of physical VPTO partitioning.

Alternatives considered:

- Infer and wrap sections in PTODSL frontend only. Rejected because it creates a
  second source of truth and still leaves non-PTODSL PTO input needing the same
  PTOAS logic.
- Delay or weaken verifier behavior until later passes. Rejected because it
  hides real ownership errors and makes diagnostics phase-dependent.

### 2. Section inference must recognize frontend pipe initialization

`AicInitializePipeOp` and `AivInitializePipeOp` are treated as tile-like
ownership signals during uncovered-section normalization and module-kind
inspection. AIC init contributes Cube ownership; AIV init contributes Vector
ownership.

Rationale:

- These ops are often the earliest unambiguous indicators of which physical core
  owns a region.
- Without them, section inference can fail on otherwise valid mixed-kernel IR
  and surface as `check-dsl` regressions.

Alternative considered:

- Keep init ops ownership-neutral and rely on later ops. Rejected because the
  section boundary then becomes brittle and order-dependent.

### 3. Physical split still happens in PTOAS, after section materialization

Once `pto.section.vector` / `pto.section.cube` exist in the logical kernel,
`PTOSplitCVModule` remains responsible for cloning/splitting that function into
separate physical Vector/Cube compile units.

Rationale:

- The backend already expects module-level `pto.kernel_kind` ownership after the
  split.
- This keeps host stub emission and VPTO LLVM lowering unchanged.
- It matches the intended model that section markers are an internal IR form,
  not a user contract.

Alternative considered:

- Make PTODSL emit separate entry helpers directly. Rejected because it leaks
  the physical compile shape into the frontend surface and fights the single
  logical kernel model.

### 4. Frontend verification should validate the hidden-section flow

The PTODSL frontend verification path should preserve one logical entry kernel
while still allowing PTOAS normalization to materialize inferred sections and
prove that downstream VPTO split is possible.

Rationale:

- This is the closest regression signal to the user-facing contract.
- It prevents the example/test suite from drifting back toward explicit
  decorator-authored sectioning.

## Risks / Trade-offs

- [Inference gaps for niche ops] -> Keep the classifier conservative and fail
  fast when ownership cannot be inferred uniquely; extend coverage only with
  explicit rules.
- [Pass placement regressions] -> Add pipeline tests covering idempotence and
  the exact phase where section materialization happens before split/lowering.
- [Frontend/backend contract drift] -> Lock the flash-attention cv-split example
  and `check-dsl` into regression coverage.
- [Strict diagnostics may reject previously tolerated IR] -> Prefer explicit
  failures over silently producing a wrong core partition, and document the
  unsupported patterns in tests.

## Migration Plan

1. Update the mixed-kernel PTODSL example/tests to remove explicit user-authored
   section decorators from the supported surface.
2. Extend PTOAS normalization so frontend pipe initialization participates in
   ownership inference.
3. Verify the existing VPTO split pipeline still cleanly converts materialized
   sections into physical Vector/Cube child modules.
4. Land regression coverage before relying on the new authoring model as the
   default recommendation.

## Open Questions

- Whether any remaining PTODSL helper outlining path still hard-depends on
  explicit `pto.ptodsl.subkernel_helper` ownership markers for mixed kernels.
- Whether uncovered-section normalization should run in exactly one shared
  pipeline location or be invoked in both frontend-verify and full compile
  entrypoints through a shared helper.
