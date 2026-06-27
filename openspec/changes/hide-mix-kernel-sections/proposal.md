## Why

PTODSL mixed-kernel authoring still leaks the physical Vector/Cube section
model into user code, even though PTOAS already owns the normalization and
split logic needed by the VPTO backend. The current `check-dsl` failures around
frontend pipe initialization show that the contract for inferred sections is
still incomplete and needs one canonical design.

## What Changes

- Remove the requirement for PTODSL mixed-kernel users to author
  `@pto.simd` / `@pto.cube` decorators or explicit `pto.section.vector` /
  `pto.section.cube` wrappers just to satisfy backend sectioning.
- Define one canonical PTOAS flow that infers uncovered Vector/Cube ownership
  for mixed-kernel IR, including frontend pipe initialization ops, and
  materializes `pto.section.*` before backend-specific split.
- Preserve strict diagnostics for ambiguous or invalid mixed-kernel structure
  instead of weakening verifier behavior.
- Add regression coverage for PTODSL frontend verification, VPTO split/normal
  ization, and `check-dsl`.

## Capabilities

### New Capabilities

- `mixed-kernel-hidden-sections`: PTODSL users author one logical mixed kernel
  without explicit Vector/Cube section markers, while PTOAS infers, materializes,
  and splits the physical sections required by VPTO.

### Modified Capabilities

- None.

## Impact

- Affected code: PTODSL tracing/codegen, PTO section normalization, VPTO split
  pipeline, and frontend verification flow.
- Affected examples/tests: flash-attention cv-split example, PTODSL frontend
  verify tests, VPTO lit coverage, and `check-dsl`.
- Affected behavior: mixed-kernel authoring surface, section inference for
  frontend pipe ops, and the point where physical Vector/Cube compile units are
  formed.
