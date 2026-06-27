## ADDED Requirements

### Requirement: PTODSL mixed kernels SHALL not require authored Vector or Cube section markers

The system SHALL accept a PTODSL mixed-kernel entry or kernel module that
contains both Vector-owned and Cube-owned work without requiring the user to
author `@pto.simd`, `@pto.cube`, `pto.section.vector`, or `pto.section.cube`
solely to identify the physical section boundary.

#### Scenario: Frontend verification preserves the logical mixed-kernel surface

- **WHEN** PTODSL frontend verification runs on a mixed-kernel example that uses
  plain logical helpers and uncovered Vector/Cube operations
- **THEN** the verified frontend artifact SHALL still represent one logical
  kernel surface rather than requiring separate user-authored Vector/Cube helper
  kernels

### Requirement: PTOAS SHALL infer and materialize mixed-kernel section ownership

For mixed-kernel IR that does not already contain explicit section wrappers,
PTOAS SHALL infer the ownership of each uncovered top-level segment and
materialize `pto.section.vector` or `pto.section.cube` before the VPTO backend
forms physical compile units. Section inference SHALL classify
frontend-pipe-initialization ops according to their owning core, including AIC
initialization as Cube and AIV initialization as Vector.

#### Scenario: Frontend pipe initialization anchors section ownership

- **WHEN** a mixed-kernel function contains uncovered `aic_initialize` or
  `aiv_initialize` style frontend pipe setup operations
- **THEN** uncovered-section normalization SHALL use those operations to infer
  Cube or Vector ownership instead of reporting an ambiguous section failure

#### Scenario: Existing explicit sections remain valid input

- **WHEN** the input IR already contains valid `pto.section.vector` and
  `pto.section.cube` wrappers
- **THEN** PTOAS SHALL preserve a single canonical sectioned form and SHALL not
  require PTODSL to re-partition the function into separate user-authored
  kernels

### Requirement: VPTO compilation SHALL split materialized sections into physical compile units

After section materialization, the VPTO path SHALL split each mixed kernel into
the physical Vector/Cube compile units required by downstream module-kind
lowering, host stub generation, and fatobj emission.

#### Scenario: Materialized sections produce kernel-kind children

- **WHEN** a logical mixed kernel contains both Vector and Cube sections after
  normalization
- **THEN** the VPTO split pipeline SHALL form separate child modules or
  function variants for the Vector and Cube physical compile units before backend
  emission

### Requirement: Invalid mixed-kernel ownership SHALL fail with strict diagnostics

The system SHALL reject mixed-kernel IR when uncovered ownership cannot be
inferred uniquely or when section boundaries imply illegal cross-section data
flow through ordinary SSA values.

#### Scenario: Ambiguous uncovered segment fails

- **WHEN** an uncovered top-level mixed-kernel segment contains ownership
  signals that do not resolve to exactly one section kind
- **THEN** PTOAS SHALL fail compilation with a diagnostic that the section kind
  cannot be inferred uniquely

#### Scenario: Cross-section SSA dependency fails

- **WHEN** one materialized section depends on an SSA value defined only in the
  opposite section
- **THEN** verification or split-time checking SHALL reject the IR rather than
  silently generating an invalid physical partition
