# vpto-vecscope-inference Specification

## Purpose
TBD - created by archiving change add-vpto-auto-vecscope-inference. Update Purpose after archive.
## Requirements
### Requirement: VPTO backend infers missing vector scopes before LLVM emission
The VPTO backend SHALL infer `pto.vecscope` regions for legal unscoped VPTO vector operation sequences before LLVM/HIVM emission.

#### Scenario: Unscoped vector sequence is wrapped
- **WHEN** a VPTO function contains a contiguous unscoped sequence that produces or consumes `!pto.vreg`, `!pto.mask`, or `!pto.align`
- **AND** every vector-scope value is consumed inside that sequence
- **THEN** the VPTO emission-preparation pipeline wraps the sequence in `pto.vecscope`
- **AND** `pto-validate-vpto-emission-ir` accepts the resulting IR

#### Scenario: Explicit scope remains unchanged
- **WHEN** a VPTO function already contains `pto.vecscope` or `pto.strict_vecscope`
- **THEN** auto inference does not create a nested vector scope inside the existing carrier
- **AND** strict capture semantics are preserved

### Requirement: Inference runs at the final VPTO emission boundary
The VPTO backend SHALL schedule VecScope inference after backend cleanup, pointer normalization, bridge-op expansion, canonicalization, and CSE, and before emission-stage VPTO legality validation.

#### Scenario: Inference sees normalized emission IR
- **WHEN** the VPTO backend prepares a module for `--emit-vpto`, `--vpto-emit-hivm-llvm`, or `--vpto-emit-hivm-bc`
- **THEN** pointer normalization and bridge-op expansion have already run before VecScope inference
- **AND** emission-stage validation runs after VecScope inference

### Requirement: Forbidden operations split inferred scopes
The VecScope inference pass SHALL treat DMA/copy/sync operations, unresolved `func.call`, terminators, existing vector-scope carriers, and VPTO operations forbidden inside vecscope as boundaries.

#### Scenario: DMA boundary splits vector clusters
- **WHEN** an unscoped VPTO function contains vector operations before and after a DMA/copy/sync boundary
- **THEN** the pass creates separate inferred `pto.vecscope` regions around the legal vector clusters
- **AND** the boundary operation remains outside every inferred vecscope

#### Scenario: Uninlined call is not crossed
- **WHEN** an unscoped VPTO function contains a `func.call` between vector operations
- **THEN** the pass does not infer one scope across the call
- **AND** any vector operations that remain inside an unresolved callee are validated independently by the final VPTO legality check

### Requirement: Scalar operations may be included only when safe
The VecScope inference pass SHALL include scalar/address/view operations in an inferred scope only when doing so preserves SSA use-def legality and does not hide values required outside the scope.

#### Scenario: Safe scalar computation stays with vector cluster
- **WHEN** scalar arithmetic or address computation is contiguous with vector operations
- **AND** its results are used only inside the inferred cluster
- **THEN** the pass may include it in the same `pto.vecscope`
- **AND** it must not create additional scalar-only vecscope regions

#### Scenario: Escaping scalar result prevents movement
- **WHEN** a scalar operation near vector operations has a result used outside the candidate cluster
- **THEN** the pass leaves that scalar operation outside the inferred scope or uses it as a cluster boundary
- **AND** the resulting IR remains SSA-valid

### Requirement: Vector-scope values must not escape inferred regions
The VecScope inference pass SHALL reject candidate inferred scopes where `!pto.vreg`, `!pto.mask`, or `!pto.align` results would have users outside the inferred region.

#### Scenario: Escaping vector result fails clearly
- **WHEN** a `!pto.vreg`, `!pto.mask`, or `!pto.align` value produced inside a candidate inferred scope has a user outside that scope
- **THEN** the pass fails before LLVM/HIVM emission
- **AND** the diagnostic identifies that VPTO vector-scope data cannot have external users

### Requirement: Structured control flow is handled conservatively
The VecScope inference pass SHALL handle `scf.if` and `scf.for` without inferring scopes across forbidden nested operations.

#### Scenario: Safe nested control flow is clustered
- **WHEN** an `scf.if` or `scf.for` contains vector operations and no forbidden boundary operations
- **THEN** the pass may treat the control-flow op as one member of an outer inferred `pto.vecscope`

#### Scenario: Nested forbidden operation triggers recursive inference
- **WHEN** an `scf.if` or `scf.for` contains a forbidden boundary operation
- **THEN** the pass does not wrap the whole control-flow op as one outer cluster
- **AND** it recursively infers scopes for legal nested vector clusters where possible

