## ADDED Requirements

### Requirement: Shared mainline supports flat and nested single-backend module shapes
The compiler SHALL accept both flat single-module PTO inputs and nested-wrapper
single-backend PTO inputs when they encode the same kernel semantics. Shared
mainline lowering MUST process the compile-unit functions in either shape before
backend-specific lowering begins.

#### Scenario: Nested wrapper reaches shared mainline lowering
- **WHEN** a single-backend PTO input places its kernel `func.func` inside a
  wrapper `builtin.module` instead of directly under the root module
- **THEN** shared mainline lowering processes that function as part of the same
  compile unit instead of skipping it because of container depth

#### Scenario: Flat and nested single-backend inputs are semantically aligned
- **WHEN** flat and nested-wrapper single-backend inputs describe the same
  kernel body and compile options
- **THEN** the compiler applies the same shared mainline lowering stages before
  entering the selected backend pipeline

### Requirement: Shared ModuleOp passes use compile-unit-aware traversal
Any shared-mainline `ModuleOp` pass that rewrites or validates function bodies
MUST operate on compile-unit functions through compile-unit-aware traversal
rather than assuming the relevant functions are direct children of the current
`ModuleOp`.

#### Scenario: Nested function is visited by a shared ModuleOp pass
- **WHEN** a shared-mainline `ModuleOp` pass runs on a supported nested-wrapper
  single-backend input
- **THEN** the pass visits and rewrites the nested compile-unit function body
  using the same per-function logic it applies to flat input

#### Scenario: External declarations are not treated as compile-unit bodies
- **WHEN** compile-unit-aware traversal encounters declarations or imports that
  do not define a function body
- **THEN** shared-mainline body-rewriting passes do not process them as kernel
  definitions

### Requirement: Single-backend VPTO and EmitC preserve supported module-shape behavior
Single-backend VPTO and single-backend EmitC compilation MUST support the flat
and nested-wrapper module shapes accepted by the shared mainline without
requiring frontend flattening or manual user normalization.

#### Scenario: VPTO nested-wrapper input completes shared lowering before tile expansion
- **WHEN** a supported nested-wrapper VPTO input contains operations that rely
  on shared mainline lowering before backend expansion
- **THEN** the backend pipeline receives the already-lowered form instead of
  failing because the shared pass layer skipped the nested function

#### Scenario: EmitC nested-wrapper input compiles through single-backend entry handling
- **WHEN** a supported nested-wrapper EmitC input is compiled through the
  single-backend EmitC path
- **THEN** entry validation, pre-lowering preparation, and output generation
  succeed without requiring the input module to be flattened first

### Requirement: Mixed-backend compile units remain direct-child modules
Mixed-backend compilation SHALL continue to treat the outer container's direct
child modules as backend compile units. The compiler MUST reject nested child
compile units that introduce an extra wrapper level inside a mixed-backend child.

#### Scenario: Current backend-partitioned container remains supported
- **WHEN** a mixed-backend PTO input uses an outer container whose direct child
  modules are the backend compile units
- **THEN** backend selection and child job assembly proceed using those direct
  child modules

#### Scenario: Unsupported nested mixed-backend child is rejected
- **WHEN** a mixed-backend PTO input wraps a backend child compile unit inside
  an additional nested module under the outer container
- **THEN** the compiler emits a clear diagnostic instead of silently treating
  that nested shape as supported mixed-backend input
