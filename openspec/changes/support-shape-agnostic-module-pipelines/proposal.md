## Why

PTODSL and hand-authored PTO inputs now exercise more than one valid module
container shape, but parts of PTOAS still assume that relevant `func.func`
definitions are direct children of the current `ModuleOp`. This mismatch causes
shape-dependent compile failures and makes single-backend nested-wrapper inputs
behave differently from equivalent flat inputs.

## What Changes

- Define a compiler contract for shape-agnostic single-backend PTO module
  inputs: flat single-module inputs and nested-wrapper inputs SHALL compile
  through the same shared mainline semantics.
- Update shared mainline `ModuleOp` passes to process compile-unit functions
  through a common traversal model instead of relying on direct-child
  `func.func` enumeration.
- Make single-backend VPTO and EmitC pipelines accept supported wrapper-module
  inputs without requiring PTODSL or users to flatten containers first.
- Preserve the current mixed-backend direct-child compile-unit contract and add
  explicit diagnostics/tests for unsupported nested child compile units.
- Add regression coverage for flat, nested single-backend, backend-partitioned,
  and unsupported mixed-backend nested-child cases.

## Capabilities

### New Capabilities
- `pto-module-shape-compatibility`: Defines the accepted PTO module container
  shapes and the required compiler behavior for shared mainline, single-backend
  VPTO, single-backend EmitC, and mixed-backend boundary handling.

### Modified Capabilities

## Impact

- Affected code: shared PTOAS pass traversal, single-backend VPTO/EmitC entry
  handling, and mixed-backend diagnostics in `tools/ptoas` and `lib/PTO`.
- No new CLI flags or PTODSL surface APIs.
- Adds new compiler contract coverage in OpenSpec and regression tests.
