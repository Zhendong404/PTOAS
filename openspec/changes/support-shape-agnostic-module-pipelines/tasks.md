## 1. Shared traversal foundation

- [x] 1.1 Add a shared compile-unit traversal helper for function-bearing modules in the PTO transform utilities layer.
- [x] 1.2 Document the helper semantics in code comments: wrapper-only modules are containers, direct function-bearing modules are compile units, declarations are not rewrite targets.
- [x] 1.3 Add focused unit or lit coverage that proves the helper enumerates flat and nested-wrapper compile units in stable source order.

## 2. Shared mainline pass conversion

- [x] 2.1 Update `PTOViewToMemref` to use compile-unit-aware traversal without changing its per-function rewrite order.
- [x] 2.2 Update `PlanMemory` and `PTOResolveReservedBuffers` to use the same compile-unit-aware traversal.
- [x] 2.3 Update `PTOMaterializeTileHandles` to restore/helper-materialize nested compile-unit functions through the shared traversal model.
- [x] 2.4 Add regression tests covering flat and nested-wrapper single-backend inputs through the shared mainline.

## 3. Single-backend VPTO and EmitC support

- [x] 3.1 Audit and update single-backend VPTO entry handling so supported nested-wrapper input reaches backend passes only after shared lowering has processed the real function bodies.
- [x] 3.2 Audit and update single-backend EmitC entry validation and pre-lowering logic to support the same nested-wrapper contract.
- [x] 3.3 Add compile-success regressions for nested-wrapper VPTO and nested-wrapper EmitC control cases, plus flat-input non-regression controls.

## 4. Mixed-backend contract and diagnostics

- [x] 4.1 Preserve the direct-child compile-unit contract in mixed-backend driver logic and avoid broadening child job assembly to nested children.
- [x] 4.2 Add an explicit negative test for mixed-backend nested child compile units with a stable diagnostic.
- [x] 4.3 Update user-facing design/spec references so the supported and unsupported container shapes are documented consistently.
