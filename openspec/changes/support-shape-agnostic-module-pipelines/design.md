## Context

PTOAS currently accepts several module/container forms in practice:
flat single-module inputs, PTODSL nested-wrapper inputs, and backend-partitioned
outer containers whose direct child modules are compile units. The current
failure documented in `docs/designs/ptodsl-nested-module-pass-audit.md` shows
that the shared mainline still has `ModuleOp` passes that enumerate only direct
child `func.func` operations, so nested-wrapper inputs can miss required
lowering before they reach backend-specific passes.

The repo already has one good traversal precedent in
`PTOInstantiateAndInlineOpLib.cpp`, where helper logic collects function-bearing
modules instead of assuming one flat module. This change needs a similar common
model, but only for the parts of the pipeline that are supposed to be shape
agnostic.

## Goals / Non-Goals

**Goals:**
- Make shared mainline behavior equivalent for flat single-module and supported
  nested-wrapper single-backend inputs.
- Formally support nested-wrapper inputs for both single-backend VPTO and
  single-backend EmitC.
- Centralize compile-unit traversal so shared `ModuleOp` passes stop duplicating
  shape assumptions.
- Preserve the existing mixed-backend direct-child compile-unit contract.

**Non-Goals:**
- Do not redesign PTODSL container emission.
- Do not recursively generalize every backend-specific `ModuleOp` pass.
- Do not extend mixed-backend mode to support nested child compile units in this
  change.
- Do not add new public flags or user-facing syntax for module normalization.

## Decisions

### 1. Introduce one shared compile-unit traversal helper

PTOAS will add a common helper in shared transform utilities that enumerates:
- function-bearing compile-unit modules in source order
- function definitions owned by those compile units

This helper will treat wrapper-only modules as traversal containers, not work
units. A module that directly owns one or more `func.func` operations is a
compile unit for shared-pass purposes.

Why this approach:
- It matches how the compiler already reasons about function-bearing nested
  modules in `PTOInstantiateAndInlineOpLib.cpp`.
- It keeps traversal rules consistent across shared `ModuleOp` passes.
- It avoids open-coded `module.walk(func::FuncOp)` usage that can blur compile
  boundaries and accidentally mix imported declarations with the wrong unit.

Alternative rejected:
- Early global normalization of every input shape into one rewritten container
  form before the shared mainline. This would broaden the blast radius, create
  new ownership questions for attributes/symbols, and force backend-partitioned
  and mixed-backend containers into the same refactor.

### 2. Convert only shared-mainline `ModuleOp` passes to the new traversal

The shared mainline passes that currently depend on direct-child
`func.func` enumeration will be updated to use the helper, including:
- `PTOViewToMemref`
- `PlanMemory`
- `PTOResolveReservedBuffers`
- `PTOMaterializeTileHandles`

Why this scope:
- These passes run before backend-specific pipelines and are the actual cause
  of shape-dependent lowering gaps.
- Fixing this layer gives both single-backend VPTO and single-backend EmitC a
  common seam IR contract.

Alternative rejected:
- Repo-wide recursive conversion of all `ModuleOp` passes. That would mix
  independent backend contracts into one large refactor and make review,
  debugging, and rollback harder.

### 3. Keep backend-specific passes shape-specific after their existing boundaries

Backend-specific `ModuleOp` passes will continue to assume their current
normalized/compile-unit input shapes unless they are on the single-backend
entry boundary and must explicitly accept nested-wrapper input.

Concretely:
- VPTO backend passes such as `ExpandTileOp` remain compile-unit oriented.
- EmitC single-backend entry/lowering logic must be updated where it still
  assumes root-level direct-child functions, because nested-wrapper single-
  backend input becomes a supported contract in this change.

Why this split:
- It minimizes backend refactoring while still making supported input shapes
  work end-to-end.
- It keeps the “shape agnostic” promise at the correct boundary: shared
  mainline and single-backend compiler entry, not every internal backend pass.

### 4. Preserve mixed-backend direct-child compile-unit semantics

Mixed-backend mode will continue to require an outer container whose direct
child modules are the backend compile units. Nested child compile units remain
unsupported and must diagnose clearly.

Why this decision:
- The current driver assembles child jobs from direct child modules and relies
  on that structure for symbol copying, peer lookup, and backend selection.
- Extending mixed-backend depth is a separate change with different design
  risks and test needs.

## Risks / Trade-offs

- [Traversal helper accidentally broadens compile scope] → Restrict helper
  semantics to function-bearing compile-unit modules and preserve source order.
- [Shared-pass conversion changes flat-input behavior] → Keep pass-local
  function rewrite order unchanged and add flat-input control regressions.
- [EmitC nested-wrapper support exposes more root-level assumptions] → Audit
  entry validation and EmitC pre-lowering enumeration points together instead
  of fixing only one callsite.
- [Future mixed-backend work misreads this change as full recursive support] →
  Encode the direct-child-only mixed-backend contract in the spec and negative
  tests.
