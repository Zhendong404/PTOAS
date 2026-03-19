---
phase: 02-pto-lowering
plan: 01
subsystem: testing
tags: [mlir, a5vm, pto, filecheck, scf]
requires:
  - phase: 01-a5vm-foundation
    provides: corrected A5VM primitive names and backend selector wiring
provides:
  - explicit AIV loop-scope contract fields on the public Phase 2 lowering surface
  - vec-scope fixture checks that require cce_aiv_loop_hint and llvm.loop.aivector_scope
  - Phase 2 runner guards against stale pseudo-op checks and bare scf.for nesting
affects: [02-02, 02-03, hivm-emission]
tech-stack:
  added: []
  patterns: [public lowering contracts carry loop-scope metadata explicitly, fixture runners reject obsolete contract shapes]
key-files:
  created: []
  modified:
    - include/PTO/Transforms/A5VMLowering.h
    - test/phase2/tabs_abs_loop_shape.mlir
    - test/phase2/run_phase2_checks.sh
key-decisions:
  - "Expose AIV vec-scope ownership as an explicit unary lowering contract instead of implying it from plain scf.for nesting."
  - "Keep Wave 1 enforcement in the committed runner so stale pseudo-op checks and bare loop-only fixtures fail fast."
patterns-established:
  - "Phase 2 unary lowering contracts must name both sourceAttr and loweredAttr for vec-scope metadata."
  - "Fixture suites can enforce semantic contract corrections through preflight rg guards before executing FileCheck."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 8min
completed: 2026-03-19
---

# Phase 2 Plan 1: PTO Lowering Summary

**Public Phase 2 contracts now carry explicit AIV vec-scope metadata and the committed `TABS` fixture suite rejects stale loop-only semantics**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-19T02:31:00Z
- **Completed:** 2026-03-19T02:39:33Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `A5VMLoopScopeKind` and `A5VMLoopScopeContract` to the public lowering surface and attached loop-scope carriage to unary lowering contracts.
- Rewrote the `tabs_abs_loop_shape` fixture so the chosen loop must name `cce_aiv_loop_hint` and `llvm.loop.aivector_scope` alongside `a5vm.vlds`, `a5vm.vabs`, and `a5vm.vsts`.
- Hardened the Phase 2 runner to reject legacy `a5vm.load/store/abs` references and bare `scf.for`-only vec-scope checks.

## Task Commits

Each task was committed atomically:

1. **Task 1: Define the Phase 2 contract surface with an explicit AIV loop carrier** - `b8563d9` (feat)
2. **Task 2: Rewrite the Wave 1 fixtures and runner around explicit AIV loop ownership** - `cbad0b9` (test)

## Files Created/Modified
- `include/PTO/Transforms/A5VMLowering.h` - adds the explicit loop-scope carrier and metadata attachment helper declaration to the public Phase 2 lowering contract.
- `test/phase2/tabs_abs_loop_shape.mlir` - updates the unary vec-scope fixture to require explicit AIV carrier strings and ordered vector primitive checks.
- `test/phase2/run_phase2_checks.sh` - adds preflight guards that reject stale pseudo-op references and bare `scf.for` vec-scope assertions.

## Decisions Made

- Expose AIV vec-scope ownership as explicit contract data so later lowering work cannot silently treat `__VEC_SCOPE__` as plain structural loop nesting.
- Enforce the corrected fixture semantics in the standalone runner as well as the MLIR file so regressions fail before `FileCheck` execution.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `gsd-tools init execute-phase` rejected the env-var style invocation from the executor instructions and required the positional form `init execute-phase 02-pto-lowering`; execution continued after rerunning the command.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 helper implementation can now build against an explicit unary loop-scope contract instead of inferring vec-scope semantics from loop nesting.
- The committed runner will fail fast if follow-up implementation work reintroduces obsolete pseudo-op names or weak vec-scope fixture checks.

## Self-Check

PASSED

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
