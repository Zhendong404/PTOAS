---
phase: 01-a5vm-foundation
plan: 01
subsystem: testing
tags: [mlir, filecheck, ptoas, a5vm, bash, fixtures]
requires: []
provides:
  - Corrected Phase 1 fixtures for 256-byte vectors, GM/UB copy primitives, and vlds-vabs-vsts kernel shape
  - A Phase 1 runner that exercises only the corrected fixture set and guards against legacy pseudo-op names
affects: [01-02-PLAN.md, 01-03-PLAN.md, phase-1-validation]
tech-stack:
  added: []
  patterns: [source-first MLIR/FileCheck fixture contracts, standalone bash verification runner]
key-files:
  created:
    - test/phase1/a5vm_copy_gm_to_ubuf_op.mlir
    - test/phase1/a5vm_vabs_kernel_shape.mlir
    - test/phase1/a5vm_copy_ubuf_to_gm_op.mlir
  modified:
    - test/phase1/a5vm_vec_type.mlir
    - test/phase1/a5vm_backend_switch.mlir
    - test/phase1/a5vm_shared_dialects.mlir
    - test/phase1/run_phase1_checks.sh
key-decisions:
  - "Replace the old Phase 1 pseudo-op fixtures instead of extending them, so the repository contract moves cleanly to copy-family and register primitives."
  - "Keep the no-legacy-name regression check in the standalone runner rather than in the MLIR fixtures so file-level validation can forbid obsolete spellings entirely."
patterns-established:
  - "Corrected A5VM fixture contracts name GM/UB copy primitives and vlds-vabs-vsts compute shape, not PTO-shaped pseudo-ops."
  - "Phase runners should print each fixture before invoking ptoas and include an explicit textual regression guard for obsolete surface names."
requirements-completed: [BACK-01, BACK-02, A5VM-01, A5VM-02, A5VM-03, A5VM-04]
duration: 21min
completed: 2026-03-19
---

# Phase 01 Plan 01: Corrected Phase 1 Fixture Contracts Summary

**Corrected Phase 1 A5VM fixtures for GM/UB copy primitives, vlds-vabs-vsts kernel shape, and a guarded direct-runner**

## Performance

- **Duration:** 21 min
- **Started:** 2026-03-18T19:44:15Z
- **Completed:** 2026-03-18T20:05:15Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Replaced the obsolete `a5vm.load` / `a5vm.abs` / `a5vm.store` Phase 1 fixtures with corrected copy-family and register-kernel contracts.
- Rewrote the backend-switch and shared-dialect fixtures so they describe only the corrected A5VM surface.
- Rebuilt the Phase 1 runner around the corrected fixture order and added an explicit legacy-name grep guard.

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite the Phase 1 MLIR/FileCheck fixtures around corrected A5VM primitives** - `c60d690` (feat)
2. **Task 2: Rewrite the Phase 1 runner for the corrected fixture set** - `131ba6e` (feat)

**Plan metadata:** pending docs commit

## Files Created/Modified
- `test/phase1/a5vm_vec_type.mlir` - Keeps the 256-byte vector acceptance and rejection cases while framing the corrected Phase 1 surface.
- `test/phase1/a5vm_copy_gm_to_ubuf_op.mlir` - Defines the metadata-rich GM-to-UB copy primitive fixture with a negative verifier case.
- `test/phase1/a5vm_vabs_kernel_shape.mlir` - Defines the `vlds` / `vabs` / `vsts` compute-kernel contract and a register-shape mismatch case.
- `test/phase1/a5vm_copy_ubuf_to_gm_op.mlir` - Defines the metadata-rich UB-to-GM copy primitive fixture with a wrong-direction negative case.
- `test/phase1/a5vm_backend_switch.mlir` - Pins the corrected backend seam against copy primitives plus register compute ops.
- `test/phase1/a5vm_shared_dialects.mlir` - Proves `func`, `scf`, and `arith` remain visible around corrected A5VM primitives.
- `test/phase1/run_phase1_checks.sh` - Runs the corrected Phase 1 fixture suite in order and guards against legacy pseudo-op names.

## Decisions Made
- Replaced the old fixture set outright instead of layering compatibility aliases on top of the superseded pseudo-op contract.
- Kept the legacy-name regression check in the runner so every Phase 1 MLIR fixture can stay free of obsolete surface strings and still satisfy the plan’s grep-based validation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed obsolete pseudo-op strings from the backend fixture comments**
- **Found during:** Task 1 (Rewrite the Phase 1 MLIR/FileCheck fixtures around corrected A5VM primitives)
- **Issue:** Initial `CHECK-NOT` comments in `a5vm_backend_switch.mlir` still contained the obsolete pseudo-op spellings, which broke the plan’s strict no-legacy-name fixture verification.
- **Fix:** Removed the comment lines and kept the regression guard solely in the runner.
- **Files modified:** `test/phase1/a5vm_backend_switch.mlir`
- **Verification:** Re-ran the Task 1 file-presence and no-legacy-name grep check successfully.
- **Committed in:** `c60d690` (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The auto-fix tightened compliance with the rewritten fixture contract without changing scope.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 1 implementation work can now target only the corrected copy-family and register-kernel fixture contract.
- The runner and backend fixture explicitly reject regression to the old pseudo-op surface.

## Self-Check: PASSED

- Found `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-01-SUMMARY.md`
- Found commit `c60d690`
- Found commit `131ba6e`

---
*Phase: 01-a5vm-foundation*
*Completed: 2026-03-19*
