---
phase: 01-a5vm-foundation
plan: 03
subsystem: backend
tags: [mlir, a5vm, ptoas, backend-text, phase1]
requires:
  - phase: 01-a5vm-foundation
    provides: "Corrected A5VM dialect ops, vector types, and Phase 1 fixture coverage"
provides:
  - "Raw corrected A5VM backend text emission at the existing ptoas seam"
  - "Explicit unsupported-A5VM diagnostics instead of deferred llvm.hivm fallback"
  - "Backend-switch regression coverage that forbids Phase 3 HIVM text on the Phase 1 path"
affects: [01-a5vm-foundation, 02-pto-lowering, 03-hivm-emission]
tech-stack:
  added: []
  patterns: ["Preserve shared dialect structure by printing raw MLIR at the Phase 1 A5VM seam", "Treat only non-Phase-1 a5vm ops as unsupported at this boundary"]
key-files:
  created: []
  modified: [include/PTO/Transforms/A5VMTextEmitter.h, lib/PTO/Transforms/A5VMTextEmitter.cpp, test/phase1/a5vm_backend_switch.mlir, test/phase1/run_phase1_checks.sh]
key-decisions:
  - "Keep the Phase 1 A5VM backend seam at raw corrected MLIR text and defer llvm.hivm emission to Phase 3."
  - "Detect unsupported A5VM ops by op name while allowing shared dialect ops to stay visible around the backend primitives."
patterns-established:
  - "The Phase 1 `--pto-backend=a5vm` contract prints corrected A5VM primitives verbatim instead of synthesizing later-stage intrinsic text."
  - "Phase runners should explicitly fail if the Phase 1 seam regresses to deferred HIVM output."
requirements-completed: [BACK-01, BACK-02, A5VM-02]
duration: 22min
completed: 2026-03-19
---

# Phase 1 Plan 03: A5VM Backend Boundary Summary

**Raw corrected A5VM backend text at the existing `ptoas` seam with explicit unsupported-op diagnostics and backend-switch regression coverage**

## Performance

- **Duration:** 22 min
- **Started:** 2026-03-18T20:24:00Z
- **Completed:** 2026-03-18T20:46:28Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced the Phase 3-style LLVM-like A5VM emitter with a Phase 1 seam that prints raw corrected MLIR text.
- Kept shared `func`, `scf`, and `arith` structure visible while treating unsupported `a5vm.*` ops as explicit diagnostics or placeholder comments.
- Updated the backend-switch fixture and runner so `--pto-backend=a5vm` now locks to corrected A5VM primitive text instead of deferred `llvm.hivm` output.

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite the thin A5VM text emitter around corrected primitives** - `3688e3d` (feat)
2. **Task 2: Wire the corrected backend selector at the existing emission seam** - `72f64e8` (feat)

## Files Created/Modified
- `include/PTO/Transforms/A5VMTextEmitter.h` - Documents the Phase 1 raw-text seam behavior for the A5VM backend path.
- `lib/PTO/Transforms/A5VMTextEmitter.cpp` - Preserves corrected A5VM MLIR text, validates the supported primitive set, and reports unsupported A5VM ops explicitly.
- `test/phase1/a5vm_backend_switch.mlir` - Checks that the A5VM backend emits corrected primitive text rather than `llvm.hivm` calls.
- `test/phase1/run_phase1_checks.sh` - Guards the backend switch contract and fails if the Phase 1 path regresses to deferred HIVM text.

## Decisions Made
- Kept the A5VM seam Phase 1-thin by printing raw corrected backend text instead of synthesizing later HIVM spellings early.
- Treated only unsupported `a5vm` ops as seam violations so shared dialect structure remains inspectable around backend primitives.
- Locked the regression runner to the raw-text contract because the backend selector is the user-visible seam this plan is responsible for.

## Deviations from Plan

None - plan executed exactly as written against the current workspace. `tools/ptoas/ptoas.cpp` already routed `--pto-backend=a5vm` through the direct A5VM seam, so the implementation work centered on correcting the seam behavior and its contract coverage.

## Issues Encountered
- `FileCheck` was not on `PATH`, so verification used `/data/mouliangyu/projects/github.com/llvm/llvm-project/build/bin/FileCheck`.
- `cmake --build build --target all` still fails in an unrelated Python-link target; focused verification used `cmake --build build --target pto-opt`, which rebuilt the relevant CLI successfully.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 1 now exposes the corrected A5VM backend boundary without leaking Phase 3 HIVM emission behavior into the earlier seam.
- Phase 2 lowering and Phase 3 HIVM emission can now evolve independently against a clear backend-switch contract.

## Self-Check: PASSED

- FOUND: `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-03-SUMMARY.md`
- FOUND: `3688e3d`
- FOUND: `72f64e8`

---
*Phase: 01-a5vm-foundation*
*Completed: 2026-03-19*
