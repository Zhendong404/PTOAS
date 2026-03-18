---
phase: 01-a5vm-foundation
plan: 03
subsystem: backend
tags: [mlir, a5vm, llvm-ir, ptoas, emitc]
requires:
  - phase: 01-a5vm-foundation
    provides: "A5VM dialect ops, vector type support, and Phase 1 backend fixtures"
provides:
  - "Dedicated A5VM text emitter with unresolved-report support"
  - "Explicit `--pto-backend=a5vm` selection in `ptoas`"
  - "Developer diagnostics for backend IR, intrinsic tracing, and unresolved mappings"
affects: [01-a5vm-foundation, 02-pto-lowering, hivm-emission]
tech-stack:
  added: []
  patterns: ["Direct textual backend emission at the final ptoas boundary", "Explicit unresolved mapping comments plus sidecar reports"]
key-files:
  created: [include/PTO/Transforms/A5VMTextEmitter.h, lib/PTO/Transforms/A5VMTextEmitter.cpp]
  modified: [lib/PTO/Transforms/CMakeLists.txt, tools/ptoas/ptoas.cpp, include/PTO/Transforms/Passes.h, include/PTO/Transforms/Passes.td]
key-decisions:
  - "Keep the EmitC path as the default backend and introduce A5VM through an explicit CLI selector."
  - "Skip PTO pre-backend passes for raw A5VM textual inputs on the A5VM backend so debug IR preserves hardware-facing ops."
  - "Handle unresolved A5VM mappings with explicit comments, diagnostics, and a sidecar report instead of guessing."
patterns-established:
  - "Backend selection belongs in `tools/ptoas/ptoas.cpp`, not in pass registration."
  - "A5VM intrinsic naming is synthesized in the text emitter from op family and vector metadata."
requirements-completed: [BACK-01, BACK-02]
duration: 34min
completed: 2026-03-19
---

# Phase 1 Plan 03: A5VM Backend Boundary Summary

**Explicit `--pto-backend=a5vm` selection with a dedicated LLVM-like A5VM text emitter, intrinsic tracing, and unresolved sidecar reporting**

## Performance

- **Duration:** 34 min
- **Started:** 2026-03-18T17:08:30Z
- **Completed:** 2026-03-18T17:42:45Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Added a standalone A5VM text emitter API and implementation that emits `llvm.hivm`-style calls, unresolved comments, and report files.
- Wired `ptoas` to keep `emitc` as the default backend while routing `--pto-backend=a5vm` through direct textual emission with debug flags.
- Preserved shared `arith` and `scf` constructs in `--a5vm-print-ir` diagnostics while keeping unresolved A5VM mappings explicit and developer-facing.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add the dedicated A5VM textual emitter with unresolved tracking** - `54f8587` (feat)
2. **Task 2: Wire backend selection and developer diagnostics into ptoas** - `778ed00` (feat)

## Files Created/Modified
- `include/PTO/Transforms/A5VMTextEmitter.h` - Declares A5VM emission options and the textual emission entrypoint.
- `lib/PTO/Transforms/A5VMTextEmitter.cpp` - Implements LLVM-like A5VM text emission, intrinsic selection logging, unresolved tracking, and minimal shared-dialect tolerance.
- `lib/PTO/Transforms/CMakeLists.txt` - Builds the new A5VM emitter into `PTOTransforms`.
- `tools/ptoas/ptoas.cpp` - Adds backend selection, A5VM diagnostics flags, raw-A5VM handling, and direct emission at the final backend seam.
- `include/PTO/Transforms/Passes.h` - Documents the backend boundary split between pass-driven EmitC and direct A5VM emission.
- `include/PTO/Transforms/Passes.td` - Keeps pass-registration intent aligned with the new backend path.

## Decisions Made
- Kept `emitc` as the default backend so existing CLI behavior remains stable while A5VM development can proceed behind `--pto-backend=a5vm`.
- Wrote unresolved mappings as both terminal diagnostics and optional sidecar records so backend spelling gaps stay reviewable.
- Treated raw A5VM textual fixtures as already-lowered backend IR on the A5VM path, avoiding PTO pre-backend passes that strip or disturb those ops.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Preserved raw A5VM IR on the A5VM backend path**
- **Found during:** Task 2 (Wire backend selection and developer diagnostics into ptoas)
- **Issue:** Running PTO pre-backend passes on raw A5VM textual inputs removed `a5vm.abs` from the debug IR fixture and broke the intended backend inspection path.
- **Fix:** Skipped the PTO pre-backend pass pipeline when `--pto-backend=a5vm` receives already-lowered A5VM textual IR, then continued through debug printing and textual emission.
- **Files modified:** `tools/ptoas/ptoas.cpp`
- **Verification:** `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase1/a5vm_shared_dialects.mlir -o /dev/null 2>&1 | FileCheck test/phase1/a5vm_shared_dialects.mlir`
- **Committed in:** `778ed00`

**2. [Rule 2 - Missing Critical] Added shared-dialect tolerance to the A5VM text emitter**
- **Found during:** Task 1 (Add the dedicated A5VM textual emitter with unresolved tracking)
- **Issue:** The new A5VM backend needed to coexist with `arith` and `scf` constructs for debug and transitional emission, but the first emitter version aborted on non-A5VM ops.
- **Fix:** Added minimal `arith.addi` lowering plus non-lowered `scf.for` placeholders so the A5VM path can preserve shared-dialect visibility without forcing those ops into A5VM-specific forms.
- **Files modified:** `lib/PTO/Transforms/A5VMTextEmitter.cpp`
- **Verification:** `bash test/phase1/run_phase1_checks.sh` with LLVM `FileCheck` on `PATH`
- **Committed in:** `54f8587`

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both fixes were required to satisfy the Phase 1 backend boundary and debug-IR requirements. No scope creep beyond correctness.

## Issues Encountered
- The workspace PATH did not include `FileCheck`, so verification was run with `/data/mouliangyu/projects/github.com/llvm/llvm-project/build/bin` prepended.
- The full `cmake --build build` path still fails in an unrelated Python module target; plan verification was completed with the focused `pto-opt` target instead.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 1 now has a selectable A5VM backend boundary, textual emission path, and developer diagnostics suitable for expanding real lowering coverage.
- Phase 2 can build on the new emitter and CLI seam without redesigning the existing EmitC path.

## Self-Check: PASSED

- FOUND: `.planning/phases/01-a5vm-foundation/01-a5vm-foundation-03-SUMMARY.md`
- FOUND: `54f8587`
- FOUND: `778ed00`

---
*Phase: 01-a5vm-foundation*
*Completed: 2026-03-19*
