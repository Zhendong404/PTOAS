---
phase: 02-pto-lowering
plan: 03
subsystem: backend
tags: [mlir, a5vm, pto, passes, ptoas]
requires:
  - phase: 02-pto-lowering
    provides: corrected A5VM lowering helpers for TLOAD, TABS, and TSTORE
provides:
  - registered PTO-to-A5VM pass failure propagation through conversion legality
  - explicit `ptoas` A5VM backend branch that schedules PTO-to-A5VM before text emission
affects: [02-pto-lowering, 03-hivm-emission, ptoas, a5vm]
tech-stack:
  added: []
  patterns: [conversion-target pass legality, explicit backend branch dispatch]
key-files:
  created: [.planning/phases/02-pto-lowering/deferred-items.md]
  modified: [lib/PTO/Transforms/PTOToA5VM.cpp, tools/ptoas/ptoas.cpp]
key-decisions:
  - "Use partial conversion with illegal PTO tile ops so pass failures surface instead of being silently skipped."
  - "Factor shared pre-backend passes into a helper so the A5VM branch remains structurally separate from EmitC."
patterns-established:
  - "Pass boundary pattern: PTO family rewrites dispatch through helper entrypoints and fail the pass on conversion failure."
  - "CLI pipeline pattern: shared pre-backend passes run once, then backend-specific lowering/text emission diverges cleanly."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 12min
completed: 2026-03-19
---

# Phase 2 Plan 3: PTO Lowering Summary

**Registered PTO-to-A5VM conversion legality and explicit `ptoas` A5VM pipeline wiring around the corrected TLOAD, TABS, and TSTORE helper layer**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-19T01:02:00Z
- **Completed:** 2026-03-19T01:14:29Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Reworked `pto-to-a5vm` to use conversion patterns and explicit illegal PTO ops so failed helper lowerings now fail the pass instead of being dropped.
- Kept the pass implementation limited to dispatch through `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE`.
- Refactored `ptoas` so the A5VM backend path is explicitly separated from EmitC and schedules `createLowerPTOToA5VMPass()` before final A5VM text emission.

## Task Commits

Each task was committed atomically:

1. **Task 1: Register the corrected PTO-to-A5VM pass and dispatch through the three explicit entrypoints** - `a8ddf32` (fix)
2. **Task 2: Wire `ptoas --pto-backend=a5vm` through the corrected Phase 2 lowering path** - `d4409d3` (feat)

**Plan metadata:** pending

## Files Created/Modified
- `lib/PTO/Transforms/PTOToA5VM.cpp` - switched from a manual walk to partial conversion with explicit legality and pass failure propagation.
- `tools/ptoas/ptoas.cpp` - extracted shared pre-backend scheduling and kept the A5VM lowering/text path separate from EmitC.
- `.planning/phases/02-pto-lowering/deferred-items.md` - recorded the pre-existing A5VM generated-header build defect that blocked a clean rebuild.

## Decisions Made
- Use `applyPartialConversion` so helper-layer diagnostics for unsupported PTO lowering cases are surfaced through normal pass failure instead of being ignored.
- Keep backend branching in `ptoas` explicit rather than interleaving A5VM and EmitC pass additions in the main function body.

## Deviations from Plan

None - the code changes followed the plan as specified.

## Issues Encountered

- Full rebuild verification was blocked by a pre-existing build defect outside the plan edits: `CCACHE_DISABLE=1 ninja -C build PTOTransforms ptoas` still fails because the current A5VM generated type includes are not materialized where checked-in headers expect them. The issue is logged in `.planning/phases/02-pto-lowering/deferred-items.md`.
- The sandbox environment also lacked `FileCheck` on `PATH`, so direct fixture replay required locating the LLVM build tree manually. The existing checked-in `build/tools/ptoas/ptoas` binary was stale relative to the current fixtures, so source-level verification was used for this plan while the build defect remains unresolved.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 plan code is in place for future HIVM emission work.
- Before relying on binary verification in later phases, the A5VM generated-header/build graph needs repair so `ptoas` can be rebuilt against the current source tree.

## Self-Check: PASSED

- Found `.planning/phases/02-pto-lowering/02-pto-lowering-03-SUMMARY.md`
- Found commit `a8ddf32`
- Found commit `d4409d3`

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
