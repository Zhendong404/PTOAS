---
phase: 02-pto-lowering
plan: 03
subsystem: backend
tags: [mlir, pto, a5vm, lowering, compiler]
requires:
  - phase: 02-01
    provides: "Phase 2 lowering fixtures and verification contracts for TLOAD, TABS, and TSTORE"
  - phase: 02-02
    provides: "Reusable PTO-to-A5VM lowering contracts and helper entrypoints"
provides:
  - "Registered pto-to-a5vm module pass"
  - "Lowering of PTO TLOAD, TABS, and vec TSTORE into A5VM ops with explicit metadata"
  - "ptoas A5VM backend wiring that runs PTO-to-A5VM lowering before textual emission"
affects: [03-hivm-emission, phase2-verification, a5vm-backend]
tech-stack:
  added: []
  patterns: [module pass lowering, bound-tile metadata backtracking, backend-specific final lowering]
key-files:
  created: []
  modified:
    - include/PTO/Transforms/Passes.h
    - include/PTO/Transforms/Passes.td
    - lib/PTO/Transforms/PTOToA5VM.cpp
    - tools/ptoas/ptoas.cpp
    - test/phase2/tload_contract_trace.mlir
    - test/phase2/tstore_branch_shape.mlir
key-decisions:
  - "Run PTO-to-A5VM only on the --pto-backend=a5vm branch after the shared pre-backend passes."
  - "Extract tile layout, valid dims, and address-space metadata from bind_tile and pointer_cast SSA chains because the A5VM boundary sees memref-backed tile values."
  - "Use an explicit rewrite walk instead of greedy pattern application so single-op Phase 2 fixtures retain visible a5vm.load and a5vm.abs ops in debug IR."
patterns-established:
  - "A5VM lowering helpers accept either original TileBufType operands or memref-plus-bind_tile forms."
  - "ptoas raw-A5VM detection must ignore comment lines so textual fixtures do not skip the lowering pipeline accidentally."
requirements-completed: [PTO-01, PTO-02, PTO-03, PTO-04]
duration: 24min
completed: 2026-03-19
---

# Phase 2 Plan 03: PTO-to-A5VM Lowering Summary

**Registered PTO-to-A5VM lowering with explicit TLOAD/TABS/TSTORE metadata and wired `ptoas --pto-backend=a5vm` through the new backend pass**

## Performance

- **Duration:** 24 min
- **Started:** 2026-03-18T18:13:14Z
- **Completed:** 2026-03-18T18:37:40Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Registered the `pto-to-a5vm` module pass and exposed `createLowerPTOToA5VMPass()` through the public pass surface.
- Implemented PTO lowering for `TLOAD`, `TABS`, and vec-path `TSTORE`, including the Phase 2 metadata attributes and explicit ACC/MAT TODO diagnostics.
- Wired the A5VM backend branch in `ptoas` to run PTO-to-A5VM lowering before textual emission while preserving the existing EmitC path as the default backend.

## Task Commits

Each task was committed atomically:

1. **Task 1: Register and implement the PTO-to-A5VM lowering pass** - `5212d32` (feat)
2. **Task 2: Wire the A5VM backend branch to run PTO-to-A5VM lowering** - `0755afd` (feat)

## Files Created/Modified

- `include/PTO/Transforms/Passes.h` - Declares the public PTO-to-A5VM pass factory.
- `include/PTO/Transforms/Passes.td` - Registers the `pto-to-a5vm` `ModuleOp` pass and its dependent dialects.
- `lib/PTO/Transforms/PTOToA5VM.cpp` - Implements the pass, memref-aware metadata extraction, and lowering for TLOAD/TABS/vec TSTORE.
- `tools/ptoas/ptoas.cpp` - Runs PTO-to-A5VM only on the `a5vm` backend path and ignores comment text when detecting raw A5VM input.
- `test/phase2/tload_contract_trace.mlir` - Uses parser-accepted pointer-backed tensor view syntax for the Phase 2 TLOAD fixture.
- `test/phase2/tstore_branch_shape.mlir` - Uses parser-accepted pointer-backed tensor view syntax for the Phase 2 TSTORE fixture.

## Decisions Made

- Kept the EmitC backend unchanged and inserted PTO-to-A5VM only in the `a5vm` backend branch to preserve the Phase 1 backend split.
- Backtracked tile metadata from `bind_tile`, `pointer_cast`, and memref view ops instead of assuming `TileBufType` operands survive to the final lowering boundary.
- Preserved unsupported ACC and MAT TSTORE branches as visible TODO diagnostics rather than forcing speculative lowering behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed comment-sensitive raw A5VM detection in `ptoas`**
- **Found during:** Task 2 (Wire the A5VM backend branch to run PTO-to-A5VM lowering)
- **Issue:** `containsA5VMIR()` treated `// RUN:` and `// CHECK:` comments containing `a5vm.` as already-lowered backend IR, which skipped the new lowering pass on the Phase 2 fixtures.
- **Fix:** Changed raw-A5VM detection to scan only non-comment lines before deciding to bypass the shared PTO pipeline.
- **Files modified:** `tools/ptoas/ptoas.cpp`
- **Verification:** `--a5vm-print-ir` on `test/phase2/unary_template_shape.mlir` and `test/phase2/tload_contract_trace.mlir` now shows `a5vm.abs` and `a5vm.load` instead of raw PTO ops.
- **Committed in:** `0755afd` (part of task commit)

**2. [Rule 3 - Blocking] Repaired Phase 2 fixture syntax to match the repo's current PTO parser**
- **Found during:** Task 2 (Wire the A5VM backend branch to run PTO-to-A5VM lowering)
- **Issue:** The committed TLOAD/TSTORE fixtures used `pto.make_tensor_view` syntax that omitted the required comma before `strides` and passed memrefs where the custom parser currently expects PTO pointer operands.
- **Fix:** Updated the two fixtures to use the parser-accepted `!pto.ptr<f32>` inputs and `shape = [...], strides = [...]` syntax.
- **Files modified:** `test/phase2/tload_contract_trace.mlir`, `test/phase2/tstore_branch_shape.mlir`
- **Verification:** Both fixtures now parse and produce debug IR that includes `a5vm.load` and `a5vm.store`.
- **Committed in:** `0755afd` (part of task commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes were required to make the planned backend verification path reachable. No architectural scope changed.

## Issues Encountered

- The workspace build directory is configured with `Unix Makefiles`, so the plan's `ninja -C build ...` verification command loops on stale `build.ninja`. Verification was run with `CCACHE_DISABLE=1 cmake --build build --target ...` instead.
- `FileCheck` is not available on the current `PATH`, so verification used direct `rg` checks against `--a5vm-print-ir` output instead of the plan's literal `FileCheck` commands.
- The A5VM text emitter still reports unsupported non-A5VM ops during final emission, but the planned Phase 2 debug IR checks succeed because the lowered A5VM ops and diagnostics appear before emission fails.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The A5VM backend boundary now receives explicit `a5vm.load`, `a5vm.abs`, and vec-path `a5vm.store` ops with preserved Phase 2 metadata.
- Phase 3 can build on this lowering boundary to improve final A5VM text emission and the remaining mixed PTO/shared-dialect cleanup on the backend path.

## Self-Check: PASSED

- Found summary file `.planning/phases/02-pto-lowering/02-pto-lowering-03-SUMMARY.md`
- Found task commit `5212d32`
- Found task commit `0755afd`

---
*Phase: 02-pto-lowering*
*Completed: 2026-03-19*
