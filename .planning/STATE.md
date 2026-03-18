# Project State

**Updated:** 2026-03-18
**Status:** Ready for Phase 1 planning

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-03-18)

**Core value:** Preserve PTO library semantics and template-driven behavior inside PTOAS so backend lowering retains enough information to enable optimization instead of losing it during library instantiation.
**Current focus:** Phase 1 - A5VM Foundation

## Current Position

- Project initialized
- Workflow preferences captured
- Research completed
- Requirements defined
- Roadmap created
- No phase plans created yet

## Active Milestone

**Name:** Initial A5VM backend bring-up
**Goal:** Replace the `emitc` backend slot with a PTOAS-native `a5vm` path that can compile the `Abs` sample and produce textual LLVM HIVM intrinsic IR.

## Phase Status

| Phase | Name | Status |
|-------|------|--------|
| 1 | A5VM Foundation | Pending |
| 2 | PTO Lowering | Pending |
| 3 | HIVM Emission | Pending |
| 4 | Abs Validation | Pending |

## Requirements Snapshot

- Total v1 requirements: 16
- Complete: 0
- In Progress: 0
- Pending: 16
- Blocked: 0

## Key Decisions Snapshot

- Introduce `a5vm` as the hardware-facing backend dialect.
- Replace the current `emitc` slot rather than redesigning the pass pipeline.
- Keep v1 limited to the `Abs` sample and the minimum PTO interface set it requires.
- Emit textual LLVM HIVM intrinsic IR first, then confirm final intrinsic spellings externally.

## Recent Progress

- Created `.planning/PROJECT.md`
- Created `.planning/config.json`
- Created `.planning/research/`
- Created `.planning/REQUIREMENTS.md`
- Created `.planning/ROADMAP.md`

## Open Questions

- Which exact LLVM HIVM intrinsic spellings correspond to each builtin variant exercised by the final `Abs` path
- Whether the implemented `Abs` path needs only the currently expected load/abs/store intrinsic families or additional helper intrinsics

## Session Continuity

- Next recommended command: `$gsd-plan-phase 1`
- First execution target: establish the `a5vm` dialect and backend boundary before lowering PTO ops

---
*Last updated: 2026-03-18 after initial roadmap creation*
