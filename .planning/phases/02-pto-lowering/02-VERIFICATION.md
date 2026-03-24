---
phase: 02-pto-lowering
verified: 2026-03-19T04:15:29Z
status: human_needed
score: 8/9 must-haves verified
re_verification:
  previous_status: gaps_found
  previous_score: 3/9
  gaps_closed:
    - "Exact TLOAD/TSTORE metadata recovery is implemented through the lowering-time memref adapter chain and locked in committed fixtures."
    - "The lowered TABS path now constructs an explicit one-trip dummy vec-scope carrier loop and locks that contract in the fixture."
    - "Phase 2 lowering now preserves full TABS/TSTORE diagnostic coverage by walking PTO ops and delaying pass failure until the module walk completes."
    - "The committed runner no longer trips on the stale bare-`scf.for` fixture mismatch and now fails missing-FileCheck as an explicit actionable prerequisite error."
  gaps_remaining: []
  regressions: []
human_verification:
  - test: "Provision a runnable build tree and an accepted FileCheck path, then run `bash test/phase2/run_phase2_checks.sh`."
    expected: "All six Phase 2 fixtures pass and the script reaches the final `ctest --test-dir build --output-on-failure` step without stale-fixture or PATH-only FileCheck failures."
    why_human: "This workspace currently lacks both an executable `./build/tools/ptoas/ptoas` and a FileCheck binary in the runner's committed search path."
  - test: "Run `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tabs_abs_loop_shape.mlir -o /dev/null 2>&1 | <FileCheck> test/phase2/tabs_abs_loop_shape.mlir` in a provisioned build."
    expected: "The printed IR shows the outer one-trip `scf.for` carrier with `cce_aiv_loop_hint` and `llvm.loop.aivector_scope`, a distinct inner chunk loop, and ordered `a5vm.vlds`, `a5vm.vabs`, `a5vm.vsts`."
    why_human: "The source and fixture wiring show the intended behavior, but emitted IR printing cannot be exercised in this workspace."
  - test: "Run the copy-family and diagnostic fixture commands from `test/phase2/run_phase2_checks.sh` in a provisioned build."
    expected: "TLOAD/TSTORE emitted IR satisfies the exact `[32, 1]`, `[0, 0]`, `[32, 32]` metadata checks, and the TABS/TSTORE diagnostic fixtures surface all locked messages in one run."
    why_human: "The current workspace cannot execute `ptoas | FileCheck`, so end-to-end lowering output remains environment-gated."
---

# Phase 2: PTO Lowering Verification Report

**Phase Goal:** Implement PTO-to-A5VM lowering that preserves the real PTO-library control structure and semantic decisions for the `Abs` path.
**Verified:** 2026-03-19T04:15:29Z
**Status:** human_needed
**Re-verification:** Yes — after gap-closure execution

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | The public Phase 2 lowering contract exposes explicit AIV loop-scope carriage and stable `lowerTLOAD` / `lowerTABS` / `lowerTSTORE` entrypoints. | ✓ VERIFIED | `A5VMLoopScopeContract`, `attachLoopScopeMetadata`, `programCopyGmToUbLoops`, `programCopyUbToGmLoops`, `buildUnaryVecScope`, and the three public entrypoints are present in `include/PTO/Transforms/A5VMLowering.h`. |
| 2 | TLOAD lowering preserves PTO-library copy-family structure and recovers exact source stride and partition-trace metadata through the lowering-time value chain. | ✓ VERIFIED | `resolveTensorViewBase` walks `pto.partition_view`, `memref.subview`, `memref.reinterpret_cast`, and `memref.cast`, and `extractPartitionTrace` composes lowered subview offsets and sizes before `lowerTLOAD` materializes `a5vm::CopyGmToUbufOp` plus explicit loop-programming attrs in [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L308), [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L459), and [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L764). |
| 3 | TSTORE lowering preserves PTO-library copy-family structure, explicit VEC/ACC/MAT branching, and recovered destination stride and partition-trace metadata. | ✓ VERIFIED | `lowerTSTORE` branches on `A5VMTileDomain`, emits dedicated ACC/MAT TODO diagnostics, resolves destination tensor-view metadata through the same adapter chain, and materializes `a5vm::CopyUbufToGmOp` plus UB-to-out loop programming in [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L524), [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L590), and [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L842). |
| 4 | The TABS `Abs` path preserves `__VEC_SCOPE__` as a dedicated dummy loop carrier distinct from the inner chunk loop and ordered vector pipeline. | ✓ VERIFIED | `buildUnaryVecScope` creates an outer one-trip `scf.for`, attaches `cce_aiv_loop_hint` and `llvm.loop.aivector_scope`, then creates a separate inner chunk loop around `a5vm::VldsOp`, `a5vm::VabsOp`, and `a5vm::VstsOp` in [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L601) and [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L710). |
| 5 | Unsupported TABS and TSTORE cases now surface all locked diagnostics instead of stopping after the first failed PTO op. | ✓ VERIFIED | `lowerTABS` accumulates all four precheck diagnostics before returning failure, and `PTOToA5VMPass` now walks all PTO ops, records failure, and only calls `signalPassFailure()` after the walk in [lib/PTO/Transforms/PTOToA5VMLowering.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VMLowering.cpp#L808) and [lib/PTO/Transforms/PTOToA5VM.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VM.cpp#L64). |
| 6 | `ptoas --pto-backend=a5vm` remains wired through the Phase 2 helper layer rather than regressing to EmitC-side pseudo lowering. | ✓ VERIFIED | `tools/ptoas/ptoas.cpp` still routes the A5VM backend through `createLowerPTOToA5VMPass()` before A5VM debug IR emission in [tools/ptoas/ptoas.cpp](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp#L292). |
| 7 | The committed Phase 2 fixtures lock the corrected semantic contracts for copy metadata, vec-scope ownership, unsupported-case diagnostics, and backend wiring. | ✓ VERIFIED | `test/phase2/tload_copy_family_shape.mlir`, `test/phase2/tstore_copy_family_shape.mlir`, `test/phase2/tabs_abs_loop_shape.mlir`, `test/phase2/tabs_precheck_a5.mlir`, `test/phase2/tstore_domain_todos.mlir`, and `test/phase2/pto_backend_a5vm_wiring.mlir` all contain the tightened checks described by the gap-closure plans. |
| 8 | The committed runner and validation docs no longer depend on the stale bare-`scf.for` contract and use deterministic FileCheck resolution. | ✓ VERIFIED | `test/phase2/run_phase2_checks.sh` rejects obsolete pseudo-op fixtures, rejects bare `// CHECK: scf.for`, resolves FileCheck through a fixed search order, and emits actionable missing-tool errors; `.planning/phases/02-pto-lowering/02-VALIDATION.md` names the same runner contract; `.planning/phases/02-pto-lowering/deferred-items.md` no longer records the old stale fixture mismatch. |
| 9 | This workspace can execute the committed Phase 2 runner end to end and prove the emitted/runtime contracts directly. | ? NEEDS HUMAN | `bash test/phase2/run_phase2_checks.sh` now fails only on external prerequisites: `error: missing FileCheck; checked: FileCheck FileCheck-19 /usr/lib/llvm-19/bin/FileCheck`, and `./build/tools/ptoas/ptoas` is also absent in this workspace. |

**Score:** 8/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `include/PTO/Transforms/A5VMLowering.h` | Public lowering contract with explicit vec-scope carrier | ✓ VERIFIED | Contains the stable contract types and helper declarations required by the Phase 2 plans. |
| `lib/PTO/Transforms/PTOToA5VMLowering.cpp` | Substantive helper implementation for TLOAD/TABS/TSTORE | ✓ VERIFIED | Implements metadata recovery, copy-family attrs, explicit vec-scope carrier lowering, and dedicated diagnostics; file is substantive and wired through the pass. |
| `lib/PTO/Transforms/PTOToA5VM.cpp` | Thin pass boundary that preserves full diagnostic coverage | ✓ VERIFIED | Dispatches only through `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE`, while walking the full module before signaling failure. |
| `tools/ptoas/ptoas.cpp` | A5VM backend branch scheduling Phase 2 lowering | ✓ VERIFIED | Keeps the A5VM backend on the pass pipeline path instead of reconstructing lowering in the emitter. |
| `test/phase2/tload_copy_family_shape.mlir` | Locks exact TLOAD copy metadata | ✓ VERIFIED | Requires `src_strides = [32, 1]`, `trace_offsets = [0, 0]`, and `trace_sizes = [32, 32]`. |
| `test/phase2/tabs_abs_loop_shape.mlir` | Locks dummy vec-scope carrier ownership | ✓ VERIFIED | Requires explicit loop-carrier attrs and ordered `a5vm.vlds` / `a5vm.vabs` / `a5vm.vsts`. |
| `test/phase2/tabs_precheck_a5.mlir` | Locks all four TABS precheck diagnostics | ✓ VERIFIED | Requires the dedicated diagnostics and forbids the previous generic legalization failure. |
| `test/phase2/tstore_copy_family_shape.mlir` | Locks exact TSTORE copy metadata | ✓ VERIFIED | Requires `dst_strides = [32, 1]`, `trace_offsets = [0, 0]`, and `trace_sizes = [32, 32]`. |
| `test/phase2/tstore_domain_todos.mlir` | Locks explicit ACC/MAT TODO diagnostics | ✓ VERIFIED | Requires both dedicated TODO messages in one run. |
| `test/phase2/run_phase2_checks.sh` | Canonical Phase 2 runner with deterministic preflight | ⚠️ ENV BLOCKED | The script is correctly updated, but end-to-end execution requires a provisioned `FileCheck` in the committed search path and an executable `./build/tools/ptoas/ptoas`. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `lib/PTO/Transforms/PTOToA5VMLowering.cpp` | `test/phase2/tload_copy_family_shape.mlir` | recovered source metadata + `a5vm.copy_gm_to_ubuf` loop programming | ✓ VERIFIED | Lowering code now computes and attaches the exact metadata that the fixture locks. |
| `lib/PTO/Transforms/PTOToA5VMLowering.cpp` | `test/phase2/tstore_copy_family_shape.mlir` | recovered destination metadata + `a5vm.copy_ubuf_to_gm` loop programming | ✓ VERIFIED | Lowering code now computes and attaches the exact metadata that the fixture locks. |
| `lib/PTO/Transforms/PTOToA5VMLowering.cpp` | `test/phase2/tabs_abs_loop_shape.mlir` | dummy carrier loop plus ordered `vlds -> vabs -> vsts` | ✓ VERIFIED | The helper creates the carrier loop and vector sequence that the fixture checks. |
| `lib/PTO/Transforms/PTOToA5VM.cpp` | `test/phase2/tabs_precheck_a5.mlir` | whole-module PTO walk preserves full diagnostic surface | ✓ VERIFIED | The pass no longer short-circuits after the first failed PTO op. |
| `lib/PTO/Transforms/PTOToA5VM.cpp` | `test/phase2/tstore_domain_todos.mlir` | whole-module PTO walk surfaces both ACC and MAT TODO diagnostics | ✓ VERIFIED | The pass will continue visiting later PTO ops after an earlier failure. |
| `tools/ptoas/ptoas.cpp` | `test/phase2/pto_backend_a5vm_wiring.mlir` | `createLowerPTOToA5VMPass` before A5VM IR inspection | ✓ VERIFIED | The backend route remains correctly wired in the CLI path. |
| `test/phase2/run_phase2_checks.sh` | `.planning/phases/02-pto-lowering/02-VALIDATION.md` | same runner-first validation contract | ✓ VERIFIED | Both now document `bash test/phase2/run_phase2_checks.sh` as the Phase 2 entrypoint and describe deterministic FileCheck resolution. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| PTO-01 | `02-01`, `02-02`, `02-04`, `02-06` | Lower PTO `TLOAD` on the `Abs` path into real PTO-library GM-to-UB copy structure while preserving layout, shape, valid-region, stride, and trace decisions. | ? NEEDS HUMAN | Code and fixtures now preserve the required structure and metadata, but emitted IR cannot be exercised in this workspace because the runner prerequisites are absent. |
| PTO-02 | `02-01`, `02-02`, `02-03`, `02-05`, `02-06` | Lower PTO `TABS` on the `Abs` path into the PTO-library vector pipeline and loop structure, including explicit hardware/software loop semantics. | ? NEEDS HUMAN | Source-level wiring now preserves the explicit dummy vec-scope carrier and full precheck diagnostics, but printed runtime IR still needs execution in a provisioned build. |
| PTO-03 | `02-01`, `02-02`, `02-04`, `02-05`, `02-06` | Lower PTO `TSTORE` on the `Abs` path into real PTO-library UB-to-GM copy structure while preserving source-tile-domain and destination-layout behavior. | ? NEEDS HUMAN | Source-level wiring now preserves copy metadata and explicit ACC/MAT diagnostics, but emitted/runtime output still needs execution in a provisioned build. |
| PTO-04 | `02-01`, `02-02`, `02-03`, `02-04`, `02-05`, `02-06` | Add new PTO-to-A5VM lowerings through the same framework without changing the backend architecture established for `Abs`. | ✓ SATISFIED | The public contracts, thin pass boundary, fixture contracts, and `ptoas` backend branch remain framework-shaped and reusable for future PTO ops. |

Orphaned requirements: none. All requirement IDs declared in Phase 2 plan frontmatter (`PTO-01`, `PTO-02`, `PTO-03`, `PTO-04`) are present in `.planning/REQUIREMENTS.md` and accounted for above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `.planning/phases/02-pto-lowering/02-VALIDATION.md` | 43 | Per-task command still shows bare `FileCheck` instead of the runner's resolved `filecheck_bin` strategy | ⚠️ Warning | Documentation is mostly aligned at the runner level, but the per-task map still reflects the older direct-command assumption. |
| `lib/PTO/Transforms/PTOToA5VMLowering.cpp` | 590 | `TSTORE ACC lowering TODO for a5vm backend` | ℹ️ Info | Intentional unsupported-branch diagnostic locked by the fixture contract. |
| `lib/PTO/Transforms/PTOToA5VMLowering.cpp` | 595 | `TSTORE MAT lowering TODO for a5vm backend` | ℹ️ Info | Intentional unsupported-branch diagnostic locked by the fixture contract. |

### Human Verification Required

### 1. Runner Smoke In A Provisioned Build

**Test:** Provision a runnable `./build/tools/ptoas/ptoas` and a FileCheck binary in the committed search path, then run `bash test/phase2/run_phase2_checks.sh`.
**Expected:** All six Phase 2 fixtures pass, then the script reaches `ctest --test-dir build --output-on-failure`.
**Why human:** This workspace currently lacks both prerequisites, so the runner cannot exercise emitted/runtime behavior here.

### 2. TABS Carrier IR Contract

**Test:** Run the direct `tabs_abs_loop_shape.mlir` `ptoas --a5vm-print-ir | FileCheck` command from the fixture in a provisioned build.
**Expected:** The emitted IR shows the explicit outer dummy vec-scope carrier, distinct inner chunk loop, and ordered `a5vm.vlds -> a5vm.vabs -> a5vm.vsts`.
**Why human:** The helper and fixture are aligned, but printed IR cannot be checked without a runnable toolchain.

### 3. Copy-Family Metadata And Diagnostic Contracts

**Test:** Run the direct copy-family and diagnostic fixture commands from `test/phase2/run_phase2_checks.sh` in a provisioned build.
**Expected:** TLOAD/TSTORE emitted IR matches the exact stride/trace checks, and the TABS/TSTORE unsupported-case fixtures emit all locked diagnostics in one run.
**Why human:** The current workspace cannot execute `ptoas | FileCheck`, so the observable runtime contract remains environment-gated.

### Gaps Summary

The previous semantic gaps are closed in the codebase. The lowering helper now recovers copy-family stride and partition metadata through the value forms that actually reach Phase 2 lowering; the TABS path now builds an explicit one-trip vec-scope carrier loop with both `cce_aiv_loop_hint` and `llvm.loop.aivector_scope`; and the pass now walks all PTO ops so dedicated diagnostics remain observable across the whole module instead of dying on the first failed conversion.

What remains is not a new semantic regression. The committed runner now fails in the intended, explicit way for missing prerequisites: it no longer rejects the fixture contract itself, but this workspace still does not provide an executable `./build/tools/ptoas/ptoas` or a FileCheck binary in the runner's committed search path. Because the Phase 2 goal is semantic lowering correctness, not local tool installation, this is recorded as `human_needed` rather than `gaps_found`.

---

_Verified: 2026-03-19T04:15:29Z_
_Verifier: Claude (gsd-verifier)_
