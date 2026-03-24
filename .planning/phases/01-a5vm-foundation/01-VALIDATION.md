---
phase: 01
slug: a5vm-foundation
status: ready
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-19
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | other — MLIR `RUN:` + `FileCheck` tests plus build-tree smoke checks |
| **Config file** | none committed in source tree; direct `RUN:` lines plus a phase runner script |
| **Quick run command** | `./build/tools/ptoas/ptoas test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir` |
| **Full suite command** | `bash test/phase1/run_phase1_checks.sh && ctest --test-dir build --output-on-failure` |
| **Estimated runtime** | ~20 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./build/tools/ptoas/ptoas test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir`
- **After every plan wave:** Run `bash test/phase1/run_phase1_checks.sh && ctest --test-dir build --output-on-failure`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 20 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | A5VM-01 | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 0 | A5VM-02,A5VM-04 | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_copy_gm_to_ubuf_op.mlir -o - | FileCheck test/phase1/a5vm_copy_gm_to_ubuf_op.mlir && ./build/tools/ptoas/ptoas test/phase1/a5vm_copy_ubuf_to_gm_op.mlir -o - | FileCheck test/phase1/a5vm_copy_ubuf_to_gm_op.mlir` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 0 | A5VM-03 | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_vabs_kernel_shape.mlir -o - | FileCheck test/phase1/a5vm_vabs_kernel_shape.mlir` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 0 | BACK-01 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-allow-unresolved --a5vm-unresolved-report=%t test/phase1/a5vm_backend_switch.mlir -o - | FileCheck test/phase1/a5vm_backend_switch.mlir` | ❌ W0 | ⬜ pending |
| 01-01-05 | 01 | 0 | BACK-02 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase1/a5vm_shared_dialects.mlir -o /dev/null 2>&1 | FileCheck test/phase1/a5vm_shared_dialects.mlir` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `test/phase1/a5vm_vec_type.mlir` — legal/illegal corrected type parsing and verifier coverage for `mlir::a5vm`
- [ ] `test/phase1/a5vm_copy_gm_to_ubuf_op.mlir` — corrected GM-to-UB primitive assembly, printing, and verifier coverage
- [ ] `test/phase1/a5vm_vabs_kernel_shape.mlir` — corrected `vlds` / `vabs` / `vsts` compute-kernel primitive shape coverage
- [ ] `test/phase1/a5vm_copy_ubuf_to_gm_op.mlir` — corrected UB-to-GM primitive assembly, printing, and verifier coverage
- [ ] `test/phase1/a5vm_backend_switch.mlir` — corrected backend selection and thin text-emission path for BACK-01
- [ ] `test/phase1/a5vm_shared_dialects.mlir` — shared-dialect preservation for BACK-02 under the corrected primitive surface
- [ ] `test/phase1/run_phase1_checks.sh` — direct invocation path for all corrected Phase 1 checks

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Corrected A5VM primitive naming stays close to the A5 builtin layer instead of regressing to pseudo PTO ops | A5VM-02,A5VM-03,A5VM-04 | exact naming closeness to A5 builtin intent is partly a design review | Inspect the corrected dialect ops and one emitted backend sample; confirm op names track copy-family and `vlds` / `vabs` / `vsts` style primitives rather than `a5vm.load` / `a5vm.store` |
| Backend text and diagnostics remain usable while primitive A5VM semantics are still partially unresolved | BACK-01 | readability and usefulness of unresolved diagnostics are partly qualitative | Run `ptoas` with `--pto-backend=a5vm`, `--a5vm-print-ir`, and unresolved-report flags; confirm output stays inspectable and unresolved cases are explicit |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 45s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** ready for execution planning
