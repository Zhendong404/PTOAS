---
phase: 01
slug: a5vm-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-18
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | other — MLIR `RUN:` + `FileCheck` tests plus `ctest` smoke checks |
| **Config file** | none committed in source tree; current lit config appears external or build-generated |
| **Quick run command** | `./build/tools/ptoas/ptoas test/phase1/<case>.mlir -o - | FileCheck test/phase1/<case>.mlir` |
| **Full suite command** | `ctest --test-dir build --output-on-failure` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./build/tools/ptoas/ptoas test/phase1/<case>.mlir -o - | FileCheck test/phase1/<case>.mlir`
- **After every plan wave:** Run `ctest --test-dir build --output-on-failure`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 0 | BACK-01 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase1/a5vm_backend_switch.mlir -o - | FileCheck test/phase1/a5vm_backend_switch.mlir` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 0 | BACK-02 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase1/a5vm_shared_dialects.mlir -o - | FileCheck test/phase1/a5vm_shared_dialects.mlir` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 0 | A5VM-01 | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir` | ❌ W0 | ⬜ pending |
| 01-01-04 | 01 | 0 | A5VM-02 | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_load_op.mlir -o - | FileCheck test/phase1/a5vm_load_op.mlir` | ❌ W0 | ⬜ pending |
| 01-01-05 | 01 | 0 | A5VM-03 | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_abs_op.mlir -o - | FileCheck test/phase1/a5vm_abs_op.mlir` | ❌ W0 | ⬜ pending |
| 01-01-06 | 01 | 0 | A5VM-04 | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_store_op.mlir -o - | FileCheck test/phase1/a5vm_store_op.mlir` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `test/phase1/a5vm_vec_type.mlir` — legal/illegal type parsing and verifier coverage for A5VM-01
- [ ] `test/phase1/a5vm_load_op.mlir` — op assembly, printing, and verifier coverage for A5VM-02
- [ ] `test/phase1/a5vm_abs_op.mlir` — unary op verifier and print/parse coverage for A5VM-03
- [ ] `test/phase1/a5vm_store_op.mlir` — store op verifier and metadata coverage for A5VM-04
- [ ] `test/phase1/a5vm_backend_switch.mlir` — backend selection and text emission path for BACK-01
- [ ] `test/phase1/a5vm_shared_dialects.mlir` — shared-dialect preservation for BACK-02
- [ ] A documented lit/FileCheck invocation path for source tests if no committed lit config is added in this phase

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LLVM-like textual HIVM readability and unresolved-marker usefulness | BACK-01 | final consumer formatting is judged externally and not fully machine-validated locally | Build a Phase 1 sample with `--pto-backend=a5vm`, inspect emitted text, confirm it is close to `.ll` style and unresolved cases are explicitly marked |
| Developer diagnostics quality for intrinsic selection and unresolved summary reporting | BACK-01 | usefulness of debug output is partly qualitative | Run `ptoas` with the new debug flags, confirm intermediate IR, intrinsic-selection traces, and unresolved-summary artifact are all produced |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
