---
phase: 03
slug: hivm-emission
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-19
---

# Phase 03 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | other — MLIR source fixtures with `FileCheck`, direct runner script, and local `llvm-as` parse checks |
| **Config file** | none committed in source tree; direct `RUN:` lines plus `test/phase3/run_phase3_checks.sh` |
| **Quick run command** | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase3/<case>.mlir -o - | FileCheck test/phase3/<case>.mlir` |
| **Full suite command** | `bash test/phase3/run_phase3_checks.sh && ctest --test-dir build --output-on-failure` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase3/<case>.mlir -o - | FileCheck test/phase3/<case>.mlir`
- **After every plan wave:** Run `bash test/phase3/run_phase3_checks.sh && ctest --test-dir build --output-on-failure`
- **Before `$gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 0 | HIVM-01 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase3/backend_switch.mlir -o - | FileCheck test/phase3/backend_switch.mlir` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 0 | HIVM-02 | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --dump-a5vm-ir test/phase3/intrinsic_naming.mlir -o - 2>&1 | FileCheck test/phase3/intrinsic_naming.mlir` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 0 | HIVM-03 | integration | `out=$(mktemp) && ./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase3/abs_emit.mlir -o "$out" && /data/mouliangyu/projects/github.com/llvm/llvm-project/install/bin/llvm-as "$out" -o /dev/null && FileCheck test/phase3/abs_emit.mlir < "$out"` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 0 | HIVM-01,HIVM-02,HIVM-03 | integration | `bash test/phase3/run_phase3_checks.sh` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `test/phase3/backend_switch.mlir` — lock the exact replacement of the final `emitc` output slot with LLVM-like HIVM text emission
- [ ] `test/phase3/intrinsic_naming.mlir` — lock intrinsic spelling derived from family, vector type, and variant metadata
- [ ] `test/phase3/unresolved_report.mlir` — lock placeholder call shape plus unresolved sidecar content
- [ ] `test/phase3/abs_emit.mlir` — lock full module shape and parseable `Abs`-path output
- [ ] `test/phase3/run_phase3_checks.sh` — direct runner for all Phase 3 fixtures plus `llvm-as`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| The emitted text remains close to final downstream-consumed HIVM style instead of regressing into a debug dump | HIVM-01,HIVM-03 | textual quality and closeness to final consumer format are partly qualitative | Inspect one emitted `Abs` output file and confirm it has a real module/function/declaration shape, LLVM-style types, and no stray A5VM syntax in the main text |
| Unresolved placeholder policy remains honest about missing mappings while still preserving structure | HIVM-02,HIVM-03 | whether placeholders are appropriately conservative is partly a policy review, not only a syntax check | Inspect emitted placeholder calls and the unresolved report together; confirm the report includes op, placeholder or candidate name, participating fields, missing fields, and location |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
