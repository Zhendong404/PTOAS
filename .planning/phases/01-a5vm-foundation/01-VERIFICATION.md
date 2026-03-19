---
phase: 01-a5vm-foundation
verified: 2026-03-18T17:48:59Z
status: human_needed
score: 4/4 must-haves verified
human_verification:
  - test: "Inspect LLVM-like A5VM text output for backend-switch fixture"
    expected: "Output begins with ModuleID and declare/define lines, contains llvm.hivm calls, and unresolved mappings are explicit comments rather than silent guesses."
    why_human: "Readability and downstream usefulness of the textual form are qualitative."
  - test: "Inspect developer diagnostics for A5VM debug flags"
    expected: "--a5vm-print-ir shows shared arith/scf ops alongside a5vm ops, --a5vm-print-intrinsics prints mapping decisions, and unresolved report contents are understandable."
    why_human: "Diagnostic usefulness is partly qualitative, and this shell does not provide FileCheck for the exact scripted verification path."
---

# Phase 1: A5VM Foundation Verification Report

**Phase Goal:** Create the new backend entry point and the minimum `a5vm` dialect surface required to represent the `Abs` vector path without relying on `emitc`.
**Verified:** 2026-03-18T17:48:59Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | A new backend path exists at the current `emitc` boundary without redesigning the overall pass pipeline. | ✓ VERIFIED | `ptoas` exposes `--pto-backend=emitc|a5vm` and keeps `emitc` default in `tools/ptoas/ptoas.cpp:136`; the A5VM branch skips `createEmitPTOManualPass`/`emitc::translateToCpp` and calls `translateA5VMModuleToText(...)` in `tools/ptoas/ptoas.cpp:908` and `tools/ptoas/ptoas.cpp:925`; direct invocation produced LLVM-like text plus unresolved output for `test/phase1/a5vm_backend_switch.mlir`. |
| 2 | `a5vm` defines legal fixed-width 256-byte vector typing and rejects illegal widths. | ✓ VERIFIED | `A5VM_VecType` is declared in `include/PTO/IR/A5VMTypes.td:16`; verifier enforces `elementCount * bitWidth == 2048` and emits `expected exactly 256 bytes` in `lib/PTO/IR/A5VM.cpp:69`; direct `ptoas` run on `test/phase1/a5vm_vec_type.mlir` printed both legal forms and both expected width errors. |
| 3 | `a5vm` contains the minimum load, abs, and store style operations needed for the `Abs` path. | ✓ VERIFIED | `A5VM_LoadOp`, `A5VM_AbsOp`, and `A5VM_StoreOp` are defined in `include/PTO/IR/A5VMOps.td:23`, `include/PTO/IR/A5VMOps.td:44`, and `include/PTO/IR/A5VMOps.td:55`; verifiers for load/abs/store are implemented in `lib/PTO/IR/A5VM.cpp:123`, `lib/PTO/IR/A5VM.cpp:142`, and `lib/PTO/IR/A5VM.cpp:152`; direct `ptoas` runs on the phase fixtures printed canonical load/store assembly and the expected `a5vm.abs` mismatch error. |
| 4 | General control flow and scalar arithmetic remain handled by shared dialects rather than moving into `a5vm`. | ✓ VERIFIED | The shared-dialect fixture explicitly mixes `arith.addi`, `scf.for`, and `a5vm.abs` in `test/phase1/a5vm_shared_dialects.mlir:9`; `--a5vm-print-ir` output showed `A5VM IR op: arith.addi`, `scf.for`, `scf.yield`, and `a5vm.abs`; the emitter handles `arith.addi` and preserves `scf.for` as `A5VM-NONLOWERED` instead of inventing new A5VM ops in `lib/PTO/Transforms/A5VMTextEmitter.cpp:343` and `lib/PTO/Transforms/A5VMTextEmitter.cpp:358`. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `test/phase1/a5vm_vec_type.mlir` | Fixture for legal/illegal 256-byte vector typing | ✓ VERIFIED | Exists, 31 lines, includes legal and illegal width cases plus `RUN:` line (`test/phase1/a5vm_vec_type.mlir:1`). Wired to `run_phase1_checks.sh:17`. |
| `test/phase1/a5vm_load_op.mlir` | Fixture for `a5vm.load` syntax and metadata | ✓ VERIFIED | Exists, 17 lines, asserts `layout`, `domain`, `valid_rows`, and `valid_cols` (`test/phase1/a5vm_load_op.mlir:7`). Wired to `run_phase1_checks.sh:20`. |
| `test/phase1/a5vm_abs_op.mlir` | Fixture for `a5vm.abs` typing and mismatch diagnostic | ✓ VERIFIED | Exists, 18 lines, includes positive and negative cases (`test/phase1/a5vm_abs_op.mlir:5`). Wired to `run_phase1_checks.sh:23`. |
| `test/phase1/a5vm_store_op.mlir` | Fixture for `a5vm.store` syntax and metadata | ✓ VERIFIED | Exists, 18 lines, checks metadata and omission of load-only attrs (`test/phase1/a5vm_store_op.mlir:7`). Wired to `run_phase1_checks.sh:26`. |
| `test/phase1/a5vm_backend_switch.mlir` | Fixture for A5VM backend selection and unresolved markers | ✓ VERIFIED | Exists, 21 lines, exercises `--pto-backend=a5vm` expectations (`test/phase1/a5vm_backend_switch.mlir:1`). Wired to `run_phase1_checks.sh:29`. |
| `test/phase1/a5vm_shared_dialects.mlir` | Fixture for shared `arith`/`scf` preservation | ✓ VERIFIED | Exists, 21 lines, checks mixed dialect debug IR (`test/phase1/a5vm_shared_dialects.mlir:1`). Wired to `run_phase1_checks.sh:40`. |
| `test/phase1/run_phase1_checks.sh` | Executable verification runner | ✓ VERIFIED | Exists, executable, guards missing `ptoas`, drives all six fixtures plus unresolved report and intrinsic tracing (`test/phase1/run_phase1_checks.sh:1`). Exact end-to-end execution in this shell is blocked only by missing `FileCheck`. |
| `include/PTO/IR/A5VMTypes.td` | A5VM vector type contract | ✓ VERIFIED | Defines `A5VM_VecType` with custom assembly and verifier declaration (`include/PTO/IR/A5VMTypes.td:16`). Used by ops and handwritten implementation. |
| `include/PTO/IR/A5VMOps.td` | Minimal load/abs/store op contracts | ✓ VERIFIED | Defines all three ops with required operands/attrs and verifiers (`include/PTO/IR/A5VMOps.td:23`). Implemented by `lib/PTO/IR/A5VM.cpp`. |
| `lib/PTO/IR/A5VM.cpp` | Type parsing/printing and op/type verification | ✓ VERIFIED | Exists, 161 lines, initializes dialect and implements type parser/printer plus verifiers (`lib/PTO/IR/A5VM.cpp:47`). Linked into `PTOIR` via `lib/PTO/IR/CMakeLists.txt:2`. |
| `tools/ptoas/ptoas.cpp` | Dialect registration, backend selector, and A5VM emission branch | ✓ VERIFIED | Registers and loads `A5VMDialect`, exposes backend/debug flags, and branches to A5VM text emission (`tools/ptoas/ptoas.cpp:726`, `tools/ptoas/ptoas.cpp:763`, `tools/ptoas/ptoas.cpp:925`). |
| `include/PTO/Transforms/A5VMTextEmitter.h` | A5VM emitter API and options | ✓ VERIFIED | Declares `A5VMEmissionOptions` and `translateA5VMModuleToText` (`include/PTO/Transforms/A5VMTextEmitter.h:11`). Used by `ptoas`. |
| `lib/PTO/Transforms/A5VMTextEmitter.cpp` | LLVM-like text emission and unresolved-report logic | ✓ VERIFIED | Exists, 397 lines, synthesizes `@llvm.hivm.*` names, emits unresolved comments, writes sidecar reports, and preserves shared ops (`lib/PTO/Transforms/A5VMTextEmitter.cpp:127`, `lib/PTO/Transforms/A5VMTextEmitter.cpp:301`, `lib/PTO/Transforms/A5VMTextEmitter.cpp:379`). |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `include/PTO/IR/A5VMOps.td` | `lib/PTO/IR/A5VM.cpp` | Generated op declarations plus handwritten verifiers | ✓ WIRED | `A5VM_LoadOp`/`AbsOp`/`StoreOp` are declared in `include/PTO/IR/A5VMOps.td:23` and implemented as `LoadOp::verify`, `AbsOp::verify`, and `StoreOp::verify` in `lib/PTO/IR/A5VM.cpp:123`. |
| `tools/ptoas/ptoas.cpp` | `include/PTO/IR/A5VM.h` | Dialect registration and loading | ✓ WIRED | `PTO/IR/A5VM.h` is included at `tools/ptoas/ptoas.cpp:9`; registry/load calls for `A5VMDialect` are at `tools/ptoas/ptoas.cpp:726` and `tools/ptoas/ptoas.cpp:763`. |
| `tools/ptoas/ptoas.cpp` | `include/PTO/Transforms/A5VMTextEmitter.h` | Final emission branch invokes emitter API | ✓ WIRED | Header included at `tools/ptoas/ptoas.cpp:11`; `translateA5VMModuleToText(...)` is called in the A5VM backend branch at `tools/ptoas/ptoas.cpp:937`. |
| `lib/PTO/Transforms/A5VMTextEmitter.cpp` | `test/phase1/a5vm_backend_switch.mlir` | `llvm.hivm` call spelling and unresolved comment format | ✓ WIRED | Emitter creates `@llvm.hivm.` names at `lib/PTO/Transforms/A5VMTextEmitter.cpp:151`, `lib/PTO/Transforms/A5VMTextEmitter.cpp:162`, and `lib/PTO/Transforms/A5VMTextEmitter.cpp:176`, and writes `; A5VM-UNRESOLVED:` at `lib/PTO/Transforms/A5VMTextEmitter.cpp:313`; direct `ptoas` output matched the fixture’s intended shape. |
| `test/phase1/run_phase1_checks.sh` | All phase fixtures | Direct `ptoas` plus report/intrinsics commands | ✓ WIRED | The runner invokes each fixture in order and checks unresolved report generation plus intrinsic tracing (`test/phase1/run_phase1_checks.sh:17` through `test/phase1/run_phase1_checks.sh:42`). |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| `BACK-01` | `01-01-PLAN.md`, `01-03-PLAN.md` | Developer can run the existing PTOAS compilation flow with a backend path that replaces the current `emitc` generation slot without requiring a pass-pipeline redesign. | ✓ SATISFIED | Backend selector defaults to `emitc` and adds an explicit A5VM branch in `tools/ptoas/ptoas.cpp:136` and `tools/ptoas/ptoas.cpp:925`; the A5VM backend-switch fixture and runner encode the command path in `test/phase1/a5vm_backend_switch.mlir:1` and `test/phase1/run_phase1_checks.sh:29`; direct invocation emitted LLVM-like text and unresolved sidecar output. |
| `BACK-02` | `01-01-PLAN.md`, `01-03-PLAN.md` | Developer can keep ordinary control flow and scalar arithmetic in shared dialects such as `scf` and `arith` while only hardware-facing PTO operations enter the new backend path. | ✓ SATISFIED | Shared-dialect fixture encodes this contract in `test/phase1/a5vm_shared_dialects.mlir:5`; `--a5vm-print-ir` output preserved `arith.addi` and `scf.for`; the emitter explicitly handles `arith.addi` and leaves `scf.for` as non-lowered shared structure in `lib/PTO/Transforms/A5VMTextEmitter.cpp:343`. |
| `A5VM-01` | `01-01-PLAN.md`, `01-02-PLAN.md` | Developer can represent legal `a5vm` vector types whose total width is always exactly 256 bytes. | ✓ SATISFIED | Type contract is defined in `include/PTO/IR/A5VMTypes.td:16`; verifier enforces the 256-byte rule in `lib/PTO/IR/A5VM.cpp:86`; fixture coverage is in `test/phase1/a5vm_vec_type.mlir:3`; direct `ptoas` run produced both legal and illegal cases. |
| `A5VM-02` | `01-01-PLAN.md`, `01-02-PLAN.md` | Developer can represent the `Abs` load path with an `a5vm` load operation whose result type is a legal `a5vm` vector type. | ✓ SATISFIED | `A5VM_LoadOp` is declared in `include/PTO/IR/A5VMOps.td:23`; verifier requires vector result plus metadata attrs in `lib/PTO/IR/A5VM.cpp:123`; fixture exercises the path in `test/phase1/a5vm_load_op.mlir:7`; direct `ptoas` run printed canonical `a5vm.load`. |
| `A5VM-03` | `01-01-PLAN.md`, `01-02-PLAN.md` | Developer can represent the `Abs` compute path with an `a5vm` absolute-value operation whose operand and result types are legal `a5vm` vector types. | ✓ SATISFIED | `A5VM_AbsOp` is declared in `include/PTO/IR/A5VMOps.td:44`; verifier enforces matching legal vector types in `lib/PTO/IR/A5VM.cpp:142`; fixture coverage is in `test/phase1/a5vm_abs_op.mlir:3`; direct `ptoas` run emitted the expected mismatch diagnostic. |
| `A5VM-04` | `01-01-PLAN.md`, `01-02-PLAN.md` | Developer can represent the `Abs` store path with an `a5vm` store operation that consumes a legal `a5vm` vector value and backend-specific addressing inputs. | ✓ SATISFIED | `A5VM_StoreOp` is declared in `include/PTO/IR/A5VMOps.td:55`; verifier requires legal vector value plus `layout`/`domain` attrs in `lib/PTO/IR/A5VM.cpp:152`; fixture coverage is in `test/phase1/a5vm_store_op.mlir:3`; direct `ptoas` run printed canonical `a5vm.store`. |

All requirement IDs declared in phase plan frontmatter are accounted for in `.planning/REQUIREMENTS.md`, and `.planning/REQUIREMENTS.md` does not map any additional Phase 1 requirement IDs beyond `BACK-01`, `BACK-02`, `A5VM-01`, `A5VM-02`, `A5VM-03`, and `A5VM-04`.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `include/PTO/Transforms/Passes.h` | 9 | `TODO` comment | ℹ️ Info | Legacy comment only; does not affect the verified A5VM backend path. |
| `include/PTO/Transforms/Passes.td` | 11 | `TODO` comment | ℹ️ Info | Legacy comment only; does not affect Phase 1 goal achievement. |

### Human Verification Required

### 1. LLVM-like Output Review

**Test:** Run `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-allow-unresolved --a5vm-unresolved-report=/tmp/a5vm.report test/phase1/a5vm_backend_switch.mlir -o -` and inspect the emitted text.
**Expected:** The output starts with a `ModuleID`, contains `declare` and `define` lines, uses `@llvm.hivm.*` spellings for resolved ops, and marks unresolved mappings with `; A5VM-UNRESOLVED:`.
**Why human:** The exact readability and downstream usability of the textual form are qualitative.

### 2. Diagnostic Quality Review

**Test:** Run `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase1/a5vm_shared_dialects.mlir -o /dev/null` and `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-intrinsics --a5vm-allow-unresolved --a5vm-unresolved-report=/tmp/a5vm.report test/phase1/a5vm_backend_switch.mlir -o -`.
**Expected:** `--a5vm-print-ir` shows shared `arith`/`scf` operations alongside `a5vm.abs`; `--a5vm-print-intrinsics` prints `A5VM intrinsic:` lines with mapping details; the unresolved report contains one line per unresolved mapping.
**Why human:** The implementation is present and produces output, but the usefulness of these diagnostics is partly subjective; this shell also lacks `FileCheck`, so the exact scripted runner could not be executed verbatim.

### Gaps Summary

No implementation gaps were found against the phase goal or the six Phase 1 requirement IDs. The codebase contains the new A5VM IR surface, an explicit backend boundary in `ptoas`, a dedicated A5VM text emitter, and committed verification fixtures/runner. The only remaining work is qualitative human review of emitted text and diagnostics, plus running the scripted `FileCheck`-based loop in an environment where `FileCheck` is installed.

---

_Verified: 2026-03-18T17:48:59Z_
_Verifier: Claude (gsd-verifier)_
