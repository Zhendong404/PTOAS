# Phase 1: A5VM Foundation - Research

**Researched:** 2026-03-18
**Domain:** MLIR dialect bring-up and backend boundary replacement for PTOAS
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### Backend switching
- Keep dual backend paths during Phase 1.
- Select backend through an explicit CLI flag rather than a hidden or hardwired mode switch.
- Default CLI behavior should remain compatible with current usage, but new backend selection must be available for developers.

### Output format
- The new backend should target output that is as close as possible to final consumer-facing LLVM `.ll` style text.
- The output should avoid retaining extra intermediate semantic noise unless needed for unresolved cases.
- The textual HIVM emitter should be connected in `ptoas` at the current final emission point, directly replacing `emitc::translateToCpp` when the `a5vm` backend is selected.

### A5VM abstraction boundary
- `a5vm` lives in the existing PTO directory and namespace structure as a new dialect module, not as an ad hoc extension inside the old EmitC pass.
- `a5vm` should use a simple unified vector type spelling such as `!a5vm.vec<64xf32>`.
- `a5vm` is responsible for the hardware-facing abstraction layer, while the textual emitter is responsible for final HIVM intrinsic name synthesis.
- Phase 1 should already connect the thinnest usable textual HIVM emitter path rather than stopping at raw `a5vm` IR.

### Failure and placeholder policy
- When something can be emitted as legal textual IR, do that even if some details remain provisional.
- When a case cannot be emitted legally, output should include explicit unresolved markers or comments instead of silently guessing.
- Placeholder handling should preserve enough context that required intrinsic mappings can be reviewed and confirmed later.

### Developer diagnostics
- Provide explicit developer-facing debug flags for:
  - printing backend intermediate IR
  - printing intrinsic-selection decisions
  - continuing through unresolved mappings and producing a summary list
- Error messages for the new backend should be developer-oriented and include operation/type/mapping context rather than only end-user summaries.
- Unresolved intrinsic or emission gaps should be reported both in terminal/log output and in a separate artifact or file.

### Claude's Discretion
- Exact flag names for backend selection and debug controls
- Exact organization of the new dialect source files within the PTO directory hierarchy
- Exact mechanics for how unresolved intrinsic summaries are accumulated and written

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BACK-01 | Developer can run the existing PTOAS compilation flow with a backend path that replaces the current `emitc` generation slot without requiring a pass-pipeline redesign. | Reuse `tools/ptoas/ptoas.cpp` pipeline through the current `createEmitPTOManualPass(...)` / `emitc::translateToCpp(...)` boundary; add a backend selector and parallel A5VM emission branch instead of redesigning earlier passes. |
| BACK-02 | Developer can keep ordinary control flow and scalar arithmetic in shared dialects such as `scf` and `arith` while only hardware-facing PTO operations enter the new backend path. | Keep `func`/`scf`/`arith`/`memref` in the existing registry and pass pipeline; only lower hardware-facing PTO ops to `a5vm` and leave non-hardware code in shared dialects until textual emission. |
| A5VM-01 | Developer can represent legal `a5vm` vector types whose total width is always exactly 256 bytes. | Implement a first-class `A5VMVecType` verifier/parser/printer using MLIR type definitions and reject element-count combinations whose total width is not 256 bytes. |
| A5VM-02 | Developer can represent the `Abs` load path with an `a5vm` load operation whose result type is a legal `a5vm` vector type. | Add a minimal `a5vm.load` op whose result is `!a5vm.vec<...>` and whose operands/attrs carry unresolved addressing and layout metadata needed later by PTO lowering and emission. |
| A5VM-03 | Developer can represent the `Abs` compute path with an `a5vm` absolute-value operation whose operand and result types are legal `a5vm` vector types. | Add a minimal `a5vm.abs` unary op that enforces same-typed operand/result vectors and keeps intrinsic family selection in the emitter, not in the op name. |
| A5VM-04 | Developer can represent the `Abs` store path with an `a5vm` store operation that consumes a legal `a5vm` vector value and backend-specific addressing inputs. | Add a minimal `a5vm.store` op with a vector operand plus backend-facing address/layout metadata attributes or operands, matching later `TSTORE` codegen needs without collapsing to final intrinsic names yet. |
</phase_requirements>

## Summary

Phase 1 should be planned as a first-class MLIR dialect bring-up plus a backend switch at the existing `emitc` boundary, not as an incremental edit inside [`PTOToEmitC.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToEmitC.cpp). The repository already follows the standard MLIR out-of-tree pattern: TableGen-backed dialects under `include/PTO/IR`, implementations under `lib/PTO/IR`, pass declarations in `include/PTO/Transforms/Passes.h`, and tool orchestration in [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp). Planning should preserve that structure.

The minimum viable `a5vm` surface for this phase is smaller than PTO lowering but larger than a pure stub. It needs one verified vector type constrained to 256 bytes total width, three hardware-facing ops (`load`, `abs`, `store`), a pass/emitter entry point reachable from `ptoas` through an explicit backend flag, and developer diagnostics that preserve unresolved mapping context. Because CONTEXT.md explicitly locks in a thin textual HIVM path during Phase 1, planning should include emitter plumbing and unresolved-reporting scaffolding now, even if complete PTO semantic lowering and final intrinsic selection land later.

**Primary recommendation:** Build `a5vm` as a separate MLIR dialect and connect it through a new `ptoas` backend selector that keeps the current pass pipeline intact, emits minimal textual HIVM output for `Abs`, and records unresolved cases explicitly.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LLVM | 19.1.7 | Compiler infrastructure and support libraries | Already pinned by the workspace CMake toolchain and matched exactly by MLIR in the local install. |
| MLIR | 19.1.7 | Dialect/type/op definitions, passes, conversion infra, asm/printer/parser | The repo already uses MLIR TableGen dialect libraries and pass infrastructure throughout PTOAS. |
| PTOIR / PTOTransforms | workspace | Existing frontend dialect and passes | Phase 1 must integrate at the current PTOAS boundary instead of replacing the frontend architecture. |
| Ascend CANN PTO headers | 8.5.0 | Semantic reference for `TLOAD`, `TABS`, `TSTORE` behavior | These headers define the shape/layout/valid-region constraints the new backend must preserve. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| MLIR TableGen (`mlir_tablegen`, `add_mlir_dialect_library`) | 19.1.7 | Generate dialect/type/op declarations and defs | Use for the new `a5vm` dialect instead of handwritten registration boilerplate. |
| FileCheck / lit-style `RUN:` tests | LLVM 19.1.7 toolchain | Fast structural verification of MLIR/text output | Use for new phase tests because the repo already contains `RUN:` + `FileCheck` tests under `test/basic`. |
| `ctest` | CMake 4.2.3 workspace | Existing smoke test harness for `ptobc` integration | Use only for broader smoke coverage, not as the primary Phase 1 dialect verification loop. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| First-class `a5vm` dialect | Extend `PTOToEmitC.cpp` with more ad hoc cases | Faster short-term, but it violates the locked abstraction boundary and makes Phase 2/3 rework likely. |
| Verified `A5VMVecType` | Reuse builtin `vector<...>` directly | Simpler mechanically, but it loses the locked textual syntax (`!a5vm.vec<64xf32>`) and weakens backend-specific verification. |
| Backend selector in `ptoas` | Hidden environment switch or hardwired A5 path | Conflicts with locked developer-controlled dual-backend behavior. |

**Toolchain verification:**
```bash
sed -n '1,20p' /data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/llvm/LLVMConfigVersion.cmake
sed -n '1,20p' /data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/mlir/MLIRConfigVersion.cmake
sed -n '330,420p' build/CMakeCache.txt
```

**Verified local versions:**
- LLVM `19.1.7` from `install/lib/cmake/llvm/LLVMConfigVersion.cmake`
- MLIR `19.1.7` from `install/lib/cmake/mlir/MLIRConfigVersion.cmake`
- CANN PTO reference tree `8.5.0` from `/usr/local/Ascend/cann-8.5.0/...`

## Architecture Patterns

### Recommended Project Structure
```text
include/PTO/
├── IR/
│   ├── A5VM.h                # aggregate include
│   ├── A5VMDialect.h         # dialect class
│   ├── A5VMDialect.td        # dialect definition
│   ├── A5VMOps.td            # load/abs/store ops
│   └── A5VMTypeDefs.td       # !a5vm.vec<...>
└── Transforms/
    └── Passes.h              # new pass/emitter declarations

lib/PTO/
├── IR/
│   ├── A5VM.cpp
│   └── CMakeLists.txt
└── Transforms/
    ├── PTOToA5VM.cpp         # phase-appropriate boundary pass
    ├── A5VMToText.cpp        # thin textual HIVM emitter path
    └── CMakeLists.txt
```

### Pattern 1: First-Class MLIR Dialect Module
**What:** Add `a5vm` using the same TableGen and `add_mlir_dialect_library` pattern the repo already uses for PTO.
**When to use:** Immediately; this is the canonical repo fit for new IR.
**Example:**
```cmake
set(LLVM_TARGET_DEFINITIONS A5VMOps.td)
mlir_tablegen(A5VMDialect.h.inc -gen-dialect-decls -dialect=a5vm)
mlir_tablegen(A5VMDialect.cpp.inc -gen-dialect-defs -dialect=a5vm)
mlir_tablegen(A5VMOps.h.inc -gen-op-decls)
mlir_tablegen(A5VMOps.cpp.inc -gen-op-defs)
mlir_tablegen(A5VMTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=a5vm)
mlir_tablegen(A5VMTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=a5vm)
add_mlir_dialect_library(A5VMIR ...)
```
Source: official MLIR dialect/TableGen docs and the repo’s existing [`include/PTO/IR/CMakeLists.txt`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/CMakeLists.txt)

### Pattern 2: Backend Selection at the Tool Boundary
**What:** Put backend selection in [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp), close to the current `createEmitPTOManualPass(...)` and `emitc::translateToCpp(...)` branch.
**When to use:** For Phase 1 dual-backend coexistence.
**Example:**
```c++
enum class PTOBackend { EmitC, A5VM };

if (backend == PTOBackend::EmitC) {
  pm.addPass(pto::createEmitPTOManualPass(targetArch));
  pm.addPass(emitc::createFormExpressionsPass());
  runEmitCTranslation(module, outputFile);
} else {
  pm.addPass(pto::createLowerPTOToA5VMPass(targetArch));
  runA5VMTextEmission(module, outputFile, debugOptions);
}
```
Source: repo integration point in [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp)

### Pattern 3: Keep Shared Dialects Shared
**What:** Lower only hardware-facing PTO ops to `a5vm`; leave `func`, `scf`, `arith`, `memref`, and ordinary control flow in the existing dialect set.
**When to use:** Throughout Phase 1 and later phases.
**Example:**
```text
PTO + func/scf/arith/memref
        |
        | only hardware-facing ops
        v
 func/scf/arith/memref + a5vm
        |
        v
 textual HIVM emission
```
Source: project requirements plus MLIR dialect conversion guidance at https://mlir.llvm.org/docs/DialectConversion/

### Pattern 4: Preserve Semantics as Attributes, Not Strings
**What:** Carry unresolved layout, valid-region, addressing, and variant metadata as typed operands/attrs on `a5vm` ops, then let the emitter synthesize final intrinsic spellings.
**When to use:** For all three Phase 1 ops.
**Example:**
```mlir
%v = a5vm.load %base[%offset]
     {layout = #pto.layout<nd>, valid_shape = [32, 32], domain = "vec"}
     : memref<?xf32>, !a5vm.vec<64xf32>
%r = a5vm.abs %v : !a5vm.vec<64xf32>
a5vm.store %r, %out[%offset]
  {layout = #pto.layout<nd>, valid_shape = [32, 32], domain = "vec"}
  : !a5vm.vec<64xf32>, memref<?xf32>
```
Source: locked phase decisions and PTO semantic references in the CANN headers

### Anti-Patterns to Avoid
- **Embedding `a5vm` logic in `PTOToEmitC.cpp`:** This violates the locked dialect boundary and guarantees Phase 2/3 churn.
- **Hardcoding final HIVM intrinsic names in op names:** The emitter, not the IR, must own final intrinsic spelling synthesis.
- **Encoding unresolved cases as silent defaults:** Locked policy requires explicit unresolved markers or comments.
- **Moving `scf`/`arith` into `a5vm`:** This breaks BACK-02 and increases backend surface unnecessarily.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dialect registration and IR boilerplate | Handwritten parsers/printers for everything | MLIR TableGen dialect/type/op generation | The repo already uses it, and official MLIR docs treat it as the standard path. |
| Backend conversion framework | Custom tree walk that rewrites ops by hand everywhere | MLIR pass + pattern/conversion infrastructure | Easier verification, easier later expansion to more PTO ops. |
| Type legality enforcement | Ad hoc checks spread across emitters and passes | A single verified `A5VMVecType` verifier | Prevents illegal `a5vm` values from existing at all. |
| Golden text assertions | Bespoke shell diffs | `RUN:` + `FileCheck` tests | Existing repo convention, fast, and purpose-built for compiler output. |
| Intrinsic spelling storage | Frozen string literals on ops | Emitter-side name synthesis from op family + type + variant attrs | Matches locked abstraction boundary and avoids redesign when mappings evolve. |

**Key insight:** In this domain, custom shortcuts usually leak semantics across layers. The plan should centralize legality in types, centralize backend selection in `ptoas`, and centralize final intrinsic spelling in the emitter.

## Common Pitfalls

### Pitfall 1: Making `a5vm` Too Semantic
**What goes wrong:** `a5vm` starts mirroring PTO template behavior instead of representing a hardware-facing abstraction boundary.
**Why it happens:** PTO semantic detail is available first, and it is tempting to copy it directly into op names or custom types.
**How to avoid:** Limit Phase 1 `a5vm` to vector type legality plus load/abs/store ops with metadata attrs needed for later emission.
**Warning signs:** Op names such as `a5vm.vabs_f32_nd_rowmajor` or type families multiplying per layout.

### Pitfall 2: Making `a5vm` Too Thin
**What goes wrong:** The dialect becomes only a debug dump, forcing emitter-specific guesses later.
**Why it happens:** Over-optimizing for speed in the first phase.
**How to avoid:** Preserve unresolved-but-required codegen facts as attrs or operands on `a5vm.load` / `a5vm.store`.
**Warning signs:** The emitter cannot explain why it picked a load/store family or has to inspect old PTO ops.

### Pitfall 3: Vector Width Checks Done Too Late
**What goes wrong:** Illegal vectors survive until emission, where failures become opaque.
**Why it happens:** Width logic is implemented in lowering helpers instead of the type verifier.
**How to avoid:** Reject any `!a5vm.vec<...>` whose `element_count * element_bit_width != 2048`.
**Warning signs:** Multiple callers recalculate “256 bytes” instead of asking the type.

### Pitfall 4: Reusing EmitC Output Assumptions
**What goes wrong:** The new backend still depends on `emitc::translateToCpp` cleanup patterns or marker rewrites.
**Why it happens:** The current tool already has post-processing helpers for EmitC C++.
**How to avoid:** Keep A5VM text emission on a separate path that writes final textual output directly.
**Warning signs:** New code calls EmitC form-expression or C++ rewrite helpers from the A5VM branch.

### Pitfall 5: Losing PTO Layout/Valid-Region Context
**What goes wrong:** `TLOAD` / `TSTORE` become sample-only stubs that cannot grow past `Abs`.
**Why it happens:** The current sample is simple, so shape/layout metadata looks optional.
**How to avoid:** Plan Phase 1 ops with explicit fields for layout, valid rows/cols, and backend-facing address operands even if Phase 1 uses only one shape.
**Warning signs:** `a5vm.load` and `a5vm.store` have only a base pointer and a vector result/value.

### Pitfall 6: Underplanning Diagnostics
**What goes wrong:** Unresolved mappings fail with generic “unsupported” errors and no artifact for later review.
**Why it happens:** Diagnostics feel secondary during bring-up.
**How to avoid:** Include a summary collector and file emission in the initial design.
**Warning signs:** Error paths only print the op name without type/layout/mapping detail.

## Code Examples

Verified patterns from official sources and current repo conventions:

### A5VM Vector Type Verifier
```c++
static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                            Type elementType, int64_t elementCount) {
  auto intOrFloat = dyn_cast<IntegerType, FloatType>(elementType);
  if (!intOrFloat)
    return emitError() << "requires integer or float element type";
  const unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || elementCount <= 0)
    return emitError() << "requires positive element width/count";
  if (static_cast<int64_t>(bitWidth) * elementCount != 2048)
    return emitError() << "requires total width of exactly 2048 bits (256 bytes)";
  return success();
}
```
Source: MLIR attributes/types verifier pattern at https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/

### Minimal Backend Split in `ptoas`
```c++
if (backend == "emitc") {
  pm.addPass(pto::createEmitPTOManualPass(selectedArch));
  pm.addPass(emitc::createFormExpressionsPass());
  return emitEmitC(*module, outputFile);
}

pm.addPass(pto::createLowerPTOToA5VMPass(selectedArch));
if (failed(pm.run(*module)))
  return fail();
return pto::emitA5VMText(*module, outputFile.os(), debugOptions);
```
Source: repo pattern from [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp)

### Phase-Appropriate `a5vm.abs`
```tablegen
def A5VM_AbsOp : A5VM_Op<"abs"> {
  let arguments = (ins A5VM_VectorType:$src);
  let results = (outs A5VM_VectorType:$result);
  let assemblyFormat = "$src attr-dict `:` type($src)";
  let hasVerifier = 1;
}
```
Source: MLIR operation-definition pattern at https://mlir.llvm.org/docs/DefiningDialects/Operations/

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PTO -> EmitC -> C++ marker rewrites | PTO -> A5VM -> textual HIVM emission | Project roadmap created 2026-03-18 | Phase planning must treat `emitc` as legacy compatibility, not as the new backend substrate. |
| Backend semantics embedded in one large conversion pass | First-class dialect boundary plus dedicated emitter | Standard MLIR out-of-tree practice in current MLIR docs | Better isolation, testability, and extensibility. |
| Output-specific naming encoded early | Final names synthesized late from op/type/variant metadata | Locked in CONTEXT.md for this phase | Keeps `a5vm` stable while intrinsic mappings evolve. |

**Deprecated/outdated:**
- Extending only `PTOToEmitC.cpp` for the new backend: outdated for this project because it conflicts with the locked phase boundary.
- Using `emitc::translateToCpp` for the new backend path: outdated for the A5VM branch because the output target is textual LLVM-style HIVM, not C++.

## Open Questions

1. **How thin can Phase 1 textual emission be without stepping on Phase 3?**
   - What we know: CONTEXT.md explicitly requires the thinnest usable textual HIVM path in Phase 1.
   - What's unclear: Whether the plan should include real textual emission for only manually-constructed `a5vm` test inputs or also partial PTO-to-A5VM plumbing for `Abs`.
   - Recommendation: Plan Phase 1 to include direct `a5vm` textual emission plus backend selection and diagnostics; keep PTO semantic lowering itself in Phase 2.

2. **What exact operands/attrs should `a5vm.load` and `a5vm.store` carry?**
   - What we know: CANN `TLoad`/`TStore` behavior depends on layout, valid region, and address-space/domain details.
   - What's unclear: The minimum attribute set that avoids rework while not overfitting to full PTO semantics too early.
   - Recommendation: Plan an explicit metadata struct for layout, valid rows/cols, and source/destination domain, but defer full PTO shape decomposition to Phase 2.

3. **How should unresolved mapping summaries be persisted?**
   - What we know: Locked decisions require both terminal/log output and a separate artifact or file.
   - What's unclear: Whether the artifact should always be produced or only with a debug flag.
   - Recommendation: Always support an explicit `--a5vm-unresolved-report=<path>` flag and optionally default to a sidecar file when the backend is selected.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | lit/FileCheck-style source tests plus `ctest` smoke checks |
| Config file | none committed in source tree; current lit config appears to be external or build-generated |
| Quick run command | `./build/tools/ptoas/ptoas test/basic/empty_func.mlir | FileCheck test/basic/empty_func.mlir` |
| Full suite command | `ctest --test-dir build --output-on-failure` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACK-01 | Backend flag selects new A5VM path at the existing emission boundary | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase1/a5vm_backend_switch.mlir -o - | FileCheck test/phase1/a5vm_backend_switch.mlir` | ❌ Wave 0 |
| BACK-02 | Shared `scf`/`arith` IR survives while only hardware-facing ops enter `a5vm` | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase1/a5vm_shared_dialects.mlir -o - | FileCheck test/phase1/a5vm_shared_dialects.mlir` | ❌ Wave 0 |
| A5VM-01 | `!a5vm.vec<...>` accepts only 256-byte vectors | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir` | ❌ Wave 0 |
| A5VM-02 | `a5vm.load` returns a legal `a5vm` vector value | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_load_op.mlir -o - | FileCheck test/phase1/a5vm_load_op.mlir` | ❌ Wave 0 |
| A5VM-03 | `a5vm.abs` preserves legal vector typing | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_abs_op.mlir -o - | FileCheck test/phase1/a5vm_abs_op.mlir` | ❌ Wave 0 |
| A5VM-04 | `a5vm.store` consumes a legal vector plus backend-facing address metadata | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_store_op.mlir -o - | FileCheck test/phase1/a5vm_store_op.mlir` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** targeted `ptoas` + `FileCheck` test for the touched behavior
- **Per wave merge:** `ctest --test-dir build --output-on-failure`
- **Phase gate:** all new Phase 1 lit-style tests green plus no regression in existing `ctest` smoke checks

### Wave 0 Gaps
- [ ] `test/phase1/a5vm_vec_type.mlir` — legal/illegal type parsing and verifier coverage for A5VM-01
- [ ] `test/phase1/a5vm_load_op.mlir` — op assembly, printing, and verifier coverage for A5VM-02
- [ ] `test/phase1/a5vm_abs_op.mlir` — unary op verifier and print/parse coverage for A5VM-03
- [ ] `test/phase1/a5vm_store_op.mlir` — store op verifier and metadata coverage for A5VM-04
- [ ] `test/phase1/a5vm_backend_switch.mlir` — backend selection and text emission path for BACK-01
- [ ] `test/phase1/a5vm_shared_dialects.mlir` — shared-dialect preservation for BACK-02
- [ ] A committed lit configuration or documented invocation path for source tests — current repo has `RUN:` tests but no committed `lit.cfg.py`

## Sources

### Primary (HIGH confidence)
- Local repo sources:
  - [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp) - current registry, pass pipeline, arch selection, and `emitc` emission boundary
  - [`include/PTO/Transforms/Passes.h`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/Transforms/Passes.h) - pass registration surface
  - [`include/PTO/IR/CMakeLists.txt`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/CMakeLists.txt) and [`lib/PTO/IR/CMakeLists.txt`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/IR/CMakeLists.txt) - current dialect/TableGen build pattern
  - [`include/PTO/IR/PTOOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/PTOOps.td) - current `TLoad`, `TStore`, `TAbs` op semantics and verifier patterns
- Local toolchain metadata:
  - `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/llvm/LLVMConfigVersion.cmake` - LLVM `19.1.7`
  - `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/mlir/MLIRConfigVersion.cmake` - MLIR `19.1.7`
- Official MLIR docs:
  - https://mlir.llvm.org/docs/DefiningDialects/
  - https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/
  - https://mlir.llvm.org/docs/DefiningDialects/Operations/
  - https://mlir.llvm.org/docs/DialectConversion/
- Ascend semantic references:
  - `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp`
  - `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp`
  - `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp`
  - `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp`
  - `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`

### Secondary (MEDIUM confidence)
- https://llvm.org/docs/CommandGuide/FileCheck.html - testing command behavior consistent with repo `RUN:` usage

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - verified from local toolchain metadata, repo build files, and official MLIR docs
- Architecture: HIGH - based on current repo integration points and standard MLIR out-of-tree patterns
- Pitfalls: HIGH - grounded in locked phase decisions plus current PTO/CANN semantic references

**Research date:** 2026-03-18
**Valid until:** 2026-04-17
