## Why

Issue [#122](https://github.com/mouliangyu/PTOAS/issues/122) requests automatic `pto.vecscope` inference in the vPTO backend. Today, TileLang and hand-authored VPTO IR often need explicit `pto.vecscope` regions around vector operations, and frontend-side inference is brittle because it runs before backend inlining, canonicalization, CSE, and pointer normalization have exposed the final SSA shape.

The vPTO backend should infer vector execution scopes immediately before LLVM/HIVM emission so frontend DSL authors can emit straightforward VPTO operation sequences without manually wrapping every vector interval.

## What Changes

- Add a vPTO auto VecScope inference pass, exposed as `pto-infer-vpto-vecscope`.
- Schedule the pass in the VPTO emission-preparation pipeline, directly before `pto-validate-vpto-emission-ir` and therefore immediately before `translateVPTOModuleToLLVMText` / `translateVPTOModuleToLLVMBitcode`.
- Remove TileLang DSL semantic-stage implicit `pto.vecscope` inference; the DSL still supports explicit `pto.vecscope()` / `pto.strict_vecscope()` authoring, but automatic insertion belongs to the VPTO backend.
- Infer `pto.vecscope` regions by greedily clustering contiguous vector-related operations inside each function block after inline/canonicalize/CSE, ptr normalization, and bridge-op expansion.
- Treat DMA/copy/sync operations, unresolved `func.call`, terminators, and explicitly forbidden VPTO ops as cluster boundaries.
- Preserve existing explicit `pto.vecscope` and `pto.strict_vecscope` carriers without nesting new scopes inside them.
- Do not add `pto.yield` or result-bearing `pto.vecscope` syntax. Vector-scope data (`!pto.vreg`, `!pto.mask`, `!pto.align`) MUST NOT have users outside the inferred scope; violating inputs fail with a clear diagnostic.

## Capabilities

### New Capabilities

- `vpto-vecscope-inference`: Covers the backend pass that infers missing VPTO vector scope regions before LLVM/HIVM emission.

### Modified Capabilities

- None.

## Impact

- Affected issue: `mouliangyu/PTOAS#122`.
- Affected pipeline code: `tools/ptoas/ptoas.cpp`, PTO pass registration, and VPTO emission-preparation passes.
- Affected transform code: new pass implementation under `lib/PTO/Transforms`.
- Affected DSL code: TileLang DSL no longer emits implicit `pto.vecscope` from semantic analysis.
- Affected IR contract: unscoped VPTO vector operation sequences become legal backend input when every vector value is consumed inside the inferred scope.
- Affected validation: focused lit coverage for inferred scope insertion, forbidden boundaries, nested control flow, existing explicit scopes, escaping vector values, and VPTO LLVM emission smoke tests.
