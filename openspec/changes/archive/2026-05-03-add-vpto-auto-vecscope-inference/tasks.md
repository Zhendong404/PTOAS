## 1. Pass Registration

- [x] 1.1 Add `PTOInferVPTOVecScope` to `include/PTO/Transforms/Passes.td` as a `func::FuncOp` pass named `pto-infer-vpto-vecscope`.
- [x] 1.2 Add `createPTOInferVPTOVecScopePass()` to `include/PTO/Transforms/Passes.h`.
- [x] 1.3 Add `lib/PTO/Transforms/PTOInferVPTOVecScope.cpp` and include it in `lib/PTO/Transforms/CMakeLists.txt`.
- [x] 1.4 Register the generated pass with the existing PTO pass registration flow so `--pass-pipeline` tests can invoke it.

## 2. Inference Implementation

- [x] 2.1 Implement block-local greedy clustering for unscoped VPTO vector operation sequences.
- [x] 2.2 Centralize operation classification for vector, safe scalar, and boundary operations.
- [x] 2.3 Skip existing `pto.vecscope` and `pto.strict_vecscope` carriers.
- [x] 2.4 Support `scf.if` and `scf.for` as atomic cluster members when their nested regions contain vector ops and no forbidden boundaries.
- [x] 2.5 Reject inferred clusters whose `!pto.vreg`, `!pto.mask`, or `!pto.align` results have users outside the cluster.
- [x] 2.6 Avoid generating empty or scalar-only `pto.vecscope` regions.
- [x] 2.7 Remove TileLang DSL semantic-stage implicit VecScope inference while preserving explicit `pto.vecscope` / `pto.strict_vecscope`.

## 3. Pipeline Integration

- [x] 3.1 Insert `func.func(pto-infer-vpto-vecscope)` in `prepareVPTOForEmission()` after `PTOVPTOExpandBridgeOps` and CSE.
- [x] 3.2 Run canonicalize and CSE after inference before `pto-validate-vpto-emission-ir`.
- [x] 3.3 Confirm the pass runs for both already-VPTO input and TileOp-expanded VPTO backend input.
- [x] 3.4 Ensure explicit `--emit-vpto` output shows inferred scopes before LLVM/HIVM emission.

## 4. Focused Tests

- [x] 4.1 Add a simple unscoped `pset/plt + vlds + vector compute + vsts` test that checks inferred `pto.vecscope`.
- [x] 4.2 Add a scalar-in-cluster test that keeps safe scalar arithmetic inside one inferred scope.
- [x] 4.3 Add a forbidden-boundary test that proves DMA/copy/sync or `func.call` splits scopes.
- [x] 4.4 Add a nested `scf.for` / `scf.if` test for atomic control-flow clustering and recursive fallback.
- [x] 4.5 Add an existing-scope test proving the pass does not nest or rewrite explicit `pto.vecscope` / `pto.strict_vecscope`.
- [x] 4.6 Add a negative escaping-vector-value test with a clear diagnostic.
- [x] 4.7 Add or update a VPTO LLVM emission smoke test using `--vpto-emit-hivm-llvm`.

## 5. Validation

- [x] 5.1 Build `ptoas`.
- [x] 5.2 Run the focused new tests.
- [ ] 5.3 Run `python3 -m lit -sv test/basic`.
- [ ] 5.4 Run `python3 -m lit -sv test/vpto`.
- [x] 5.5 Record any unsupported edge cases as follow-up issues before closing issue #122.
