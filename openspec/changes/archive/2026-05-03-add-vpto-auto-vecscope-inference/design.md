## Context

VPTO already models vector execution intervals with `pto.vecscope` and `pto.strict_vecscope`. The emission path later materializes dedicated scope ops into dummy carrier loops and attaches LLVM loop metadata before exporting LLVM IR.

The current VPTO legality validator rejects operations that consume or produce `!pto.vreg`, `!pto.mask`, or `!pto.align` unless they are enclosed by one of these scope carriers. Issue #122 asks to move scope inference from the Python DSL layer into an MLIR backend pass so the inference can use the post-inline, post-canonicalize SSA form.

## Goals / Non-Goals

**Goals:**

- Infer missing `pto.vecscope` regions for unscoped VPTO vector operation sequences.
- Run inference at the emission boundary, after existing cleanup, pointer normalization, bridge-op expansion, canonicalization, and CSE.
- Keep explicit scope constructs stable and never generate nested vector scopes.
- Keep the pass conservative when operation motion could expose semantic ambiguity.
- Produce diagnostics before LLVM emission when vector-scope values escape the inferred region.

**Non-Goals:**

- Extending `pto.vecscope` to return values.
- Adding a `pto.yield` operation.
- Inferring scopes across DMA, sync, copy, unresolved function calls, terminators, or verifier-forbidden VPTO operations.
- Reordering operations for scheduling quality beyond moving a contiguous safe cluster into a region.
- Replacing `pto.strict_vecscope`; strict explicit-capture semantics remain author-controlled.

## Decisions

### Decision: Run directly before emission validation

Schedule `func.func(pto-infer-vpto-vecscope)` inside `prepareVPTOForEmission()` after `PTOVPTOExpandBridgeOps` and CSE, followed by canonicalize/CSE and then `PTOValidateVPTOEmissionIR`.

This places the pass as close as possible to EmitLLVM while still letting the existing validator be the final contract checker.

### Decision: Keep `pto.vecscope` resultless

The pass SHALL NOT introduce `pto.yield` or result-bearing `pto.vecscope`. Inputs where `!pto.vreg`, `!pto.mask`, or `!pto.align` values have users outside the inferred scope are invalid for automatic inference and fail with a diagnostic.

This matches the current dialect implementation and the project decision that vecscope-internal vector data has no external users.

### Decision: Use greedy block-local clustering

For each function block, scan operations in order and collect a pending cluster until a boundary is reached. If the cluster contains at least one vector operation, wrap the contiguous cluster in `pto.vecscope`; otherwise discard it without emitting an empty scope.

Scalar operations may be included only when they are contiguous, safe to move into a no-result region, and have no uses outside the inferred cluster.

### Decision: Treat nested control flow as an atomic candidate when safe

`scf.if` and `scf.for` may enter an outer cluster as a single operation if their nested regions contain vector operations and no forbidden boundaries. If their nested regions contain forbidden boundaries, the outer cluster is flushed and nested blocks are scanned independently.

This keeps common TileLang loop shapes compact while avoiding accidental scope inference across DMA/call/sync boundaries inside structured control flow.

### Decision: Preserve explicit vector scope carriers

The pass skips the bodies of existing `pto.vecscope` and `pto.strict_vecscope`. These carriers already satisfy VPTO legality and should not be nested or rewritten by automatic inference.

### Decision: Remove DSL frontend implicit VecScope inference

TileLang DSL semantic analysis no longer wraps vector-looking statement runs in
implicit `SemanticVecscopeStmt`. The DSL still lowers explicit `with
pto.vecscope():` and `with pto.strict_vecscope(...):` constructs, but ordinary
DSL vector surface now emits straightforward unscoped VPTO authoring IR and
relies on `pto-infer-vpto-vecscope` during backend emission preparation.

## Classification Rules

- **Vector operation**: a PTO operation whose operands or results include `!pto.vreg`, `!pto.mask`, or `!pto.align`, plus known control-sequence operations that the verifier requires to be nested in a vecscope.
- **Safe scalar operation**: a pure scalar/address/view operation that can be moved into a no-result region without creating external result users.
- **Boundary operation**: DMA/copy/sync operations, `func.call`, terminators, existing vector-scope carriers, operations explicitly forbidden inside vecscope, or any operation whose movement would invalidate external SSA uses.

## Risks / Trade-offs

- Conservative boundaries may create more scopes than a future scheduler would prefer, but keep v1 correctness clear.
- Rejecting escaping vector values limits unusual authoring patterns, but avoids dialect/result extensions not needed by the confirmed use case.
- Classifier drift is possible as new VPTO ops are added. The implementation should centralize classification helpers and share logic style with `PTOValidateVPTOIR.cpp` where practical.

## Rollout Plan

1. Add pass registration and the pass implementation.
2. Remove DSL frontend implicit vecscope inference while preserving explicit scope syntax.
3. Add focused pass-pipeline tests for inference behavior.
4. Wire the pass into VPTO emission preparation.
5. Add backend smoke coverage with `--emit-vpto` and at least one `--vpto-emit-hivm-llvm` test.
6. Run existing `test/basic` and `test/vpto` lit suites.

Rollback is simple: remove the pass from `prepareVPTOForEmission()` and restore
DSL semantic-stage inference while leaving explicit `pto.vecscope` behavior
unchanged.
