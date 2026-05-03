## Context

The main PTOAS pipeline now keeps local tile state in tilebuf-native form. In
that world, `PlanMemory`, reserved-buffer resolution, and optional `InsertSync`
operate before VPTO backend lowering and should not require the old
`PTOViewToMemref`/memref-local tile bridge.

The VPTO backend currently still schedules `MemrefToTileBuf` immediately before
`ExpandTileOp`. That creates a contradictory contract:

- the pre-backend PTO pipeline claims tilebuf-native IR,
- the VPTO backend still expects memref-shaped local tile input,
- `MemrefToTileBuf` recreates `!pto.tile_buf` using
  `unrealized_conversion_cast`,
- `FoldTileBufIntrinsics` then depends on the exact synthetic bridge shape.

The desired model is simpler:

```text
PTO main pipeline
  tilebuf-native PTO IR
  no memref local tile representation

ExpandTileOp
  explicit PTO IR -> VPTO IR boundary

VPTO backend IR
  TileLang helper calls/bodies
  memref/ptr/index materialization is allowed and expected
```

## Goals / Non-Goals

**Goals:**

- Make `ExpandTileOp` the explicit boundary where PTO tile operations enter
  VPTO authoring IR.
- Ensure the default VPTO backend no longer schedules `MemrefToTileBuf` before
  `ExpandTileOp`.
- Make `FoldTileBufIntrinsics` resolve native tilebuf producers directly.
- Delete `findBindTileForTileBuf()` and the
  `bind_tile -> unrealized_conversion_cast` contract rather than preserving it
  as a fallback.
- Delete the `MemrefToTileBuf` pass implementation, pass definition, factory
  declaration, and build wiring.
- Preserve the legality of memref in VPTO authoring/emission preparation after
  `ExpandTileOp`.
- Add targeted tests that fail if the legacy bridge re-enters the default VPTO
  path.

**Non-Goals:**

- Banning memref from VPTO IR after `ExpandTileOp`.
- Removing every memref-compatible ODS form in one change.
- Rewriting the full VPTO pointer-normalization or LLVM-emission pipeline
  beyond what is needed to consume the new post-boundary IR.
- Changing TileLang DSL template semantics beyond the new boundary contract.

## Decisions

### Decision: `ExpandTileOp` is the PTO-to-VPTO boundary

The default VPTO backend lowering pipeline SHALL enter `ExpandTileOp` with
tilebuf-native PTO IR. Tile-level operations consumed by `ExpandTileOp` should
see `!pto.tile_buf` operands, not memref operands reconstructed by a prepass.

After `ExpandTileOp`, the pipeline is in VPTO authoring form. TileLang helpers
may expose memref views and VPTO memory operations may consume memref-like
authoring operands until later pointer-boundary canonicalization.

### Decision: Remove `MemrefToTileBuf` entirely

The default `lowerPTOToVPTOBackend` sequence should begin with `ExpandTileOp`.
`MemrefToTileBuf` should not remain registered as a manually scheduled
legacy/import pass because keeping it available preserves the obsolete
memref-to-tilebuf contract and weakens the new boundary invariant.

### Decision: Delete `findBindTileForTileBuf()` and its bridge pattern

`FoldTileBufIntrinsics` should not retain a legacy fallback for the synthetic
memref-to-tilebuf pattern:

```mlir
%bound = pto.bind_tile %src, %vrow, %vcol : memref -> memref
%tile = builtin.unrealized_conversion_cast %bound : memref -> !pto.tile_buf
```

The helper that enforces this pattern should be removed. If such IR reaches
the default VPTO path, that is a pipeline contract violation rather than a
supported compatibility mode.

### Decision: Resolve tilebuf metadata from native producers

`FoldTileBufIntrinsics` should resolve `pto.tile_buf_addr`,
`pto.tile_valid_rows`, and `pto.tile_valid_cols` from native tilebuf values.
The implementation should centralize this in a native metadata resolver rather
than scattering producer-specific logic through each intrinsic rewrite.

The resolver should support the native producers that are valid at the
PTO-to-VPTO boundary, including:

- `pto.pointer_cast` for planned local addresses and explicit address binding,
- `pto.alloc_tile` for level3/manual-address inputs where the address is still
  carried directly,
- `pto.bind_tile` as a tilebuf-to-tilebuf metadata operation,
- tile alias/view operations such as `pto.subview`, `pto.bitcast`, and
  `pto.treshape` when their address and valid-shape metadata can be traced to
  a native source,
- block arguments only when their address and dynamic valid-shape metadata are
  represented by the accepted native contract.

If a dynamic valid dimension cannot be traced to native metadata, the pass
should emit an actionable error instead of silently assuming the full static
shape.

### Decision: Keep memref materialization on the VPTO side

`FoldTileBufIntrinsics` is allowed to materialize memref, ptr, index constants,
and index arithmetic after `ExpandTileOp`. This is not a regression to the old
pre-boundary memref pipeline; it is the intended VPTO authoring representation.

The important invariant is where memref first appears:

- before `ExpandTileOp`: no memref local tile representation,
- after `ExpandTileOp`: memref/ptr materialization is legal.

## Implementation Sketch

1. Update `tools/ptoas/ptoas.cpp`:
   - remove `backendPM.addPass(pto::createMemrefToTileBufPass())`,
   - update the VPTO backend lowering comment to describe the boundary.

2. Update `ExpandTileOp`:
   - assert/diagnose that tile operands reaching tile-op expansion use
     `TileBufType`,
   - report memref tile operands as a pre-boundary contract violation.

3. Refactor `FoldTileBufIntrinsics`:
   - delete `findBindTileForTileBuf()`,
   - introduce a native tile metadata resolver,
   - rewrite `tile_buf_addr` folding to materialize memref/ptr from native
     address sources,
   - rewrite `tile_valid_rows/cols` folding to use static `TileBufType`
     valid-shape metadata or native dynamic valid-shape operands,
   - keep tensor-view intrinsic handling intact unless it conflicts with the
     new tilebuf boundary contract.

4. Update pass documentation:
   - `ExpandTileOp` describes the PTO-to-VPTO boundary,
   - `FoldTileBufIntrinsics` describes VPTO-side materialization,
   - `MemrefToTileBuf` documentation and registration are removed.

5. Add tests:
   - default VPTO backend scheduling does not include `MemrefToTileBuf`,
   - native tilebuf input expands without pre-boundary memref,
   - `FoldTileBufIntrinsics` handles native `pointer_cast`/`alloc_tile`
     sources,
   - dynamic valid rows/cols are resolved from native metadata,
   - old synthetic bridge shape is not accepted by the default path.

## Risks / Trade-offs

- Native tile alias chains may expose metadata gaps previously hidden by memref
  lowering. The resolver should fail loudly for unsupported dynamic cases and
  tests should cover the common alias producers first.
- Some legacy hand-written VPTO tests may depend on the old synthetic bridge.
  Those tests should be migrated to the native contract or removed.
- Pointer-boundary canonicalization may still contain memref compatibility
  logic. That is acceptable after `ExpandTileOp`, but tests should ensure it is
  not used to justify pre-boundary memref input.

## Open Questions

- Should level3 explicit-address `pto.alloc_tile` be canonicalized to
  `pto.pointer_cast` before `ExpandTileOp`, or should `FoldTileBufIntrinsics`
  consume both forms permanently?
- Which tile alias producers must be P0 for the first implementation:
  `bind_tile`, `subview`, `bitcast`, `treshape`, or all of them?
