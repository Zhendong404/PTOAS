## Why

The VPTO backend has moved toward the PR348 tilebuf-native PTO pipeline, but
the backend lowering path still carries a legacy memref-to-tilebuf bridge before
`ExpandTileOp`. That bridge contradicts the intended pipeline boundary: before
`ExpandTileOp` the IR is PTO IR and local tile state should be represented as
`!pto.tile_buf`, while after `ExpandTileOp` the IR is VPTO authoring IR and may
materialize memref views for TileLang helper bodies.

Keeping `MemrefToTileBuf` around keeps the old memref-world contract alive and
forces `FoldTileBufIntrinsics` to depend on a synthetic
`bind_tile -> unrealized_conversion_cast` pattern. The backend should instead
make `ExpandTileOp` the explicit PTO-to-VPTO boundary, make
`FoldTileBufIntrinsics` resolve native tilebuf producers directly, and remove
the obsolete bridge pass entirely.

## What Changes

- Define `ExpandTileOp` as the boundary between PTO IR and VPTO IR.
- Require the IR before `ExpandTileOp` to be tilebuf-native and not depend on
  memref-local tile buffers.
- Remove `MemrefToTileBuf` registration, factory declarations, build wiring,
  and implementation.
- Adapt `FoldTileBufIntrinsics` to resolve tilebuf address and valid-shape
  metadata from native tilebuf producers.
- Delete the old `findBindTileForTileBuf()` assumption instead of keeping it as
  a legacy fallback.
- Keep memref materialization legal after `ExpandTileOp`, where TileLang
  templates and VPTO authoring IR naturally use memref/ptr/index forms.
- Add regression coverage for the boundary contract and for native tilebuf
  intrinsic folding.

## Capabilities

### New Capabilities

- `vpto-expand-tile-boundary`: Establishes `ExpandTileOp` as the PTO-to-VPTO
  boundary and removes the legacy pre-boundary memref-to-tilebuf bridge from
  the VPTO backend path.

### Modified Capabilities

- `vpto-tilebuf-pipeline-integration`: Narrows the broad PR348 integration
  contract by specifying exactly where memref is allowed to appear in the VPTO
  backend pipeline.

## Impact

- Affected pipeline code: `tools/ptoas/ptoas.cpp`.
- Affected transforms: `ExpandTileOp`, `FoldTileBufIntrinsics`, and the
  surrounding VPTO emission preparation passes that consume their output.
- Affected legacy bridge code: `MemrefToTileBuf` is removed rather than kept as
  an explicit legacy/import workflow.
- Affected IR contract:
  - Before `ExpandTileOp`: PTO IR with native `!pto.tile_buf` local tiles.
  - At `ExpandTileOp`: tile-level PTO ops are replaced by TileLang calls.
  - After `ExpandTileOp`: VPTO authoring IR may contain memref, ptr, and
    structured-view intrinsic materialization.
- Affected validation: focused lit tests for pass scheduling, memref rejection
  before the boundary, native tilebuf intrinsic folding, and VPTO output that no
  longer relies on `MemrefToTileBuf`.
