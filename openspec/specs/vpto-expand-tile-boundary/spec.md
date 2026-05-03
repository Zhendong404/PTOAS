# vpto-expand-tile-boundary Specification

## Purpose
TBD - created by archiving change define-vpto-expand-tile-boundary. Update Purpose after archive.
## Requirements
### Requirement: ExpandTileOp defines the PTO-to-VPTO boundary

The default VPTO backend lowering pipeline SHALL treat `ExpandTileOp` as the
boundary between tilebuf-native PTO IR and VPTO authoring IR.

#### Scenario: PTO IR reaches ExpandTileOp in tilebuf-native form

- **WHEN** a PTO module is compiled with `--pto-backend=vpto`
- **THEN** the default backend lowering pipeline runs `ExpandTileOp` before
  TileLang inlining and tilebuf intrinsic folding
- **AND** local tile values reaching `ExpandTileOp` are represented as
  `!pto.tile_buf`
- **AND** the pipeline does not require memref-local tile representation before
  `ExpandTileOp`

#### Scenario: VPTO IR after ExpandTileOp may contain memref

- **WHEN** `ExpandTileOp` replaces a tile operation with a TileLang helper call
- **THEN** subsequent VPTO authoring IR may contain memref, ptr, and index
  values needed by TileLang helper bodies
- **AND** such memref values are considered post-boundary VPTO materialization,
  not pre-boundary PTO local tile state

### Requirement: MemrefToTileBuf is removed

The obsolete `MemrefToTileBuf` bridge MUST NOT remain available as a registered
pass, factory entry point, or compiled transform implementation.

#### Scenario: Transform pass registry is built

- **WHEN** PTOAS transform passes are built and registered
- **THEN** no `MemrefToTileBuf` pass definition or factory is present
- **AND** no `pto-memref-to-tile-buf` command-line pass is registered

#### Scenario: Backend pass sequence is constructed

- **WHEN** the VPTO backend pass manager is created for a non-VPTO input module
- **THEN** the first tile-op expansion stage is `createExpandTileOpPass()`
- **AND** the documented pass sequence does not describe memref recovery before
  tile expansion

### Requirement: ExpandTileOp rejects pre-boundary memref tile operands

`ExpandTileOp` SHALL diagnose tile operations whose tile operands are memref
values instead of native `!pto.tile_buf` values.

#### Scenario: Memref tile operand reaches ExpandTileOp

- **WHEN** a tile operation that should consume a local tile reaches
  `ExpandTileOp` with a memref operand
- **THEN** `ExpandTileOp` fails with a diagnostic explaining that memref local
  tile operands violate the PTO-to-VPTO boundary contract
- **AND** the diagnostic directs the producer to provide tilebuf-native PTO IR
  before backend lowering

### Requirement: FoldTileBufIntrinsics resolves native tilebuf metadata

`FoldTileBufIntrinsics` SHALL resolve `pto.tile_buf_addr`,
`pto.tile_valid_rows`, and `pto.tile_valid_cols` from native tilebuf producers
instead of requiring a synthetic memref-to-tilebuf bridge.

#### Scenario: Native pointer-cast tile folds

- **WHEN** a TileLang helper body uses `pto.tile_buf_addr` or
  `pto.tile_valid_*` on a tile value produced by native tilebuf address
  materialization such as `pto.pointer_cast`
- **THEN** `FoldTileBufIntrinsics` replaces address intrinsics with the
  requested memref or `!pto.ptr` VPTO-side value
- **AND** static valid dimensions fold to `arith.constant`
- **AND** dynamic valid dimensions fold to the native valid-row or valid-col
  SSA operands

#### Scenario: Native alloc-tile explicit address folds

- **WHEN** a level3/manual-address tile value still reaches folding as
  explicit-address `pto.alloc_tile`
- **THEN** `FoldTileBufIntrinsics` either resolves the explicit address
  directly or relies on a documented canonicalization to an equivalent native
  address representation before folding
- **AND** the resulting VPTO IR does not require `MemrefToTileBuf`

#### Scenario: Native tile metadata op folds

- **WHEN** a tile value is produced by a native tilebuf-to-tilebuf metadata op
  such as `pto.bind_tile`
- **THEN** `FoldTileBufIntrinsics` traces the address through the source tile
- **AND** it resolves valid-row and valid-col metadata from the native metadata
  op or from static `TileBufType` metadata

### Requirement: Synthetic bind-tile cast bridge is not supported in the default path

`FoldTileBufIntrinsics` MUST delete the old `findBindTileForTileBuf()` helper
and MUST NOT keep a legacy fallback for the synthetic pattern
`bind_tile -> unrealized_conversion_cast -> tile_buf`.

#### Scenario: Old synthetic bridge reaches default VPTO folding

- **WHEN** default VPTO backend lowering reaches `FoldTileBufIntrinsics`
- **AND** a tilebuf intrinsic depends on the old synthetic
  `bind_tile -> unrealized_conversion_cast` bridge shape
- **THEN** the pass does not silently accept that shape as a supported fallback
- **AND** the pipeline fails with a diagnostic that identifies the IR as
  violating the native tilebuf boundary contract

### Requirement: Dynamic valid dimensions require native metadata

Dynamic valid-row and valid-col values SHALL be folded only when they can be
traced to native tilebuf metadata.

#### Scenario: Dynamic valid row is available

- **WHEN** a tile type has dynamic valid row metadata
- **AND** the native tile producer carries a valid-row SSA operand
- **THEN** `pto.tile_valid_rows` folds to that operand

#### Scenario: Dynamic valid col is missing

- **WHEN** a tile type has dynamic valid col metadata
- **AND** no native tile producer in the traced chain carries a valid-col SSA
  operand
- **THEN** `FoldTileBufIntrinsics` fails with an actionable diagnostic
- **AND** it does not substitute the static tile shape or another guessed value

### Requirement: Tests cover the boundary contract

The change SHALL add focused regression coverage for the new boundary and
native intrinsic folding behavior.

#### Scenario: Default backend remains bridge-free before ExpandTileOp

- **WHEN** the focused VPTO pipeline tests are run
- **THEN** at least one test verifies the default backend path succeeds without
  scheduling `MemrefToTileBuf`
- **AND** at least one test fails if pre-boundary memref local tile operands are
  accepted by `ExpandTileOp`

#### Scenario: Native tilebuf intrinsics fold

- **WHEN** the focused `FoldTileBufIntrinsics` tests are run
- **THEN** they cover native address materialization and valid-shape folding
- **AND** they cover an error path for unsupported dynamic valid metadata

