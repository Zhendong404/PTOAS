## ADDED Requirements

### Requirement: FoldTileBufIntrinsics accepts native descriptor inputs only

`FoldTileBufIntrinsics` SHALL fold TileLang helper intrinsics only when their
source operands are native PTO descriptors.

#### Scenario: Tile-buffer intrinsic source is native tile_buf

- **WHEN** `pto.tile_buf_addr`, `pto.tile_valid_rows`, or
  `pto.tile_valid_cols` is folded
- **THEN** the source operand is `!pto.tile_buf`
- **AND** its address and dynamic valid-shape metadata are resolved from native
  tile-buffer producers

#### Scenario: Tensor-view intrinsic source is native tensor-view-like

- **WHEN** `pto.tensor_view_addr`, `pto.get_tensor_view_dim`, or
  `pto.get_tensor_view_stride` is folded
- **THEN** the source operand is `!pto.tensor_view` or
  `!pto.partition_tensor_view`
- **AND** its metadata is resolved from native PTO view producers

#### Scenario: Memref bridge source is rejected

- **WHEN** an intrinsic source depends on a memref-sourced
  `builtin.unrealized_conversion_cast`
- **THEN** `FoldTileBufIntrinsics` fails with a diagnostic that identifies the
  memref bridge as unsupported
- **AND** it does not trace through `memref.subview` or
  `memref.reinterpret_cast`

### Requirement: tensor_view_addr source contract excludes memref

`pto.tensor_view_addr` SHALL accept only `!pto.tensor_view` or
`!pto.partition_tensor_view` source operands.

#### Scenario: tensor_view_addr is parsed with tensor_view source

- **WHEN** the source operand type is `!pto.tensor_view`
- **THEN** verification accepts the source type when the result element type,
  rank, and GM memory-space contract are satisfied

#### Scenario: tensor_view_addr is parsed with partition_tensor_view source

- **WHEN** the source operand type is `!pto.partition_tensor_view`
- **THEN** verification accepts the source type when the result element type,
  rank, and GM memory-space contract are satisfied

#### Scenario: tensor_view_addr is parsed with memref source

- **WHEN** the source operand type is memref
- **THEN** parsing or verification rejects the operation
- **AND** the diagnostic names `!pto.tensor_view` and
  `!pto.partition_tensor_view` as the accepted source forms

### Requirement: Native make_tensor_view folds tensor-view intrinsics

`FoldTileBufIntrinsics` SHALL resolve tensor-view intrinsics from
`pto.make_tensor_view` without requiring memref view operations.

#### Scenario: tensor_view_addr folds from make_tensor_view

- **WHEN** `pto.tensor_view_addr` consumes a value produced by
  `pto.make_tensor_view`
- **THEN** the address intrinsic is replaced by the make-tensor-view base
  pointer or by a result materialized from that pointer
- **AND** no memref view-chain tracing is required

#### Scenario: get_tensor_view_dim folds from make_tensor_view

- **WHEN** `pto.get_tensor_view_dim` consumes a value produced by
  `pto.make_tensor_view`
- **AND** the requested dimension is static in the tensor-view type
- **THEN** the result folds to an `arith.constant` index

- **WHEN** the requested dimension is dynamic in the tensor-view type
- **THEN** the result folds to the corresponding `make_tensor_view` shape
  operand

#### Scenario: get_tensor_view_stride folds from make_tensor_view

- **WHEN** `pto.get_tensor_view_stride` consumes a value produced by
  `pto.make_tensor_view`
- **THEN** the result folds to the corresponding native stride operand

### Requirement: Native partition_view folds tensor-view intrinsics

`FoldTileBufIntrinsics` SHALL resolve partition-view intrinsics from
`pto.partition_view` metadata and its native source view.

#### Scenario: tensor_view_addr folds from partition_view

- **WHEN** `pto.tensor_view_addr` consumes a value produced by
  `pto.partition_view`
- **THEN** the pass traces to the source `pto.make_tensor_view` pointer
- **AND** computes an element offset from partition offsets and source strides
- **AND** materializes the requested address result from the base pointer plus
  that offset

#### Scenario: get_tensor_view_dim folds from partition_view

- **WHEN** `pto.get_tensor_view_dim` consumes a value produced by
  `pto.partition_view`
- **AND** the requested dimension is static in the partition-view result type
- **THEN** the result folds to an `arith.constant` index

- **WHEN** the requested dimension is dynamic in the partition-view result type
- **THEN** the result folds to the corresponding `partition_view` size operand

#### Scenario: get_tensor_view_stride folds from partition_view

- **WHEN** `pto.get_tensor_view_stride` consumes a value produced by
  `pto.partition_view`
- **THEN** the result folds to the corresponding source view stride
- **AND** `partition_view` is treated as preserving source logical strides

### Requirement: Native tile-buffer folding rejects old bridge provenance

`FoldTileBufIntrinsics` SHALL reject old memref-to-tilebuf bridge provenance
instead of accepting it as a fallback source for tile-buffer intrinsics.

#### Scenario: tile_buf_addr source is native pointer_cast

- **WHEN** `pto.tile_buf_addr` consumes a tile produced by `pto.pointer_cast`
- **THEN** the pass materializes the requested memref or `!pto.ptr` result from
  the native address metadata

#### Scenario: tile_buf_addr source is native tile alias

- **WHEN** `pto.tile_buf_addr` consumes a tile produced by native tile metadata
  ops such as `pto.bind_tile`, `pto.subview`, `pto.bitcast`, or `pto.treshape`
- **THEN** the pass traces address metadata through the native tile source
- **AND** it applies native tile subview byte offsets when required

#### Scenario: tile_buf_addr source is memref bridge

- **WHEN** `pto.tile_buf_addr` consumes a tile produced by
  `builtin.unrealized_conversion_cast` from memref
- **THEN** the pass fails with an actionable diagnostic
- **AND** it does not recover address metadata from memref producers

### Requirement: Tests cover native descriptor folding

The change SHALL add focused tests for native tensor-view folding and bridge
rejection while preserving existing native tile-buffer folding tests.

#### Scenario: Native tensor-view tests run

- **WHEN** the focused `FoldTileBufIntrinsics` test suite is run
- **THEN** it covers `make_tensor_view` and `partition_view` inputs for
  address, dimension, and stride intrinsics
- **AND** it covers rejection of memref and memref-bridge inputs

#### Scenario: Native tile-buffer tests continue to run

- **WHEN** the focused native tile-buffer folding tests are run
- **THEN** existing `pto.pointer_cast`, explicit-address `pto.alloc_tile`, and
  native tile alias coverage still passes
- **AND** old memref-to-tilebuf bridge provenance is rejected
