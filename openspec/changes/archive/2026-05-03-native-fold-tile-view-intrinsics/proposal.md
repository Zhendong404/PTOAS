## Why

`FoldTileBufIntrinsics` still contains legacy assumptions from the removed
`View2Memref` era. The pass can trace tensor-view intrinsics through
`unrealized_conversion_cast -> memref.subview -> memref.reinterpret_cast`, and
its tile-buffer diagnostics can still encounter old memref-to-tilebuf bridge
shapes.

The current VPTO pipeline is descriptor-native before intrinsic folding:
`!pto.tile_buf`, `!pto.tensor_view`, and `!pto.partition_tensor_view` carry the
semantic surface that TileLang helper intrinsics should consume. Keeping memref
chain tracing inside `FoldTileBufIntrinsics` reintroduces a deleted pipeline
model, makes the accepted IR contract ambiguous, and lets old bridge-shaped IR
fail late or unclearly.

## What Changes

- Make `FoldTileBufIntrinsics` a native-descriptor-only fold pass.
- Resolve `pto.tile_buf_addr`, `pto.tile_valid_rows`, and
  `pto.tile_valid_cols` only from native `!pto.tile_buf` producer chains.
- Resolve `pto.tensor_view_addr`, `pto.get_tensor_view_dim`, and
  `pto.get_tensor_view_stride` only from native `pto.make_tensor_view` /
  `pto.partition_view` producer chains.
- Remove all tensor-view folding logic that depends on memref view chains.
- Tighten `pto.tensor_view_addr` so its source operand accepts only
  `!pto.tensor_view` or `!pto.partition_tensor_view`.
- Reject bridge-shaped inputs explicitly, including
  `unrealized_conversion_cast` values whose operand is memref.
- Keep intrinsic result materialization unchanged: address intrinsics may still
  produce a memref or `!pto.ptr`, and dimension/stride intrinsics still produce
  `index`.
- Update pass documentation, tile instruction docs, and regression tests to
  describe the descriptor-native contract.

## Capabilities

### New Capabilities

- `vpto-native-intrinsic-folding`: Defines `FoldTileBufIntrinsics` as a
  descriptor-native VPTO folding stage for tile-buffer and tensor-view helper
  intrinsics.

### Modified Capabilities

- `vpto-expand-tile-boundary`: Narrows the post-`ExpandTileOp` folding contract
  by removing the remaining memref-chain compatibility path inside
  `FoldTileBufIntrinsics`.

## Impact

- Affected IR definitions: `pto.tensor_view_addr` source operand type and
  verifier.
- Affected transform: `lib/PTO/Transforms/FoldTileBufIntrinsics.cpp`.
- Affected pass docs: `include/PTO/Transforms/Passes.td`.
- Affected public docs/specs: tile pointer/view instruction documentation and
  release spec copy.
- Affected tests: focused lit tests for native tensor-view folding, native
  tile-buffer folding, and rejection of memref bridge-shaped inputs.
- Compatibility impact: IR that depends on `View2Memref`-style memref chains or
  memref-to-descriptor `unrealized_conversion_cast` bridges is no longer a
  supported input to these intrinsics.
