## 1. IR Contract

- [x] 1.1 Update `pto.tensor_view_addr` ODS to accept only
      `TensorViewType` and `PartitionTensorViewType` sources.
- [x] 1.2 Update `TensorViewAddrOp::verify()` diagnostics and remove memref
      source handling.
- [x] 1.3 Confirm `pto.tile_buf_addr`, `pto.tile_valid_rows`, and
      `pto.tile_valid_cols` already require `TileBufType` sources.

## 2. Native Tensor-View Folding

- [x] 2.1 Remove `ViewChain` and memref view-chain tracing from
      `FoldTileBufIntrinsics.cpp`.
- [x] 2.2 Add a native tensor-view metadata resolver for
      `pto.make_tensor_view` and `pto.partition_view`.
- [x] 2.3 Fold `pto.tensor_view_addr` from native base pointer plus accumulated
      partition element offset.
- [x] 2.4 Fold `pto.get_tensor_view_dim` from static type shape or native
      shape/size operands.
- [x] 2.5 Fold `pto.get_tensor_view_stride` from native
      `make_tensor_view` stride operands.
- [x] 2.6 Reject tensor-view descriptor values produced by memref-sourced
      `builtin.unrealized_conversion_cast`.

## 3. Native Tile-Buffer Folding

- [x] 3.1 Keep native tile metadata resolution for `pto.pointer_cast`,
      explicit-address `pto.alloc_tile`, `pto.bind_tile`, `pto.subview`,
      `pto.bitcast`, and `pto.treshape`.
- [x] 3.2 Add a direct diagnostic for tile values produced by memref-sourced
      `builtin.unrealized_conversion_cast`.
- [x] 3.3 Preserve dynamic valid-row/valid-col folding from native metadata.
- [x] 3.4 Preserve native `pto.subview` byte-offset calculation.

## 4. Documentation

- [x] 4.1 Update `include/PTO/Transforms/Passes.td` for native descriptor-only
      `FoldTileBufIntrinsics` inputs.
- [x] 4.2 Update `include/PTO/IR/VPTOOps.td` descriptions for
      `pto.tensor_view_addr` and `pto.tile_buf_addr`.
- [x] 4.3 Update `docs/isa/tile-op/03-pointer-and-view.md` to remove memref
      input wording.
- [x] 4.4 Refresh `docs/release/PTO-tile-Instruction-SPEC-v0.4.md` using the
      repository release-doc workflow or make an equivalent synchronized edit.

## 5. Regression Tests

- [x] 5.1 Add a positive native `make_tensor_view` folding test for
      `tensor_view_addr`, `get_tensor_view_dim`, and
      `get_tensor_view_stride`.
- [x] 5.2 Add a positive native `partition_view` folding test covering address
      offset, dynamic size, and inherited stride.
- [x] 5.3 Add a negative test rejecting `tensor_view_addr` with a memref source.
- [x] 5.4 Add a negative test rejecting memref-sourced
      `unrealized_conversion_cast` bridges for tensor-view intrinsics.
- [x] 5.5 Add a negative test rejecting memref-sourced
      `unrealized_conversion_cast` bridges for `tile_buf_addr`.
- [x] 5.6 Run focused lit tests for `FoldTileBufIntrinsics` and record the
      exact commands in validation notes.

## 6. Validation

- [x] 6.1 Build `ptoas`.
- [x] 6.2 Run focused native fold tests.
- [x] 6.3 Run existing native tile-buffer folding tests.
- [x] 6.4 Run `openspec validate native-fold-tile-view-intrinsics --strict`.
- [x] 6.5 Run `git diff --check`.
