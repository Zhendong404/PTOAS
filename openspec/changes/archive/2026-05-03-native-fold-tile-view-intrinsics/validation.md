## Validation Notes

Date: 2026-05-02

### Passed

- Build:
  - `cmake --build build --target ptoas -j2`
  - Result: passed, build tree was already up to date (`ninja: no work to do.`).
- Focused `FoldTileBufIntrinsics` lit suite:
  - `PTOAS_BUILD_DIR=$PWD/build LLVM_BUILD_DIR=$PWD/../llvm-project/build-shared /home/zhangzhendong/.local/bin/llvm-lit -sv test/lit/pto/fold_tile_buf_intrinsics.pto test/lit/pto/fold_tile_buf_intrinsics_level3_manual_addr.pto test/lit/pto/fold_tile_buf_intrinsics_native_metadata.pto test/lit/pto/fold_tile_buf_intrinsics_native_tensor_views.pto test/lit/pto/reject_tensor_view_addr_memref_source.pto test/lit/pto/reject_old_memref_to_tensorview_bridge.pto test/lit/pto/reject_old_memref_to_tilebuf_bridge.pto`
  - Result: 7 passed.
- Existing native tile-buffer folding lit suite:
  - `PTOAS_BUILD_DIR=$PWD/build LLVM_BUILD_DIR=$PWD/../llvm-project/build-shared /home/zhangzhendong/.local/bin/llvm-lit -sv test/lit/pto/fold_tile_buf_intrinsics.pto test/lit/pto/fold_tile_buf_intrinsics_level3_manual_addr.pto test/lit/pto/fold_tile_buf_intrinsics_native_metadata.pto test/lit/pto/reject_old_memref_to_tilebuf_bridge.pto`
  - Result: 4 passed.
- OpenSpec validation:
  - `openspec validate native-fold-tile-view-intrinsics --strict`
  - Result: passed (`Change 'native-fold-tile-view-intrinsics' is valid`).
- Diff hygiene:
  - `git diff --check`
  - Result: passed.

### Coverage Notes

- `fold_tile_buf_intrinsics_native_tensor_views.pto` covers native
  `pto.make_tensor_view` folding for `pto.tensor_view_addr`,
  `pto.get_tensor_view_dim`, and `pto.get_tensor_view_stride`, plus native
  `pto.partition_view` folding for address offset, dynamic size, and inherited
  stride.
- `reject_tensor_view_addr_memref_source.pto` covers verifier rejection of a
  direct memref operand to `pto.tensor_view_addr`.
- `reject_old_memref_to_tensorview_bridge.pto` covers pass-time rejection of a
  memref-sourced `builtin.unrealized_conversion_cast` bridge for tensor-view
  intrinsics.
- `reject_old_memref_to_tilebuf_bridge.pto` keeps the tile-buffer bridge
  rejection case covered alongside the new tensor-view regressions.
