## 1. Pipeline Boundary

- [x] 1.1 Remove `createMemrefToTileBufPass()` from the default VPTO backend
      lowering pipeline in `tools/ptoas/ptoas.cpp`.
- [x] 1.2 Update the `lowerPTOToVPTOBackend` comment to state that
      `ExpandTileOp` is the PTO IR -> VPTO IR boundary.
- [x] 1.3 Ensure the pass sequence is:
      `ExpandTileOp -> PTOInlineLibCall -> FoldTileBufIntrinsics -> SCCP ->
      Canonicalizer`.
- [x] 1.4 Confirm no default `--pto-backend=vpto` path schedules the old
      memref-to-tilebuf bridge before `ExpandTileOp`.

## 2. ExpandTileOp Contract

- [x] 2.1 Add diagnostics in `ExpandTileOp` for tile operations whose tile
      operands are memref before the boundary.
- [x] 2.2 Keep view/tensor-view operand bridging behavior only for the VPTO-side
      helper call contract, not as a pre-boundary local tile bridge.
- [x] 2.3 Update `ExpandTileOp` pass documentation to describe the boundary and
      its expected input/output IR forms.

## 3. FoldTileBufIntrinsics Native Metadata Resolver

- [x] 3.1 Delete `findBindTileForTileBuf()` from
      `lib/PTO/Transforms/FoldTileBufIntrinsics.cpp`.
- [x] 3.2 Do not add a legacy fallback for
      `bind_tile -> unrealized_conversion_cast -> tile_buf`.
- [x] 3.3 Add a native tilebuf metadata resolver for address, static valid
      shape, and dynamic valid-row/valid-col operands.
- [x] 3.4 Resolve `pto.tile_buf_addr` from native tilebuf producers and
      materialize the requested memref or `!pto.ptr` result in VPTO IR.
- [x] 3.5 Resolve `pto.tile_valid_rows` and `pto.tile_valid_cols` from
      `TileBufType` static metadata or native dynamic valid-shape operands.
- [x] 3.6 Emit actionable diagnostics when a dynamic valid dimension cannot be
      traced to native metadata.
- [x] 3.7 Preserve existing tensor-view intrinsic folding unless a conflict with
      the new boundary contract is discovered.

## 4. Documentation And Specs

- [x] 4.1 Remove `MemrefToTileBuf` pass documentation, registration, factory
      declaration, build wiring, and implementation.
- [x] 4.2 Update `FoldTileBufIntrinsics` documentation to describe native
      tilebuf metadata resolution and VPTO-side memref/ptr materialization.
- [x] 4.3 Update any tests or comments that describe the VPTO tile-op expansion
      path as `MemrefToTileBuf -> ExpandTileOp`.

## 5. Regression Tests

- [x] 5.1 Add a VPTO pipeline test showing the default backend does not require
      `MemrefToTileBuf` before `ExpandTileOp`.
- [x] 5.2 Add a native `pto.pointer_cast` tilebuf test that folds
      `tile_buf_addr`, `tile_valid_rows`, and `tile_valid_cols`.
- [x] 5.3 Add a level3/manual-address `pto.alloc_tile` test or document why it
      is canonicalized before folding.
- [x] 5.4 Add at least one native tile alias test covering `bind_tile`,
      `subview`, `bitcast`, or `treshape`.
- [x] 5.5 Add a negative test that rejects old synthetic bridge-shaped IR on the
      default VPTO path.
- [x] 5.6 Run the focused lit tests and record the exact commands in the PR or
      change validation notes.

      Validation commands:
      - `llvm-lit -v test/lit/pto/fold_tile_buf_intrinsics.pto test/lit/pto/fold_tile_buf_intrinsics_level3_manual_addr.pto test/lit/pto/fold_tile_buf_intrinsics_native_metadata.pto test/lit/pto/reject_old_memref_to_tilebuf_bridge.pto`
      - `ninja -C build ptoas`
      - `openspec validate define-vpto-expand-tile-boundary --strict`
      - `git diff --check`
