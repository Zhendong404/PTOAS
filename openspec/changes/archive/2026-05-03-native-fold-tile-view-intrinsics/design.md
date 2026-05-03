## Context

The active VPTO pipeline no longer includes a `View2Memref` conversion stage.
Before `FoldTileBufIntrinsics`, TileLang helper intrinsics should see native
PTO descriptors:

```text
local tile data      -> !pto.tile_buf
global tensor view   -> !pto.tensor_view
partitioned view     -> !pto.partition_tensor_view
```

The current `FoldTileBufIntrinsics` implementation has two contracts mixed
together:

- native tile-buffer metadata tracing for `!pto.tile_buf`,
- legacy tensor-view tracing through memref view ops produced by the old
  view-to-memref world.

With the old conversion pass gone, the memref view chain is not a valid source
of truth for these intrinsics. The source of truth is now the native PTO
producer chain:

```mlir
%tv = pto.make_tensor_view %ptr, shape = [...], strides = [...]
  : !pto.tensor_view<...>

%part = pto.partition_view %tv, offsets = [...], sizes = [...]
  : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```

## Goals / Non-Goals

**Goals:**

- Make every input consumed by `FoldTileBufIntrinsics` native descriptor form:
  `!pto.tile_buf`, `!pto.tensor_view`, or `!pto.partition_tensor_view`.
- Remove tensor-view folding support for
  `unrealized_conversion_cast -> memref.subview -> memref.reinterpret_cast`.
- Reject memref-to-descriptor bridge values with clear diagnostics.
- Fold tensor-view address, dimension, and stride intrinsics from
  `pto.make_tensor_view` and `pto.partition_view`.
- Preserve existing native tile-buffer metadata folding.
- Preserve address intrinsic result materialization to memref or `!pto.ptr`.

**Non-Goals:**

- Reintroducing `View2Memref` or any compatibility pass for old view lowering.
- Banning memref result types from `pto.tile_buf_addr` or
  `pto.tensor_view_addr`.
- Rewriting later VPTO pointer normalization or final emission passes unless
  they directly depend on the removed memref-chain folding behavior.
- Changing TileLang DSL source syntax.

## Decisions

### Decision: Intrinsic inputs are descriptor-native only

`FoldTileBufIntrinsics` SHALL accept only the descriptor value types that match
the intrinsic family:

- tile-buffer family: `!pto.tile_buf`,
- tensor-view family: `!pto.tensor_view` or `!pto.partition_tensor_view`.

Any `memref` input, or any descriptor value produced by an
`unrealized_conversion_cast` from memref, is a pipeline contract violation.

### Decision: Tensor-view folding traces native PTO producers

Tensor-view intrinsic folding should trace native PTO producer chains:

- `pto.make_tensor_view` supplies the base pointer, logical shape operands, and
  logical stride operands.
- `pto.partition_view` supplies logical offsets and sizes and traces back to
  the source `pto.make_tensor_view`.

`pto.get_tensor_view_dim` resolves as follows:

- static result shape dimensions fold to `arith.constant`,
- dynamic `!pto.tensor_view` dimensions fold to the corresponding
  `make_tensor_view` shape operand,
- dynamic `!pto.partition_tensor_view` dimensions fold to the corresponding
  `partition_view` size operand.

`pto.get_tensor_view_stride` resolves from the source `make_tensor_view`
stride operand for the requested dimension. `partition_view` does not currently
carry a stride operand, so the partition preserves the source logical stride.

`pto.tensor_view_addr` resolves from the source pointer in `make_tensor_view`.
For a partition view, the pass computes an element offset from
`partition_view` offsets and the source strides, then materializes either the
base pointer or `pto.addptr(base, offset)`.

### Decision: Tile-buffer folding remains native producer based

The tile-buffer resolver remains centered on native tilebuf producers:

- `pto.pointer_cast`,
- explicit-address `pto.alloc_tile`,
- tilebuf-to-tilebuf `pto.bind_tile`,
- `pto.subview`,
- `pto.bitcast`,
- `pto.treshape`.

If the tile value is produced by `unrealized_conversion_cast` from memref, the
pass should emit a diagnostic that identifies the old bridge shape and asks for
native tilebuf producers instead.

### Decision: Result materialization is still VPTO-side lowering

Native descriptor-only input does not mean descriptor-only output. Address
intrinsics are the point where VPTO authoring IR materializes concrete pointer
or memref values:

- `pto.tile_buf_addr` may produce a memref or `!pto.ptr`,
- `pto.tensor_view_addr` may produce a memref or `!pto.ptr`,
- dim and stride intrinsics produce `index`.

The ban is on memref-shaped input provenance, not on the requested lowered
result form.

## Implementation Sketch

1. Update `TensorViewAddrOp`:
   - remove `AnyMemRef` from the source ODS constraint,
   - update verifier diagnostics to name only tensor-view source types,
   - delete docs that describe memref source compatibility.

2. Refactor `FoldTileBufIntrinsics`:
   - delete `ViewChain` and all `memref::SubViewOp` /
     `memref::ReinterpretCastOp` tracing helpers,
   - introduce a native view metadata resolver that returns base pointer,
     shape operands, stride operands, and accumulated partition offsets,
   - rewrite tensor-view addr/dim/stride folding on top of that resolver,
   - add explicit rejection for `UnrealizedConversionCastOp` values sourced
     from memref for both tile and tensor-view families,
   - delete dead memref-chain cleanup logic at the end of the pass.

3. Preserve native tilebuf folding:
   - keep existing address and dynamic valid-shape metadata tracing,
   - improve diagnostics for unsupported producers and bridge-shaped IR,
   - keep subview byte-offset calculation for native `pto.subview`.

4. Update documentation and tests:
   - pass docs state native descriptor inputs only,
   - tile docs remove `tile-bound memref` and memref-source wording,
   - lit tests cover native view folding and memref bridge rejection.

## Risks / Trade-offs

- Some hand-written tests may still build tensor-view intrinsics with memref
  sources. Those tests should be migrated to `pto.make_tensor_view` /
  `pto.partition_view`.
- Native tensor-view address folding relies on `make_tensor_view` carrying a
  `!pto.ptr` base pointer. If other producer forms exist, they should either
  be added explicitly to the native resolver or rejected with actionable
  diagnostics.
- Partition stride semantics are assumed to preserve source strides because
  `pto.partition_view` does not expose stride operands today. If strided
  partition views are introduced later, the resolver must be extended.
