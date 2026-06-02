# CCE -> PTODSL Patterns

## Construct Mapping

- `static_assert(cond, "...")`
  - Port to a nearby Python `assert cond, "..."`.

- `constexpr uint32_t Cube_S0 = CUBE_S0;`
  - Keep the same local name in Python.
  - Do not rename just because the PTODSL code could inline it.

- `TLOAD(tile, global)`
  - Prefer `pto.tile.load(...)`.

- `TSTORE(global, tile)`
  - Prefer `pto.tile.store(...)`.

- `TINSERT(dst, src, row_offset, col_offset)`
  - Use `pto.tile.insert(src, dst, row_offset, col_offset)`.

- `TMOV(dst, src)`
  - Use `pto.tile.mov(src, dst)`.

- `TMOV<..., AccToVecMode::DualModeSplitN>(dst, src)`
  - Use `pto.tile.mov(src, dst, mode="split_n")`.

- `TMOV<..., AccToVecMode::DualModeSplitM>(dst, src)`
  - Use `pto.tile.mov(src, dst, mode="split_m")`.

- `wait_intra_block(pipe, flag)`
  - Use `pto.wait_intra_flag(pipe, flag)`.

- `set_intra_block(pipe, flag)`
  - Use `pto.set_intra_flag(pipe, flag)`.

- `TASSIGN(tile, addr)`
  - Prefer `pto.alloc_tile(..., addr=...)` when authoring a subtile view or aliased tile buffer.

## Pointer And Offset Rules

- `pto.addptr(ptr, offset)` uses element offsets, not byte offsets.
- If C++ computes one byte offset and PTODSL needs `addptr`, convert it back to element count first.
- For row-slice reduce tiles, `addr=pto.addptr(tile.as_ptr(), reduce_slice_rows)` is usually the correct PTODSL form, not a byte offset.

## Hook Translation

Small C++ hook structs usually map well to demo-local dataclasses:

```python
@dataclass(frozen=True)
class PReadyHook:
    sync: object
    enable: object

    def __call__(self):
        with pto.if_(self.enable) as enable_br:
            with enable_br.then_:
                self.sync.wait()
```

Use this pattern for:

- `PReadyHook`
- `QReadyHook`
- `Sm2PvFreeHook`
- `PreATExtOpReadyHook`
- `PreBTExtOpReadyHook`

## Early-Return Pattern

When C++ does:

```cpp
if (skip) {
    ...
    return;
}
main_path();
```

do not write:

```python
with pto.if_(skip) as br:
    with br.then_:
        ...
    with br.else_:
        pass
main_path()
```

Instead write:

```python
def emit_main_path():
    ...

with pto.if_(skip) as br:
    with br.then_:
        ...
    with br.else_:
        emit_main_path()
```

## FlashAttention-Specific Porting Order

For FA-like CCE kernels, the most stable order is:

1. constants, enums, and sync flags
2. hook structs
3. macro helpers such as `pto_macro_matmul`
4. `compute_qk`
5. `compute_p`
6. `compute_pv`
7. `compute_gu`
8. top-level prologue / preload / steady-state / epilogue loops

This keeps the code reviewable and avoids rewriting large stage bodies before the common helpers settle.

## Macro Helper Guidance

When porting `pto_macro_matmul`:

- keep `Cube_M`, `Tile_K`, `Cube_N`, and `L1LoadBFirst` explicit in the interface
- keep hook order visible at the callsite
- keep ping-pong bookkeeping even if the first PTODSL version is only structurally faithful
- if one stage uses an operand order different from the helper's generic `aMatTile` / `bMatTile` view, prefer one thin adapter instead of flattening the helper call away

## Validation Strategy

When the user cares about structure more than compilation:

- run `python3 -m py_compile ...`
- check for broken signatures and unresolved names
- explicitly call out any placeholders that preserve structure but not full semantics yet

When the user later asks for compile or runtime fidelity, continue from the structurally aligned version instead of rewriting from scratch.
