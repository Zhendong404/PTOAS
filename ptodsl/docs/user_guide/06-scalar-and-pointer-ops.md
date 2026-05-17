# 6. Scalar and Pointer Operations

Chapter 5 established the rule: Python constructs are resolved at trace time, PTO constructs produce device-side behavior. This chapter applies that distinction to scalars and pointers — when to use a plain Python number, when to use a `pto.*` scalar operation, and how to work with typed pointers.

## 6.1 Python scalars vs PTO scalars

A **Python scalar** is any value computed by Python during tracing: a literal (`3.14159`), a shape dimension (`A.shape[0]`), a constexpr parameter (`BLOCK`), or an arithmetic expression built from these (`1.0 / sqrt(dim)`). These are evaluated at trace time and their results are baked into the device code as constants.

A **PTO scalar** is a value that lives on the device at runtime. It comes from a `pto.load` read, a device-side computation (`pto.max`, `pto.exp`), or a runtime query (`pto.get_block_idx()`). PTO scalars flow through the recorded program and are not resolved until the kernel executes.

### The mixed expression

In practice, a single expression can mix both kinds:

```python
alpha * o_prev + beta * pv_val
# ^ Python float (trace-time constant, e.g. 1.0 / sqrt(dim))
#        ^ PTO scalar (loaded from tile at runtime)
#                  ^ PTO scalar (loaded from tile at runtime)
```

`alpha` is a Python float computed from compile-time information — it becomes an immediate constant in the device code. `o_prev` and `pv_val` are PTO scalars read from tiles at runtime. The `*` and `+` operators are recorded as device-side multiply-add instructions. The tracer sees the whole expression and produces the appropriate device instructions, embedding the constant operand where possible.

### Rule of thumb

| If the value... | Use... | Example |
|-----------------|--------|---------|
| Is known at compile time | Python scalar | `BLOCK`, `1.0 / sqrt(dim)`, `A.shape[0]` |
| Comes from device memory | PTO scalar | `pto.load(tile[r, c])` |
| Depends on a runtime value | PTO scalar | `pto.max(m_prev, row_max)` |
| Is a block/subblock index | PTO scalar | `pto.get_block_idx()` |

When in doubt, ask: *can this value change between launches of the same compiled kernel?* If yes, it must be a PTO scalar.

## 6.2 Scalar access: load and store

`pto.load` reads a single scalar element from a typed pointer or tile location. `pto.store` writes a scalar back. These are the canonical scalar memory ops for SIMT authoring.

### load — load scalar

```python
val = pto.load(tile[row, col])
```

`tile[row, col]` is index syntax — it selects one element at the given row and column. The result is a PTO scalar whose type matches the tile's element type. Row and column indices are PTO scalars (or Python integers that the tracer promotes).

`pto.load` also accepts a typed pointer with an element offset:

```python
val = pto.load(ptr, offset)
```

Or with pointer arithmetic:

```python
val = pto.load(ptr + offset)
```

The offset is counted in elements, not bytes.

### store — store scalar

```python
pto.store(value, tile[row, col])
```

Writes `value` into the tile at `[row, col]`. This is the scalar counterpart of `vsts` — it moves one element, not a vector register.

With a pointer and offset:

```python
pto.store(value, ptr, offset)
```

### Typical SIMT usage

`pto.load` and `pto.store` are the primary data access pattern inside `@pto.simt` kernels. Each `load`/`store` operates on one element per work-item, but the SIMT unit executes the same instruction across many work-items in parallel:

```python
@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile, pv_tile: pto.Tile,
    alpha_tile: pto.Tile, beta_tile: pto.Tile,
    o_next_tile: pto.Tile,
    row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
):
    with pto.for_(row_start, row_stop, step=1) as row:
        alpha = pto.load(alpha_tile[row, 0])
        beta = pto.load(beta_tile[row, 0])
        with pto.for_(0, valid_dim, step=1) as col:
            o_prev = pto.load(o_prev_tile[row, col])
            pv_val = pto.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            pto.store(o_next, o_next_tile[row, col])
```

When writing to a raw pointer (e.g., a small metadata buffer obtained via `as_ptr()`), use the pointer-plus-offset form:

```python
meta_ptr = meta_tile.as_ptr()
pto.store(0, meta_ptr, 0)                    # store at element offset 0
pto.store(valid_rows, meta_ptr, 4)           # store at element offset 4
row_start = pto.load(meta_ptr, 0)
row_stop  = pto.load(meta_ptr, 4)
```

## 6.3 Scalar arithmetic and math

PTO scalar values support standard arithmetic operators. These are recorded as device-side instructions:

```python
# Arithmetic operators
sum_val = a + b
diff = a - b
prod = a * b
quot = a / b

# Comparisons (produce pto.i1)
big = pto.gt(val, threshold)
small = pto.lt(val, threshold)
equal = pto.eq(a, b)
```

Built-in math functions for PTO scalars:

| Function | Description |
|----------|-------------|
| `pto.max(a, b)` | Maximum of two scalars |
| `pto.min(a, b)` | Minimum of two scalars |
| `pto.exp(x)` | Exponential, e^x |
| `pto.log(x)` | Natural logarithm |
| `pto.sqrt(x)` | Square root |
| `pto.abs(x)` | Absolute value |

```python
m_next = pto.max(m_prev, row_max)
l_scaled = l_prev * pto.exp(m_prev - m_next)
```

These are the scalar-path counterparts of the vector math operations covered in Chapter 8. Use them inside `@pto.simt` kernels and in `@pto.ukernel` orchestration code where you need to compute a loop bound or a scalar coefficient from runtime data.

## 6.4 Pointer operations

Typed pointers (Section 4.4) carry both an element type and a memory space. This section covers the operations that create and manipulate them.

### Obtaining pointers: as_ptr()

Tiles and tensor views expose their base address via `as_ptr()`:

```python
gm_ptr = partition.as_ptr()    # GM pointer from a PartitionTensorView
ub_ptr = tile.as_ptr()         # UB pointer from a Tile
```

`as_ptr()` is the preferred way to get a typed pointer from a high-level descriptor. The result carries the correct element type and memory space from the source.

### addptr — pointer arithmetic

`pto.addptr` advances a pointer by a number of elements (not bytes):

```python
next = pto.addptr(ptr, offset)
# offset is in elements, not bytes
```

For example, to advance past 1024 `f32` elements:

```python
ptr = pto.addptr(base_ptr, 1024)  # advances by 1024 * sizeof(f32) bytes
```

Both `addptr` and the `+` shorthand on pointers count in elements, not bytes.

### castptr — reinterpret pointer type

`pto.castptr` reinterprets an address as a different pointer type:

```python
ptr = pto.castptr(address, pto.ptr(pto.f32, pto.MemorySpace.UB))
```

This is an advanced operation. Prefer `as_ptr()` when the source already carries type information.

## 6.5 Compile-time queries

These functions return values that are known at trace time from type information or hardware constants:

### bytewidth

```python
pto.bytewidth(dtype)  # → Python int (trace-time)
```

Returns the size in bytes of a single element of `dtype`:

```python
bw = pto.bytewidth(pto.f32)   # 4
bw = pto.bytewidth(pto.f16)   # 2
bw = pto.bytewidth(pto.i8)    # 1
```

Since the result is a Python integer, it can be used in Python arithmetic that runs at trace time — for example, computing byte offsets inside a constexpr loop.

### elements_per_vreg

```python
pto.elements_per_vreg(dtype)  # → Python int (trace-time)
```

Returns how many elements of `dtype` fit in one 256-byte vector register:

```python
vec = pto.elements_per_vreg(pto.f32)   # 64
vec = pto.elements_per_vreg(pto.f16)   # 128
vec = pto.elements_per_vreg(pto.i8)    # 256
```

This is the standard stride for chunking column loops in SIMD kernels:

```python
VEC = pto.elements_per_vreg(pto.f32)
with pto.for_(0, cols, step=VEC) as c:
    ...
```

## 6.6 Per-element tile traversal in @pto.simt

`@pto.simt` kernels are the natural home for per-element scalar work. A typical pattern uses nested `pto.for_` loops to walk over a tile row by row, column by column:

```python
@pto.simt
def elementwise_scale(
    src_tile: pto.Tile,
    dst_tile: pto.Tile,
    scale: pto.f32,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.for_(0, rows, step=1) as r:
        with pto.for_(0, cols, step=1) as c:
            val = pto.load(src_tile[r, c])
            scaled = val * scale
            pto.store(scaled, dst_tile[r, c])
```

This reads each element from `src_tile`, multiplies by `scale`, and writes to `dst_tile`. The SIMT unit executes the body in parallel across work-items, so this scalar-looking code achieves high throughput — each work-item handles a different `(r, c)` pair.

For operations that need per-row metadata alongside per-element computation, lift the row-level scalar out of the inner loop:

```python
@pto.simt
def blend_with_per_row_coeffs(
    o_prev_tile: pto.Tile,
    pv_tile: pto.Tile,
    alpha_tile: pto.Tile,    # [rows, 1] — one coefficient per row
    beta_tile: pto.Tile,     # [rows, 1]
    o_next_tile: pto.Tile,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.for_(0, rows, step=1) as r:
        alpha = pto.load(alpha_tile[r, 0])   # read once per row
        beta = pto.load(beta_tile[r, 0])     # read once per row
        with pto.for_(0, cols, step=1) as c:
            o_prev = pto.load(o_prev_tile[r, c])
            pv_val = pto.load(pv_tile[r, c])
            o_next = alpha * o_prev + beta * pv_val
            pto.store(o_next, o_next_tile[r, c])
```

This hoists `alpha` and `beta` out of the inner loop — the row coefficients are loaded once and broadcast across all columns in that row.
