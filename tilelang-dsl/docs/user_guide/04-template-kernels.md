### Template-based Kernel Authoring

For operations that share similar computation patterns but differ in their core vector operations, the DSL supports template-based kernel authoring. This allows a single kernel implementation to serve multiple related operations through parameterized templates.

#### Multi-operation Kernels with `ops` Parameter

Instead of specifying a single `op` parameter, you can provide an `ops` list to match multiple operations:

```python
@pto.vkernel(
    target="a5",
    ops=["tadd", "tsub", "tmul", "tdiv"],  # List of operations
    dtypes=[(T, T, T)],                    # Type signature using type variable
    advanced=True,
    templates={
        "core": {
            "tadd": "vadd",
            "tsub": "vsub", 
            "tmul": "vmul",
            "tdiv": "vdiv",
        }
    }
)
def elementwise_arithmetic(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    dtype = dst.element_type
    rows, cols = dst.valid_shape
    elems_per_vreg = pto.elements_per_vreg(dtype)  # Number of elements per vector register
    for row in range(0, rows, 1):
        remained = cols
        for col in range(0, cols, elems_per_vreg):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            out = pto.tpl("core", lhs, rhs, mask)  # Template dispatch
            pto.vsts(out, dst[row, col:], mask)
```

`op` and `ops` are mutually exclusive, and exactly one of them must be
provided. `ops=[...]` only widens the matcher set; callers still use
`pto.select_kernel(target, concrete_op, operand_types, ...)` with a concrete
PTO op such as `"tadd"` or `"tmul"`.

#### Template System

The template system consists of three components:

1. **`templates` parameter**: A dictionary mapping template names to operation-specific implementations
2. **`pto.tpl()` function**: A compile-time placeholder that resolves to the appropriate implementation for the currently selected concrete op
3. **`ops` parameter**: Replaces the singular `op` parameter for multi-operation kernels

##### Template Definition

Templates are defined in the `templates` parameter of `@pto.vkernel`. Each template is a dictionary mapping operation names to implementation strings:

```python
templates={
    "template_name": {
        "op1": "implementation_for_op1",
        "op2": "implementation_for_op2",
        # ...
    },
    "another_template": {
        "op1": "different_implementation_for_op1",
        # ...
    }
}
```

Template-slot metadata is static and validated when the descriptor is
registered:

- slot names must be non-empty strings
- mapping keys must be concrete ops covered by the descriptor matcher set
- mapping values must be supported real `pto.*` op names

The implementation strings are typically vector operation names such as
`"vadd"`, `"vsub"`, `"vmul"`, and `"vdiv"`, which are resolved during kernel
expansion.

##### Template Usage with `pto.tpl()`

The `pto.tpl()` operation enables template dispatch for multi-operation kernels, allowing code reuse across related operations through compile-time substitution.

#### `pto.tpl(template_name: str, *args) -> Any`

**Description**: Template dispatch operation for multi-operation kernels. Resolves to different implementations based on the current operation being expanded.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `template_name` | `str` | Name of the template to dispatch |
| `*args` | `Any` | Positional arguments passed unchanged to the resolved real implementation |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `Any` | Result of the template implementation |

**Behavior**:
- Only valid inside kernels decorated with `@pto.vkernel` that have a `templates` parameter
- The first argument must be a string literal template-slot name
- During kernel expansion for a specific operation `op_name`, `pto.tpl("template_name", ...)` is replaced with the implementation specified in `templates["template_name"]["op_name"]`
- The replacement is a direct compile-time substitution; positional arguments are passed unchanged
- Template implementations are typically string names of vector operations (e.g., `"vadd"`, `"vsub"`)
- `pto.select_kernel(...)` must bind a concrete op before template expansion can happen
- Python dict lookup, callable values, lambdas, and other runtime dispatch patterns are not part of the supported kernel-body surface

**Example**:
```python
@pto.vkernel(
    ops=["tadd", "tsub"],
    dtypes=[(T, T, T)],
    templates={
        "core": {
            "tadd": "vadd",
            "tsub": "vsub",
        }
    }
)
def elementwise_kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    # ... load vectors
    result = pto.tpl("core", lhs, rhs, mask)  # Expands to vadd for tadd, vsub for tsub
    # ... store result
```

**Constraints**:
- Template names must be defined in the `templates` parameter of the `@pto.vkernel` decorator
- When a kernel body uses `pto.tpl("slot", ...)`, that slot must define an implementation for the currently selected concrete op
- Template implementations must be valid operation names in the DSL

#### Decorator Parameters Update

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `target` | `str` | Yes | Target hardware architecture (e.g., `"a5"` for Ascend 950). |
| `op` | `str` | No* | Name of the PTO operation to match. **Mutually exclusive with `ops`**. |
| `ops` | `List[str]` | No* | List of PTO operation names to match. **Mutually exclusive with `op`**. |
| `dtypes` | `List[Tuple[Type, ...]]` | Yes | List of type signatures. Each tuple specifies the expected data types for the operation's operands. |
| `templates` | `Dict[str, Dict[str, str]]` | No | Static slot mappings from concrete matcher ops to real `pto.*` op names. Required when the kernel body uses `pto.tpl(...)`. |
| `constraints` | `List[Constraint]` | No | Additional constraints that must be satisfied for kernel selection. |
| `priority` | `int` | No | Selection priority when multiple kernels match. Default: `0`. |
| `name` | `str` | No | Kernel name (used for debugging and profiling). Defaults to the decorated function's name. |
| `advanced` | `bool` | No | Enable advanced-tier DSL surfaces (for example `strict_vecscope`, raw pointer family, and low-level DMA family). Implicit vecscope inference is mode-independent and runs only when no explicit `with pto.vecscope():` is present. Default: `False`. |

**Note**:
- Either `op` or `ops` must be provided, but not both.
- `templates` is only needed when the kernel body uses `pto.tpl(...)`.
- `pto.select_kernel(...)` still queries with a concrete op even for `ops=[...]` descriptors.

#### Advanced Template Patterns

##### Multiple Templates per Kernel

A kernel can define multiple templates for different aspects of the computation:

```python
@pto.vkernel(
    target="a5",
    ops=["tadd_relu", "tsub_relu", "tadd_abs", "tsub_abs"],
    dtypes=[(T, T, T)],
    templates={
        "arithmetic": {
            "tadd_relu": "vadd",
            "tsub_relu": "vsub",
            "tadd_abs": "vadd",
            "tsub_abs": "vsub",
        },
        "postprocess": {
            "tadd_relu": "vrelu",
            "tsub_relu": "vrelu",  # Same activation for both
            "tadd_abs": "vabs",
            "tsub_abs": "vabs",
        }
    }
)
def elementwise_with_postprocess(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    # ... load vectors
    arith_result = pto.tpl("arithmetic", lhs, rhs, mask)
    postprocessed = pto.tpl("postprocess", arith_result, mask)
    # ... store result
```

##### Compile-time Substitution Model

Template-slot expansion happens before semantic checking and lowering:

- `pto.select_kernel(...)` first binds a concrete op such as `"tadd"`
- the frontend then resolves `pto.tpl("core", ...)` using `templates["core"]["tadd"]`
- the placeholder is rewritten to a real `pto.*` call before semantic analysis
- diagnostics for unknown slots, missing mappings, or unsupported resolved surfaces are raised before any VPTO IR is generated

#### Type Variables in Template Kernels

Template kernels often use type variables to enforce type consistency:

```python
T = pto.TypeVar('T')

@pto.vkernel(
    target="a5",
    ops=["tadd", "tsub"],
    dtypes=[(T, T, T)],  # All three operands share type T
    templates={
        "core": {
            "tadd": "vadd",
            "tsub": "vsub",
        }
    }
)
def typed_elementwise(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    # Type variable T ensures all tiles have same element type
    dtype = dst.element_type  # This is type T
    # ... implementation
```

#### Selection Mechanism for Template Kernels

When a PTO operation matches a template kernel:
1. The system selects the descriptor based on `op` exact match or `ops` list inclusion.
2. `pto.select_kernel(...)` binds the concrete query op as the descriptor's `selected_op`.
3. During frontend expansion, `pto.tpl()` calls are resolved using that bound concrete op.
4. For operation `"op_name"`, template `"template_name"` resolves to `templates["template_name"]["op_name"]`.
5. The resolved string (e.g., `"vadd"`) is replaced with the corresponding real DSL operation before semantic analysis and lowering.

#### Example: Unified Arithmetic Kernel

```python
T = pto.TypeVar('T')

@pto.vkernel(
    ops=["tadd", "tsub", "tmul", "tdiv", "tmax", "tmin"],
    dtypes=[(T, T, T)],
    advanced=True,
    templates={
        "arithmetic": {
            "tadd": "vadd",
            "tsub": "vsub", 
            "tmul": "vmul",
            "tdiv": "vdiv",
            "tmax": "vmax",
            "tmin": "vmin",
        }
    }
)
def unified_arithmetic(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    """Single implementation for six arithmetic operations."""
    dtype = dst.element_type
    rows, cols = dst.valid_shape
    elems_per_vreg = pto.elements_per_vreg(dtype)  # Number of elements per vector register
    
    for row in range(0, rows, 1):
        remained = cols
        for col in range(0, cols, elems_per_vreg):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            out = pto.tpl("arithmetic", lhs, rhs, mask)
            pto.vsts(out, dst[row, col:], mask)
```

#### Example: Indexed Concatenation Kernel (tconcatidx)

`pto.tconcatidx` concatenates two source tiles along the column dimension with per-row index control. Two additional index tiles (`src0Idx`, `src1Idx`) specify the number of columns to copy from each source on a per-row basis.

**Constraint function:**

```python
def _supports_tconcatidx(src0, src1, src0Idx, src1Idx, dst) -> bool:
    if src0.rank != 2 or src1.rank != 2 or src0Idx.rank != 2 or src1Idx.rank != 2 or dst.rank != 2:
        return False
    if src0.config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src1.config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if dst.config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src0Idx.config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src1Idx.config.b_layout != pto.BLayout.ROW_MAJOR:
        return False
    if src0.valid_shape[0] != dst.valid_shape[0] or src1.valid_shape[0] != dst.valid_shape[0]:
        return False
    if src0Idx.valid_shape[0] != dst.valid_shape[0] or src1Idx.valid_shape[0] != dst.valid_shape[0]:
        return False
    if src0Idx.shape[1] < 1 or src1Idx.shape[1] < 1:
        return False
    return True
```

**Kernel definition:**

```python
@pto.vkernel(
    target="a5",
    op="pto.tconcatidx",
    advanced=True,
    constraints=[_supports_tconcatidx],
)
def template_tconcatidx(
    src0: pto.Tile,
    src1: pto.Tile,
    src0Idx: pto.Tile,
    src1Idx: pto.Tile,
    dst: pto.Tile,
):
    dtype = dst.element_type
    lanes = pto.get_lanes(dtype)
    idx_dtype = src0Idx.element_type
    idx_elem_bytes = pto.i32(pto.bytewidth(idx_dtype))
    src0_idx_ptr = src0Idx.as_ptr()
    src1_idx_ptr = src1Idx.as_ptr()
    dst_ptr = dst.as_ptr()

    valid_rows, dst_valid_cols = dst.valid_shape
    dst_valid_cols_i32 = pto.i32(dst_valid_cols)
    lanes_i32 = pto.i32(lanes)

    for row in range(0, valid_rows, 1):
        idx0_num = pto.load_scalar(src0_idx_ptr, row * src0Idx.shape[1]) // idx_elem_bytes
        idx1_num = pto.load_scalar(src1_idx_ptr, row * src1Idx.shape[1]) // idx_elem_bytes

        src0_cols = idx0_num
        if src0_cols > dst_valid_cols_i32:
            src0_cols = dst_valid_cols_i32

        src1_capacity = dst_valid_cols_i32 - src0_cols
        if src1_capacity < pto.i32(0):
            src1_capacity = pto.i32(0)

        src1_cols = idx1_num
        if src1_cols > src1_capacity:
            src1_cols = src1_capacity

        remained0 = src0_cols
        for col in range(0, dst_valid_cols, lanes):
            mask0, remained0 = pto.make_mask(dtype, remained0)
            vec0 = pto.vlds(src0[row, col:])
            pto.vsts(vec0, dst[row, col:], mask0)

        pto.mem_bar(pto.BarrierType.VST_VLD)

        remained1 = src1_cols
        for col in range(0, dst_valid_cols, lanes):
            active_lanes = remained1
            if active_lanes > lanes_i32:
                active_lanes = lanes_i32
            base = pto.i32(row * dst.shape[1] + col) + src0_cols
            vec1 = pto.vlds(src1[row, col:])
            offsets = pto.vci(base, pto.OrderMode.ASC)
            pto.vscatter(vec1, dst_ptr, offsets, active_lanes)
            remained1 = remained1 - active_lanes
```

**Parameter constraints:**

| Parameter | Rank | Layout | valid_row | Notes |
|-----------|------|--------|-----------|-------|
| `src0` | 2 | row_major | == dst.valid_row | Source data tile |
| `src1` | 2 | row_major | == dst.valid_row | Source data tile |
| `src0Idx` | 2 | row_major | == dst.valid_row | Index tile, cols >= 1 |
| `src1Idx` | 2 | row_major | == dst.valid_row | Index tile, cols >= 1 |
| `dst` | 2 | row_major | — | Destination tile |

**Type constraints:**

- Data types (`src0`, `src1`, `dst`): must be the same, one of `i8/i16/i32/f16/f32/bf16`
- Index types (`src0Idx`, `src1Idx`): must be the same, one of `i8/i16/i32`
- All tiles must use `loc=vec`

**Semantics (per row):**

Let `idx0_num = src0Idx[row, 0]`, `idx1_num = src1Idx[row, 0]`:
- Copy `min(idx0_num, src0_valid_col, dst_valid_col)` columns from `src0` to `dst`
- Copy `min(idx1_num, src1_valid_col, dst_valid_col - copied_from_src0)` columns from `src1` to `dst`

#### Compile-time Specialization with `pto.constexpr`

The `pto.constexpr` construct enables compile-time branching for kernel specialization, allowing different code paths to be selected based on static compile-time information. Unlike runtime conditionals that generate control flow, `pto.constexpr` branches are resolved during kernel descriptor materialization, with only the selected branch retained for lowering.

**Syntax and Usage**:
```python
if pto.constexpr(condition):
    # Branch taken if condition evaluates to True at compile time
    ...
else:
    # Branch taken if condition evaluates to False at compile time
    ...
```

**Semantics**:
- The `condition` must be evaluable at compile time during kernel descriptor materialization.
- Only the selected branch is analyzed, semantically checked, and lowered to VPTO IR.
- The non-selected branch is discarded entirely and does not contribute to runtime control flow or value merging.
- If the condition cannot be proven static, descriptor materialization fails with a frontend diagnostic.

**Comparison with Runtime Conditionals**:

| Aspect | Runtime `if` | `pto.constexpr` |
|--------|--------------|-----------------|
| **Evaluation time** | Runtime | Compile-time (descriptor materialization) |
| **Control flow** | Generates `scf.if` with merge logic | No runtime control flow; branch eliminated |
| **Value merging** | Both branches must produce compatible values for merge | No value merging; only one branch exists after elimination |
| **Use case** | Dynamic decision making based on runtime values | Code generation specialization based on static parameters |

**Typical Static Inputs**:
- Literal integers, booleans, and strings
- Data type symbols (`src.element_type`, `dst.element_type`) and comparisons derived from them
- Statically specialized `Tile.shape` and `Tile.valid_shape` values
- Frontend query helpers such as `pto.bytewidth(dtype)` and `pto.elements_per_vreg(dtype)` (which computes elements per vector register)

**Constraints and Notes**:
- `TensorView.shape` and `TensorView.strides` may be represented by hidden kernel parameters rather than descriptor-time constants. They should not be assumed constexpr unless separately bound through specialization or other compile-time context.
- `pto.constexpr` is a frontend-only authoring construct; it does not correspond to any runtime VPTO instruction.

**Guidelines**:
- Use `constraints=[...]` and `pto.select_kernel(...)` when specialization requires selecting an entirely different kernel descriptor.
- Use `pto.constexpr` when the kernel remains the same but internal regions require specialization based on compile-time parameters.

**Example**:
```python
@pto.vkernel(target="a5", op="pto.trowsum")
def template_trowsum(dst: pto.Tile, src: pto.Tile, tmp: pto.Tile):
    acc_dtype = tmp.element_type
    dst_dtype = dst.element_type
    acc_mask_1, _ = pto.make_mask(acc_dtype, 1)
    dst_mask_1, _ = pto.make_mask(dst_dtype, 1)

    if pto.constexpr(acc_dtype != dst_dtype):
        # Type conversion required
        v_acc_casted = pto.vcvt(v_acc, dst_dtype, acc_mask_1)
        pto.vsts(v_acc_casted, dst[row, 0:], dst_mask_1)
    else:
        # No conversion needed
        pto.vsts(v_acc, dst[row, 0:], dst_mask_1)
```

### Value Model

The DSL operates on symbolic values, not Python runtime values:
- **Constants**: Python literals that are typed to machine types
- **Operation results**: Values produced by DSL operations
- **Block arguments**: Values introduced by control flow structures

### Memory Spaces

The DSL supports different memory spaces:
- `MemorySpace.GM`: Global Memory
- `MemorySpace.UB`: Unified Buffer (local storage for vector computation)
