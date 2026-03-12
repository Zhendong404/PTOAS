# Level-3 OP-Lib `pto.simd` Authoring IR 规范

- 状态：Draft v3.0
- 定位：Target-State Authoring Spec
- 生效范围：PTOAS Level-3 OP-Lib 模板开发
- 目标读者：PTOAS OP-Lib 模板维护者、tile fusion / OP-Lib lowering 维护者、IR 库开发者

## 1. Overview

本文定义 **Level-3 OP-Lib 模板体**的作者可见 IR 契约。

它与 [docs/PTO_IR_manual.md](../PTO_IR_manual.md) 的关系如下：

1. `PTO_IR_manual.md` 定义上层 PTO public IR 的语义，例如 `pto.tadd`、`pto.tcmp`、`pto.trowsum`
2. 本文定义 OP-Lib 模板函数内部应如何编写这些语义的模板体
3. 对于已进入 OP-Lib 路线的向量类能力，模板作者统一使用 `pto.simd.*` 命名，不再以 `vector.*` / `arith.*` / `math.*` 作为规范化 authoring 接口

### 1.1 Status Note

当前仓库里仍存在历史 mixed-IR 模板、测试和 pass 约束，它们可能继续接受 `vector.*` / `arith.*` / `math.*` 形式。

**这些历史写法不是本文档定义的目标态 authoring contract。** 对于新模板和新 family，正文一律以 `pto.simd.*` 为规范描述。历史 mixed IR 只属于迁移背景，不属于正文规范。

### 1.2 Scope

本版文档覆盖当前 OP-Lib lowering 路线已经需要收敛的向量语义能力：

1. Float arithmetic / math
2. Reduction / broadcast
3. Compare / select
4. Integer bitwise / shift
5. SIMD bridge / load-store / predicate

本版不覆盖以下内容：

1. `PTO_IR_manual.md` 中不属于当前 OP-Lib 路线的上层能力
2. 历史 mixed IR 的兼容教程
3. matrix / DMA / sync 等非当前 `pto.simd` authoring 范围能力
4. `tpartadd` / `tpartmax` / `tpartmin` 这类尚未纳入当前 family 路线的 partial arithmetic

### 1.3 Capability Mapping

| 上层 PTO 语义 | 目标态 OP-Lib authoring 名称 |
|---------------|-------------------------------|
| `pto.tadd` / `pto.tsub` / `pto.tmul` / `pto.tdiv` | `pto.simd.add` / `sub` / `mul` / `div` |
| `pto.tmax` / `pto.tmin` / `pto.trem` / `pto.tprelu` | `pto.simd.max` / `min` / `rem` / `prelu` |
| 历史 `math.exp` / `math.log` / `math.sqrt` / `math.rsqrt` authoring 习惯 | `pto.simd.exp` / `log` / `sqrt` / `rsqrt` |
| `pto.trowsum` / `pto.trowmax` / `pto.trowmin` | `pto.simd.row_sum` / `row_max` / `row_min` |
| `pto.tcolsum` / `pto.tcolmax` / `pto.tcolmin` | `pto.simd.col_sum` / `col_max` / `col_min` |
| `pto.trowexpand` / `pto.tcolexpand` / `pto.trowexpandmul` / `pto.trowexpanddiv` / `pto.trowexpandsub` / `pto.texpands` | `pto.simd.row_expand` / `col_expand` / `row_expand_mul` / `row_expand_div` / `row_expand_sub` / `expand` |
| `pto.tcmp` / `pto.tcmps` | `pto.simd.cmp` |
| `pto.tsel` / `pto.tsels` | `pto.simd.select` |
| `pto.tand` / `pto.tands` / `pto.tor` / `pto.tors` / `pto.txor` / `pto.txors` / `pto.tshl` / `pto.tshls` / `pto.tshr` / `pto.tshrs` / `pto.tnot` | `pto.simd.and` / `or` / `xor` / `shl` / `shr` / `not` |

## 2. Authoring Model

### 2.1 Execution Model

OP-Lib 模板的调用关系如下：

```text
上层 PTO op
  -> 按 pto.oplib.kind 选择 family
  -> 匹配 variant / seed
  -> 以 !pto.tile_buf ABI 调用模板函数
  -> 模板体内部使用 pto.simd.* 表达计算与访存语义
  -> 实例化 / inline / backend lowering
```

模板函数的对外 ABI 继续保持 `!pto.tile_buf` 主导，**`pto.simd.*` 只定义模板体内部的 authoring contract**，不改变模板函数对外调用协议。

### 2.2 External ABI

1. 模板函数返回类型固定为 `()`
2. 参数个数由 `pto.oplib.kind` 决定
3. tile-like 参数使用 `!pto.tile_buf<...>`
4. scalar-like 参数使用 builtin scalar type，例如 `f32`、`i32`、`index`
5. `mask`、`tmp`、`dst` 在 ABI 层仍然是 tile-like / scalar-like 参数角色，不单独引入新的外部容器类型

### 2.3 Internal Body Model

模板体内部允许两类 `pto.simd` 语义层次：

1. **Lane-level ops**
   1. 直接操作 `vector<lanes x T>` 和 `vector<lanes x i1>`
   2. 典型 op：`load`、`store`、`add`、`cmp`、`select`、`and`
2. **Structured memory-level ops**
   1. 直接操作桥接后的 `memref` 视图和显式 `dst` / `tmp`
   2. 典型 op：`row_sum`、`col_sum`、`row_expand`、`expand`

其中：

1. Lane-level ops 适合编写显式 hardware loop / predicate / post-update 风格模板
2. Structured memory-level ops 适合直接表达某个 family 的规范语义骨架
3. 两类 op 都属于 `pto.simd.*` authoring 层，都可以出现在同一个模板体中

### 2.4 Memory Model

模板体内部继续保留作者可见 memref bridge：

1. `pto.simd.tile_to_memref` 用于从 `!pto.tile_buf` 暴露一个 backend-aware memref 视图
2. 允许必要的 `memref` 视图 / 重解释辅助，例如 `memref.reinterpret_cast`
3. 不允许在模板体中用 `memref.load` / `memref.store` 直接绕开 `pto.simd.load` / `pto.simd.store`

换句话说：

1. **tile ABI 仍然存在**
2. **memref bridge 仍然可见**
3. **规范化的访存语义统一写成 `pto.simd.load/store/load_pu/store_pu`**

### 2.5 Allowed Helper IR

除 `pto.simd.*` 外，模板体允许的辅助 IR 仅限以下类别：

1. `func`
2. `scf`
3. builtin scalar types：`f16`、`f32`、`i8`、`i16`、`i32`、`index` 等
4. `arith.constant`
5. `arith.index_cast`
6. 不直接读写数据的 `memref` 视图 / cast / reinterpret 辅助

### 2.6 Disallowed Authoring IR

在目标态规范中，以下形式不属于合法 authoring 接口：

1. 以 `vector.load` / `vector.store` 作为规范化访存语义
2. 以 `arith.addf` / `arith.maximumf` / `arith.andi` 等作为规范化计算语义
3. 以 `math.exp` / `math.log` / `math.sqrt` / `math.rsqrt` 作为规范化 math authoring 语义
4. `memref.load`
5. `memref.store`
6. `builtin.unrealized_conversion_cast`

## 3. Types, Attributes, and Naming

### 3.1 Author-Visible Types

| 类型 | 角色 | 说明 |
|------|------|------|
| `!pto.tile_buf<...>` | 模板外部 ABI | 模板函数形参 / `dst` / `tmp` / `mask` 的外部表示 |
| `memref<..., #pto.address_space<vec>>` | 模板内部桥接视图 | `pto.simd.tile_to_memref` 的结果类型 |
| `vector<lanes x T>` | lane value | `pto.simd.load`、`pto.simd.add`、`pto.simd.cmp` 等的操作数 / 结果 |
| `vector<lanes x i1>` | predicate mask | `pto.simd.predicate` 与 `pto.simd.cmp` 结果 |
| builtin scalar type | scalar operand | `f32`、`i32`、`index` 等 |

### 3.2 Common Attributes

#### Function-Level Attributes

| Attr | 位置 | 说明 |
|------|------|------|
| `pto.simd.level` | 模板函数属性 | SIMD authoring level / lowering profile |
| `pto.simd.lanes` | 模板函数属性 | 模板约定的 lane 宽度 |

目标态约束：

1. 非 legacy 的 `pto.simd` 模板应显式声明 `pto.simd.level`
2. 非 legacy 的 `pto.simd` 模板应显式声明正数 `pto.simd.lanes`
3. lane-level `pto.simd` op 的 vector lane 宽度必须与 `pto.simd.lanes` 一致

#### Op-Level Attributes

| Attr | 位置 | 说明 |
|------|------|------|
| `pto.simd.core_slot` | 单个核心算术 op | `seed` 改写槽位 |
| `pto.simd.vld_dist` | `pto.simd.load` / `load_pu` | A5 vector load 路径属性 |
| `pto.simd.vst_dist` | `pto.simd.store` / `store_pu` | A5 vector store 路径属性 |
| `pto.simd.exec_mode` | 计算 op | A5 vector arithmetic / compare / select 执行模式 |

### 3.3 Shared Enum-Like Domains

#### CmpMode

`pto.simd.cmp` 继承 `PTO_IR_manual` 中 `CmpMode` 语义。

| Value | Mnemonic |
|-------|----------|
| `EQ` | `equal` |
| `NE` | `not_equal` |
| `LT` | `less_than` |
| `LE` | `less_equal` |
| `GT` | `greater_than` |
| `GE` | `greater_equal` |

推荐属性写法：

```mlir
{mode = #pto<cmp less_than>}
```

#### SelectMode

`pto.simd.select` 的 scalar-mode overload 使用 builtin scalar operand `selectMode`，其值域继承对应上层 `pto.tsels` 语义，不单独发明新的公共枚举名字。

### 3.4 Naming Rules

`pto.simd.*` 的命名规则固定如下：

1. 从上层 `pto.t*` 语义名派生
2. 去掉前导 `t`
3. 多单词语义使用 `snake_case`
4. scalar 变体不再用 `...s` 后缀，而是通过操作数类型区分
5. compare / select / bitwise 的 scalar overload 与 tile-tile overload 共享同一 op 名

示例：

| 上层 PTO op | 新 authoring 名称 |
|-------------|-------------------|
| `pto.tadd` | `pto.simd.add` |
| `pto.tcmps` | `pto.simd.cmp` |
| `pto.tsels` | `pto.simd.select` |
| `pto.trowsum` | `pto.simd.row_sum` |
| `pto.texpands` | `pto.simd.expand` |
| `pto.tands` | `pto.simd.and` |

### 3.5 Overload Rules

以下语义族采用同名 overload：

1. `add` / `sub` / `mul` / `div` / `max` / `min`
   1. vector-vector
   2. vector-scalar
2. `cmp`
   1. vector-vector
   2. vector-scalar
3. `select`
   1. mask-vector-vector
   2. vector-vector-selectMode
4. `and` / `or` / `xor` / `shl` / `shr`
   1. integer vector-vector
   2. integer vector-scalar

## 4. Template Contract

### 4.1 `pto.oplib.kind`

所有模板继续使用 `pto.oplib.kind` 作为主分派键，不引入第二层总开关。

当前已确定的 kind 包括：

1. `l3_binary_elementwise_template`（legacy compatibility only）
2. `l3_float_binary_elementwise_template`
3. `l3_float_tile_scalar_template`
4. `l3_float_unary_template`
5. `l3_reduce_row_template`
6. `l3_reduce_col_template`
7. `l3_reduce_colsum_template`
8. `l3_broadcast_row_template`
9. `l3_broadcast_col_template`
10. `l3_broadcast_row_binary_template`
11. `l3_scalar_expand_template`
12. `l3_cmp_tile_tile_template`
13. `l3_cmp_tile_scalar_template`
14. `l3_select_mask_template`
15. `l3_select_scalar_template`
16. `l3_int_binary_elementwise_template`
17. `l3_int_tile_scalar_elementwise_template`
18. `l3_int_unary_template`

### 4.2 Signature Matrix

模板导入时按 `kind` 驱动签名校验，不再使用固定 “3 个 `tile_buf` 参数” 的全局规则。

支持的签名类别如下：

| Signature Class | 说明 |
|-----------------|------|
| `(tile, tile, dst)` | binary tile-tile family |
| `(tile, scalar, dst)` | tile-scalar family |
| `(tile, tile, tile, dst)` | ternary tile family |
| `(tile, scalar, tile, dst)` | ternary mixed family |
| `(tile, dst)` | unary / reduction / broadcast family |
| `(scalar, dst)` | scalar expand family |
| `(src, tmp, dst)` | family 需要显式 scratch / tmp |
| `(mask, src0, src1, dst)` | mask select family |
| `(src0, src1, selectMode, dst)` | scalar-mode select family |

已在 active OpenSpec 设计中锁定的代表性映射：

1. `l3_float_binary_elementwise_template` 使用 `(tile, tile, dst)`
2. `l3_cmp_tile_tile_template` 使用 `(tile, tile, dst)`
3. `l3_cmp_tile_scalar_template` 使用 `(tile, scalar, dst)`
4. `l3_select_mask_template` 使用 `(mask, src0, src1, dst)`
5. `l3_select_scalar_template` 使用 `(src0, src1, selectMode, dst)`
6. `l3_scalar_expand_template` 使用 `(scalar, dst)`

对于 reduction / broadcast family：

1. 其 ABI 仍然必须落在上述签名类别之一
2. 若 family 需要显式 `tmp`，则使用 `(src, tmp, dst)`
3. 若 family 不暴露 `tmp`，则使用 `(tile, dst)` 或 `(tile, tile, dst)`

### 4.3 Common Metadata

所有模板继续使用以下公共元数据：

1. `pto.oplib.kind`
2. `pto.oplib.entry_role`
3. `pto.oplib.op`
4. `pto.oplib.variant_id`
5. `pto.oplib.match.dtype`
6. `pto.oplib.cost`
7. `pto.oplib.priority`
8. `pto.oplib.sync`
9. `pto.oplib.seed.*`

### 4.4 Legacy Match Metadata

仅 `l3_binary_elementwise_template` 继续兼容 legacy 匹配字段：

1. `pto.oplib.match.rows`
2. `pto.oplib.match.cols`
3. `pto.oplib.match.blayout`
4. `pto.oplib.match.slayout`
5. `pto.oplib.match.fractal`

### 4.5 `argN.*` Match Metadata

除 legacy family 外，其余新 family 一律使用按参数编号的匹配元数据：

1. `pto.oplib.match.argN.rows`
2. `pto.oplib.match.argN.cols`
3. `pto.oplib.match.argN.blayout`
4. `pto.oplib.match.argN.slayout`
5. `pto.oplib.match.argN.fractal`

约束如下：

1. `N` 按函数参数编号，从 `0` 开始
2. 仅 tile-like 参数允许声明 `argN.*`
3. 若第 `N` 个参数是 scalar，则声明 `argN.*` 视为硬错误
4. 新 family 的 tile-like 参数必须完整声明该组元数据
5. `rows` / `cols` / `fractal` 允许使用 `-1` 表示 wildcard
6. `blayout` / `slayout` 允许值为 `row_major`、`col_major`、`none_box`、`any`

### 4.6 Family-Specific Match Metadata

当前 family-specific attr matching 包括：

1. `pto.oplib.match.scalar_pos`
2. `pto.oplib.match.cmp_mode`
3. `pto.oplib.match.is_binary`

约束如下：

1. `scalar_pos` 必须指向一个 scalar 参数位置
2. `l3_cmp_tile_tile_template` 与 `l3_cmp_tile_scalar_template` 必须提供 `cmp_mode`
3. `l3_reduce_colsum_template` 必须提供 `is_binary`
4. 其他 family 若未声明依赖上述元数据，则不得伪造无意义字段

### 4.7 `variant` and `seed`

继续保留两类入口：

1. `variant`
2. `seed`

`variant` 需要：

1. `pto.oplib.op`
2. `pto.oplib.variant_id`
3. `pto.oplib.match.dtype`

`seed` 需要：

1. `pto.oplib.seed_id`
2. `pto.oplib.seed_dtype`
3. `pto.oplib.seed.support_dtypes`
4. `pto.oplib.seed.support_ops`
5. 可选 `pto.oplib.seed.core_slot`

### 4.8 `seed` Core Slot Rules

首版 `seed` 仍服务于单一 core-slot family。

当前强约束：

1. 使用 `pto.simd.core_slot` 的模板必须且仅能有一个 core slot op
2. 现阶段 seed core slot 仍限定为 float binary arithmetic
3. 合法的 core slot 语义 op 为：
   1. `pto.simd.add`
   2. `pto.simd.sub`
   3. `pto.simd.mul`
   4. `pto.simd.div`
   5. `pto.simd.max`
   6. `pto.simd.min`
4. 非单 core-slot family 不应复用首版 seed 改写模型

### 4.9 Instance Key

实例函数缓存键由以下部分组成：

1. `variant_id`
2. family-specific attr choices
3. 所有 concrete argument types

因此实例化不再假设固定 3 个参数。

## 5. `pto.simd` Operations Reference

本节给出目标态 authoring op reference。

说明：

1. `Authoring Form` 是规范化写法，可能比当前仓库中已实现的 assembly 更超前
2. 对于继承自上层 `pto.t*` 的数学语义，若本文未额外说明，则默认与 `PTO_IR_manual.md` 对应条目一致
3. lane-level op 的 `T` 表示 element type，`lanes` 表示函数级 `pto.simd.lanes`

### 5.1 Bridge and Memory Operations

#### `pto.simd.vec_scope`

**Summary:** 标记需要放在 `__VEC_SCOPE__` 下执行的 region。

**Semantics:** 包裹一个单块 region；region 内的 SIMD memory / compute op 视为在同一 vector scope 中执行。

**Arguments:** 一个单 block region。

**Results:** 无。

**Authoring Form:**

```mlir
pto.simd.vec_scope {
  ...
}
```

**Constraints:**

1. region 必须为单 block
2. 通常用于显式 loop + load/store 模板

**Example:**

```mlir
pto.simd.vec_scope {
  scf.for %i = %c0 to %c1024 step %c64 {
    ...
  }
}
```

#### `pto.simd.tile_to_memref`

**Summary:** 从 `!pto.tile_buf` 暴露一个 backend-aware memref 视图。

**Semantics:** 将 tile-like 源值桥接为 memref 视图；不改变对外 ABI。

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `!pto.tile_buf<...>` 或 lowering 后的等价 memref | tile source |

**Results:** 一个 memref bridge value。

**Authoring Form:**

```mlir
%m = pto.simd.tile_to_memref %src : !pto.tile_buf<...> to memref<...>
```

**Constraints:**

1. 结果 memref 类型必须与源 tile 的 dtype / layout / address space 兼容
2. 允许在 memref-world lowering 后保留为 backend marker

**Example:**

```mlir
%m = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
```

#### `pto.simd.predicate`

**Summary:** 根据 active lane count 构造 predicate mask。

**Semantics:** 对每个 lane `i`，若 `i < active_count`，则 `mask[i] = 1`，否则 `mask[i] = 0`。

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `active_count` | `index` | 活跃 lane 数 |

**Results:** `vector<lanes x i1>` mask。

**Authoring Form:**

```mlir
%mask = pto.simd.predicate %active_count : index -> vector<lanesxi1>
```

**Constraints:**

1. result lane 数必须与 `pto.simd.lanes` 一致
2. `active_count` 可以小于或等于 `lanes`

**Example:**

```mlir
%mask = pto.simd.predicate %c64 : index -> vector<64xi1>
```

#### `pto.simd.load`

**Summary:** 使用显式 mask 从线性 memref 偏移执行 lane load。

**Semantics:** 对每个 lane `i`，若 `mask[i] = 1`，读取 `src[offset + i]`；否则结果 lane 按 backend 约定处理。

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | memref | 源 memref |
| `offset` | `index` | 线性起始偏移 |
| `mask` | `vector<lanes x i1>` | predicate mask |

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%value = pto.simd.load %src, %offset, %mask {pto.simd.vld_dist = "NORM"} : memref<...>, index, vector<lanesxi1> -> vector<lanesxT>
```

**Constraints:**

1. result lane 数必须与 `pto.simd.lanes` 一致
2. A5 路径要求 `pto.simd.vld_dist`

**Example:**

```mlir
%a = pto.simd.load %flat0, %i, %mask {pto.simd.vld_dist = "NORM"} : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
```

#### `pto.simd.store`

**Summary:** 使用显式 mask 向线性 memref 偏移执行 lane store。

**Semantics:** 对每个 lane `i`，若 `mask[i] = 1`，写回 `dst[offset + i] = value[i]`。

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `vector<lanes x T>` | 待写回数据 |
| `dst` | memref | 目标 memref |
| `offset` | `index` | 线性起始偏移 |
| `mask` | `vector<lanes x i1>` | predicate mask |

**Results:** 无。

**Authoring Form:**

```mlir
pto.simd.store %value, %dst, %offset, %mask {pto.simd.vst_dist = "DIST_NORM"} : vector<lanesxT>, memref<...>, index, vector<lanesxi1>
```

**Constraints:**

1. value lane 数必须与 `pto.simd.lanes` 一致
2. A5 路径要求 `pto.simd.vst_dist`

**Example:**

```mlir
pto.simd.store %c, %flatd, %i, %mask {pto.simd.vst_dist = "DIST_NORM"} : vector<64xf32>, memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1>
```

#### `pto.simd.load_pu`

**Summary:** 带 post-update 偏移的 masked load。

**Semantics:** 执行一次 `load`，同时返回 `next_offset = offset + step`。

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | memref | 源 memref |
| `offset` | `index` | 当前偏移 |
| `mask` | `vector<lanes x i1>` | predicate mask |
| `step` | `i64 attr` | post-update 步长 |

**Results:** `value` 与 `next_offset`。

**Authoring Form:**

```mlir
%value, %next = pto.simd.load_pu %src, %offset, %mask {step = 64 : i64, pto.simd.vld_dist = "NORM"} : memref<...>, index, vector<lanesxi1> -> vector<lanesxT>, index
```

**Constraints:**

1. 结果 vector lane 数必须与 `pto.simd.lanes` 一致
2. `step` 必须与模板访问模式一致

**Example:**

```mlir
%a, %next = pto.simd.load_pu %flat0, %i, %mask {step = 64 : i64, pto.simd.vld_dist = "NORM"} : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>, index
```

#### `pto.simd.store_pu`

**Summary:** 带 post-update 偏移的 masked store。

**Semantics:** 执行一次 `store`，同时返回 `next_offset = offset + step`。

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `vector<lanes x T>` | 待写回数据 |
| `dst` | memref | 目标 memref |
| `offset` | `index` | 当前偏移 |
| `mask` | `vector<lanes x i1>` | predicate mask |
| `step` | `i64 attr` | post-update 步长 |

**Results:** `next_offset`。

**Authoring Form:**

```mlir
%next = pto.simd.store_pu %value, %dst, %offset, %mask {step = 64 : i64, pto.simd.vst_dist = "DIST_NORM"} : vector<lanesxT>, memref<...>, index, vector<lanesxi1> -> index
```

**Constraints:**

1. value lane 数必须与 `pto.simd.lanes` 一致
2. `step` 必须与模板访问模式一致

**Example:**

```mlir
%next = pto.simd.store_pu %c, %flatd, %i, %mask {step = 64 : i64, pto.simd.vst_dist = "DIST_NORM"} : vector<64xf32>, memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> index
```

### 5.2 Arithmetic and Math Operations

#### `pto.simd.add`

**Summary:** lane-wise elementwise add。

**Semantics:** `result[i] = lhs[i] + rhs[i]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `vector<lanes x T>` | 左操作数 |
| `rhs` | `vector<lanes x T>` 或 `T` | 右操作数 |

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.add %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.add %lhs, %scalar {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:**

1. `lhs` 和 vector `rhs` 的 lane 数必须等于 `pto.simd.lanes`
2. scalar overload 只用于 family 允许的 tile-scalar 场景
3. `seed` core slot 允许把该 op 标为 `pto.simd.core_slot = "binary_ewise_core"`

**Example:**

```mlir
%c = pto.simd.add %a, %b {pto.simd.core_slot = "binary_ewise_core", pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.sub`

**Summary:** lane-wise elementwise subtract。

**Semantics:** `result[i] = lhs[i] - rhs[i]`

**Arguments:** 同 `add`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.sub %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.sub %lhs, %scalar {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `add`。

**Example:**

```mlir
%r = pto.simd.sub %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.mul`

**Summary:** lane-wise elementwise multiply。

**Semantics:** `result[i] = lhs[i] * rhs[i]`

**Arguments:** 同 `add`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.mul %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.mul %lhs, %scalar {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `add`。

**Example:**

```mlir
%r = pto.simd.mul %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.div`

**Summary:** lane-wise elementwise divide。

**Semantics:** `result[i] = lhs[i] / rhs[i]`

**Arguments:** 同 `add`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.div %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.div %lhs, %scalar {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `add`。

**Example:**

```mlir
%r = pto.simd.div %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.max`

**Summary:** lane-wise maximum。

**Semantics:** `result[i] = max(lhs[i], rhs[i])`

**Arguments:** 同 `add`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.max %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.max %lhs, %scalar {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `add`。

**Example:**

```mlir
%r = pto.simd.max %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.min`

**Summary:** lane-wise minimum。

**Semantics:** `result[i] = min(lhs[i], rhs[i])`

**Arguments:** 同 `add`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.min %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.min %lhs, %scalar {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `add`。

**Example:**

```mlir
%r = pto.simd.min %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.rem`

**Summary:** lane-wise remainder / fmod。

**Semantics:** `result[i] = rem(lhs[i], rhs[i])`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `vector<lanes x T>` | 左操作数 |
| `rhs` | `vector<lanes x T>` 或 `T` | 右操作数 |

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.rem %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.rem %lhs, %scalar {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:**

1. element type 必须属于 family 允许的 numeric domain
2. lane 数必须与 `pto.simd.lanes` 一致

**Example:**

```mlir
%r = pto.simd.rem %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.prelu`

**Summary:** lane-wise PReLU。

**Semantics:** `result[i] = lhs[i] > 0 ? lhs[i] : slope[i] * lhs[i]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `vector<lanes x T>` | 输入值 |
| `slope` | `vector<lanes x T>` 或 `T` | 斜率 |

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.prelu %lhs, %slope {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.prelu %lhs, %scalar_slope {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:**

1. lane 数必须与 `pto.simd.lanes` 一致
2. numeric domain 必须与上层 `pto.tprelu` 语义兼容

**Example:**

```mlir
%r = pto.simd.prelu %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.exp`

**Summary:** lane-wise exponential。

**Semantics:** `result[i] = exp(src[i])`

**Arguments:** `src : vector<lanes x T>`

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.exp %src {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT> -> vector<lanesxT>
```

**Constraints:**

1. `T` 必须是 family 支持的浮点类型
2. lane 数必须与 `pto.simd.lanes` 一致

**Example:**

```mlir
%r = pto.simd.exp %a {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.log`

**Summary:** lane-wise natural logarithm。

**Semantics:** `result[i] = log(src[i])`

**Arguments:** `src : vector<lanes x T>`

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.log %src {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT> -> vector<lanesxT>
```

**Constraints:** 同 `exp`。

**Example:**

```mlir
%r = pto.simd.log %a {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.sqrt`

**Summary:** lane-wise square root。

**Semantics:** `result[i] = sqrt(src[i])`

**Arguments:** `src : vector<lanes x T>`

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.sqrt %src {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT> -> vector<lanesxT>
```

**Constraints:** 同 `exp`。

**Example:**

```mlir
%r = pto.simd.sqrt %a {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32> -> vector<64xf32>
```

#### `pto.simd.rsqrt`

**Summary:** lane-wise reciprocal square root。

**Semantics:** `result[i] = rsqrt(src[i])`

**Arguments:** `src : vector<lanes x T>`

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.rsqrt %src {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT> -> vector<lanesxT>
```

**Constraints:** 同 `exp`。

**Example:**

```mlir
%r = pto.simd.rsqrt %a {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32> -> vector<64xf32>
```

### 5.3 Reduction and Broadcast Operations

以下 structured op 直接继承对应上层 PTO op 的 tile 语义，但它们工作在模板体内部的桥接 memref 视图上。

#### `pto.simd.row_sum`

**Summary:** 行方向 sum reduction。

**Semantics:** `dst[i, 0] = sum_j src[i, j]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | memref | 源 tile view |
| `dst` | memref | 行向 reduction 输出 view |

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.row_sum %src, %dst : memref<MxNxT>, memref<Mx1xT>
```

**Constraints:**

1. `dst` 逻辑形状必须与 `src` 的行数一致
2. `kind` 必须属于 row reduction family

**Example:**

```mlir
pto.simd.row_sum %ms, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>
```

#### `pto.simd.row_max`

**Summary:** 行方向 max reduction。

**Semantics:** `dst[i, 0] = max_j src[i, j]`

**Arguments:** 同 `row_sum`。

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.row_max %src, %dst : memref<MxNxT>, memref<Mx1xT>
```

**Constraints:** 同 `row_sum`。

**Example:**

```mlir
pto.simd.row_max %ms, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>
```

#### `pto.simd.row_min`

**Summary:** 行方向 min reduction。

**Semantics:** `dst[i, 0] = min_j src[i, j]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | memref | 源 tile view |
| `tmp` | memref（若 family 暴露 scratch） | reduction scratch |
| `dst` | memref | 行向 reduction 输出 view |

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.row_min %src, %dst : memref<MxNxT>, memref<Mx1xT>
pto.simd.row_min %src, %tmp, %dst : memref<MxNxT>, memref<...>, memref<Mx1xT>
```

**Constraints:**

1. 若 family ABI 暴露 `tmp`，模板必须显式传递
2. `tmp` 的具体形状由 family 契约决定，不得自行省略或伪造

**Example:**

```mlir
pto.simd.row_min %ms, %mtmp, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>
```

#### `pto.simd.col_sum`

**Summary:** 列方向 sum reduction。

**Semantics:** `dst[0, j] = sum_i src[i, j]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | memref | 源 tile view |
| `tmp` | memref | reduction scratch |
| `dst` | memref | 列向 reduction 输出 view |
| `isBinary` | `i1` / `bool attr` | 对应 `l3_reduce_colsum_template` 匹配语义 |

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.col_sum %src, %tmp, %dst {isBinary = false} : memref<MxNxT>, memref<...>, memref<1xNxT>
```

**Constraints:**

1. `l3_reduce_colsum_template` 必须提供 `pto.oplib.match.is_binary`
2. `tmp` 是该 family 的显式 ABI 角色之一

**Example:**

```mlir
pto.simd.col_sum %ms, %mtmp, %md {isBinary = false} : memref<32x32xf32, #pto.address_space<vec>>, memref<1x32xf32, #pto.address_space<vec>>, memref<1x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.col_max`

**Summary:** 列方向 max reduction。

**Semantics:** `dst[0, j] = max_i src[i, j]`

**Arguments:** `src` 与 `dst`，必要时由 family 契约补充 scratch。

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.col_max %src, %dst : memref<MxNxT>, memref<1xNxT>
pto.simd.col_max %src, %tmp, %dst : memref<MxNxT>, memref<...>, memref<1xNxT>
```

**Constraints:** 必须遵循具体 family ABI。

**Example:**

```mlir
pto.simd.col_max %ms, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<1x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.col_min`

**Summary:** 列方向 min reduction。

**Semantics:** `dst[0, j] = min_i src[i, j]`

**Arguments:** `src` 与 `dst`，必要时由 family 契约补充 scratch。

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.col_min %src, %dst : memref<MxNxT>, memref<1xNxT>
pto.simd.col_min %src, %tmp, %dst : memref<MxNxT>, memref<...>, memref<1xNxT>
```

**Constraints:** 必须遵循具体 family ABI。

**Example:**

```mlir
pto.simd.col_min %ms, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<1x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.row_expand`

**Summary:** 将 `src[i, 0]` 广播到整行。

**Semantics:** `dst[i, j] = src[i, 0]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | memref | 行向输入 view |
| `dst` | memref | 广播输出 view |

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.row_expand %src, %dst : memref<Mx1xT>, memref<MxNxT>
```

**Constraints:** `dst` 行数必须与 `src` 一致。

**Example:**

```mlir
pto.simd.row_expand %ms, %md : memref<32x1xf32, #pto.address_space<vec>>, memref<32x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.col_expand`

**Summary:** 将 `src[0, j]` 广播到整列。

**Semantics:** `dst[i, j] = src[0, j]`

**Arguments:** `src : memref<1xNxT>`，`dst : memref<MxNxT>`。

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.col_expand %src, %dst : memref<1xNxT>, memref<MxNxT>
```

**Constraints:** `dst` 列数必须与 `src` 一致。

**Example:**

```mlir
pto.simd.col_expand %ms, %md : memref<1x32xf32, #pto.address_space<vec>>, memref<32x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.row_expand_mul`

**Summary:** 行广播乘法。

**Semantics:** `dst[i, j] = src0[i, j] * src1[i, 0]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | memref | 主输入 |
| `src1` | memref | 行广播输入 |
| `dst` | memref | 输出 |

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.row_expand_mul %src0, %src1, %dst : memref<MxNxT>, memref<Mx1xT>, memref<MxNxT>
```

**Constraints:** `src1` 的行数必须与 `src0` 一致。

**Example:**

```mlir
pto.simd.row_expand_mul %ma, %mb, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>, memref<32x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.row_expand_div`

**Summary:** 行广播除法。

**Semantics:** `dst[i, j] = src0[i, j] / src1[i, 0]`

**Arguments:** 同 `row_expand_mul`。

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.row_expand_div %src0, %src1, %dst : memref<MxNxT>, memref<Mx1xT>, memref<MxNxT>
```

**Constraints:** 同 `row_expand_mul`。

**Example:**

```mlir
pto.simd.row_expand_div %ma, %mb, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>, memref<32x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.row_expand_sub`

**Summary:** 行广播减法。

**Semantics:** `dst[i, j] = src0[i, j] - src1[i, 0]`

**Arguments:** 同 `row_expand_mul`。

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.row_expand_sub %src0, %src1, %dst : memref<MxNxT>, memref<Mx1xT>, memref<MxNxT>
```

**Constraints:** 同 `row_expand_mul`。

**Example:**

```mlir
pto.simd.row_expand_sub %ma, %mb, %md : memref<32x32xf32, #pto.address_space<vec>>, memref<32x1xf32, #pto.address_space<vec>>, memref<32x32xf32, #pto.address_space<vec>>
```

#### `pto.simd.expand`

**Summary:** 将 scalar 广播为整 tile。

**Semantics:** `dst[i, j] = scalar`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `scalar` | builtin scalar | 广播值 |
| `dst` | memref | 输出 |

**Results:** 无，写入 `dst`。

**Authoring Form:**

```mlir
pto.simd.expand %scalar, %dst : T, memref<MxNxT>
```

**Constraints:** 必须对应 `l3_scalar_expand_template`。

**Example:**

```mlir
pto.simd.expand %alpha, %md : f32, memref<32x32xf32, #pto.address_space<vec>>
```

### 5.4 Compare and Select Operations

#### `pto.simd.cmp`

**Summary:** elementwise compare，结果为 predicate mask。

**Semantics:** `result[i] = (lhs[i] <cmpMode> rhs[i]) ? 1 : 0`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `vector<lanes x T>` | 左操作数 |
| `rhs` | `vector<lanes x T>` 或 `T` | 右操作数 |
| `mode` | `CmpMode` attr | 比较模式 |

**Results:** `vector<lanes x i1>`。

**Authoring Form:**

```mlir
%mask = pto.simd.cmp %lhs, %rhs {mode = #pto<cmp less_than>, pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT> -> vector<lanesxi1>
%mask = pto.simd.cmp %lhs, %scalar {mode = #pto<cmp less_than>, pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, T -> vector<lanesxi1>
```

**Constraints:**

1. lane 数必须与 `pto.simd.lanes` 一致
2. `l3_cmp_*` family 必须声明 `pto.oplib.match.cmp_mode`

**Example:**

```mlir
%mask = pto.simd.cmp %a, %b {mode = #pto<cmp greater_than>, pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>, vector<64xf32> -> vector<64xi1>
```

#### `pto.simd.select`

**Summary:** 支持 mask select 与 scalar-mode select 的统一选择语义。

**Semantics:**

1. mask overload：`result[i] = mask[i] ? on_true[i] : on_false[i]`
2. scalar-mode overload：语义继承上层 `pto.tsels`

**Arguments:**

| Overload | Arguments |
|----------|-----------|
| mask select | `mask : vector<lanes x i1>`，`on_true : vector<lanes x T>`，`on_false : vector<lanes x T>` |
| scalar-mode select | `lhs : vector<lanes x T>`，`rhs : vector<lanes x T>`，`selectMode : builtin scalar` |

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.select %mask, %on_true, %on_false {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxi1>, vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.select %lhs, %rhs, %mode {pto.simd.exec_mode = "MODE_ZEROING"} : vector<lanesxT>, vector<lanesxT>, S -> vector<lanesxT>
```

**Constraints:**

1. mask overload 对应 `l3_select_mask_template`
2. scalar-mode overload 对应 `l3_select_scalar_template`
3. lane 数必须与 `pto.simd.lanes` 一致

**Example:**

```mlir
%r = pto.simd.select %mask, %a, %b {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xi1>, vector<64xf32>, vector<64xf32> -> vector<64xf32>
```

### 5.5 Bitwise and Shift Operations

#### `pto.simd.and`

**Summary:** integer lane-wise bitwise AND。

**Semantics:** `result[i] = lhs[i] & rhs[i]`

**Arguments:** `lhs : vector<lanes x T>`，`rhs : vector<lanes x T>` 或 `T`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.and %lhs, %rhs : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.and %lhs, %scalar : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:**

1. `T` 必须是整数 element type
2. lane 数必须与 `pto.simd.lanes` 一致

**Example:**

```mlir
%r = pto.simd.and %a, %b : vector<32xi16>, vector<32xi16> -> vector<32xi16>
```

#### `pto.simd.or`

**Summary:** integer lane-wise bitwise OR。

**Semantics:** `result[i] = lhs[i] | rhs[i]`

**Arguments:** 同 `and`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.or %lhs, %rhs : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.or %lhs, %scalar : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `and`。

**Example:**

```mlir
%r = pto.simd.or %a, %b : vector<32xi16>, vector<32xi16> -> vector<32xi16>
```

#### `pto.simd.xor`

**Summary:** integer lane-wise bitwise XOR。

**Semantics:** `result[i] = lhs[i] ^ rhs[i]`

**Arguments:** 同 `and`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.xor %lhs, %rhs : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.xor %lhs, %scalar : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `and`。

**Example:**

```mlir
%r = pto.simd.xor %a, %b : vector<32xi16>, vector<32xi16> -> vector<32xi16>
```

#### `pto.simd.shl`

**Summary:** integer lane-wise shift left。

**Semantics:** `result[i] = lhs[i] << rhs[i]`

**Arguments:** `lhs : vector<lanes x T>`，`rhs : vector<lanes x T>` 或 `T`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.shl %lhs, %rhs : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.shl %lhs, %scalar : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** `T` 必须是整数类型；lane 数必须与 `pto.simd.lanes` 一致。

**Example:**

```mlir
%r = pto.simd.shl %a, %b : vector<32xi16>, vector<32xi16> -> vector<32xi16>
```

#### `pto.simd.shr`

**Summary:** integer lane-wise shift right。

**Semantics:** `result[i] = lhs[i] >> rhs[i]`

**Arguments:** 同 `shl`。

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.shr %lhs, %rhs : vector<lanesxT>, vector<lanesxT> -> vector<lanesxT>
%r = pto.simd.shr %lhs, %scalar : vector<lanesxT>, T -> vector<lanesxT>
```

**Constraints:** 同 `shl`。

**Example:**

```mlir
%r = pto.simd.shr %a, %b : vector<32xi16>, vector<32xi16> -> vector<32xi16>
```

#### `pto.simd.not`

**Summary:** integer lane-wise bitwise NOT。

**Semantics:** `result[i] = ~src[i]`

**Arguments:** `src : vector<lanes x T>`

**Results:** `vector<lanes x T>`。

**Authoring Form:**

```mlir
%r = pto.simd.not %src : vector<lanesxT> -> vector<lanesxT>
```

**Constraints:**

1. `T` 必须是整数类型
2. lane 数必须与 `pto.simd.lanes` 一致

**Example:**

```mlir
%r = pto.simd.not %a : vector<32xi16> -> vector<32xi16>
```

## 6. Validation, Errors, and Directory Rules

### 6.1 Hard Validation Rules

模板导入阶段会执行以下硬校验：

1. 模板签名与 `kind` 对应的签名类别一致
2. 新 family 的 tile-like 参数完整声明 `argN.*`
3. `scalar_pos` / `cmp_mode` / `is_binary` 在对应 family 上存在且合法
4. 模板体不出现白名单外 authoring IR
5. 使用 lane-level `pto.simd.*` 时，vector lane 宽度与 `pto.simd.lanes` 一致
6. A5 vector 类型、element type、lane 组合必须在允许集合内
7. `seed` 模板若声明 `pto.simd.core_slot`，必须且仅能有一个 core slot op
8. `seed` core slot op 只能是 `add` / `sub` / `mul` / `div` / `max` / `min`
9. 禁止空模板体和 fake-body fallback

### 6.2 Target-State Allowlist

模板体中允许出现的规范化 authoring 语义包括：

1. `pto.simd.*`
2. `func`
3. `scf`
4. `arith.constant`
5. `arith.index_cast`
6. 无直接数据访问的 `memref` 视图 / cast / reinterpret 辅助

### 6.3 Explicitly Disallowed Patterns

以下情况会在模板导入阶段或后续校验阶段视为硬失败：

1. 空模板体
2. `builtin.unrealized_conversion_cast`
3. `memref.load`
4. `memref.store`
5. 把 `vector.*` / `arith.*` / `math.*` 计算语义当作新模板的规范 authoring 接口
6. 不在 allowlist 中的其他 dialect / op

### 6.4 Typical Error Codes

| Error Code | 含义 |
|------------|------|
| `E_OPLIB_EMPTY_BODY_FOR_SIMD` | 模板体为空或签名不合法 |
| `E_OPLIB_SIMD_LANES_MISMATCH` | lane 宽度与 `pto.simd.lanes` 不一致 |
| `E_OPLIB_SIMD_INVALID_CORE_SLOT` | core slot 个数、类型或顺序非法 |
| `E_OPLIB_SIMD_UNSUPPORTED_DTYPE` | dtype 不在当前实现允许集合内 |
| `E_OPLIB_SIMD_UNSUPPORTED_LAYOUT` | layout 不在当前实现允许集合内 |
| `E_OPLIB_INSTANCE_BODY_MISSING` | 实例函数缺失 body |
| `E_OPLIB_BODY_DISALLOWED_IR` | 模板体包含白名单外 IR |
| `E_OPLIB_SIMD_ATTR_REQUIRED` | 缺少 `pto.simd` 必需属性 |

### 6.5 Directory and Test Rules

Level-3 模板源码目录统一为：

1. `oplib/level3/`

lit / 资源约束：

1. `--op-lib-dir` 应指向 `oplib/level3/`
2. 不再把 mixed-IR 模板作为第二份规范源
3. 基础设施负测资源应与新 spec 的错误码和校验规则保持一致

### 6.6 Authoring Recommendations

建议开发顺序：

1. 先确定 family 的 `kind` 和签名类别
2. 再写齐 `argN.*` 与 family-specific 元数据
3. 然后选择 lane-level 或 structured memory-level `pto.simd` 写法
4. 最后再补 A5 专用属性，例如 `vld_dist` / `vst_dist` / `exec_mode`

## 7. Quick Reference

1. 新模板体的计算 / 访存语义统一写成 `pto.simd.*`
2. 对外 ABI 继续是 `!pto.tile_buf`
3. `tile_to_memref` 与必要 memref 视图辅助仍可见
4. scalar 变体统一使用同名 overload，不再使用 `...s` 后缀
5. legacy mixed IR 不是本规范的一部分
