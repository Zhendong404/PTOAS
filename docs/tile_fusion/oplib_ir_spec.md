# Level-3 OP-Lib IR 接口规范（含 Seed 自动实例化）

- 状态：Draft v0.2
- 生效范围：`docs` 规范层，不要求当前代码立即实现
- 目标读者：PTOAS OP Fusion 维护者、OP-Lib 开发者

## 1. 设计目标与非目标

### 1.1 目标

本文定义 Level-3 OP-Lib 与 PTOAS 之间的接口契约，满足以下能力：

1. 输入输出以 `tile_buf` 语义建模。
2. 每个 OP 支持多版本并存与注册。
3. 支持 `seed` 自动实例化，实例化维度同时覆盖 `dtype + op name`。
4. 通过模板函数属性提供融合选择信息（静态 cost model）。
5. 版本选择行为确定性、可复现。
6. 降低 OP-Lib 开发与维护成本（尤其是 `tadd/tsub/tmul/tdiv` 共用骨架场景）。

首批 OP 范围限定为：

- `tmul`
- `tdiv`
- `tadd`
- `tsub`

### 1.2 非目标

1. 本文不定义 pass 代码实现细节。
2. 本文不要求与现有 V1 memref OP-Lib 机制立即兼容运行。
3. 本文不覆盖 reduce/expand/scalar 变体。
4. 本文不引入动态可执行 cost DSL（仅静态数值 cost）。

## 2. Level-3 接口模型

### 2.1 消费点约定

Level-3 OP-Lib 的版本匹配与选择逻辑，定义在**逻辑上位于 `PTOViewToMemref` 之前**的阶段，依据 `tile_buf` 元信息（dtype/shape/layout/fractal）完成候选过滤与选择。

说明：

1. 这是接口契约，不代表当前实现状态。
2. 该约束用于保证选择决策和 `tile_buf` 语义一致。

### 2.2 注册模式

注册单位为 `func.func`。支持两种入口模式：

1. `variant`：直接提供具体可选版本。
2. `seed`：提供可自动实例化的种子版本。

要求：

1. 同一 OP 允许多个 `variant` 并存。
2. 同一 `seed` 可覆盖多个 `dtype` 与多个 `op`（`tadd/tsub/tmul/tdiv`）。
3. 不依赖函数名编码语义，必须依赖函数属性驱动。

### 2.3 函数签名约束

首批二元逐元素 OP 的模板签名约束如下：

```mlir
(!pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) -> ()
```

按语义顺序为：

1. `src0`
2. `src1`
3. `dst`

### 2.4 属性契约（`pto.oplib.*`）

所有 Level-3 模板函数使用 `pto.oplib.*` 前缀属性。字段分为通用字段、`variant` 字段、`seed` 字段。

#### 2.4.1 通用字段

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `pto.oplib.kind` | `string` | 是 | 固定为 `l3_binary_elementwise_template` |
| `pto.oplib.entry_role` | `string` | 是 | `variant` 或 `seed` |
| `pto.oplib.match.rows` | `i64` | 是 | 行数精确匹配；`-1` 表示通配 |
| `pto.oplib.match.cols` | `i64` | 是 | 列数精确匹配；`-1` 表示通配 |
| `pto.oplib.match.blayout` | `string` | 是 | `row_major/col_major/any` |
| `pto.oplib.match.slayout` | `string` | 是 | `none_box/row_major/col_major/any` |
| `pto.oplib.match.fractal` | `i64` | 是 | 精确匹配；`-1` 表示通配 |
| `pto.oplib.cost` | `i64` | 是 | 静态代价，数值越小越优 |
| `pto.oplib.priority` | `i64` | 是 | 同 cost 时优先级，数值越大越优 |
| `pto.oplib.sync` | `bool` | 否 | 是否包含显式同步语义，默认 `false` |

#### 2.4.2 `variant` 字段（`entry_role = "variant"`）

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `pto.oplib.op` | `string` | 是 | 取值：`tmul/tdiv/tadd/tsub` |
| `pto.oplib.variant_id` | `string` | 是 | 版本唯一标识 |
| `pto.oplib.match.dtype` | `string` | 是 | 精确 dtype，如 `f16`/`f32` |

#### 2.4.3 `seed` 字段（`entry_role = "seed"`）

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `pto.oplib.seed_id` | `string` | 是 | 种子唯一标识 |
| `pto.oplib.seed_dtype` | `string` | 是 | seed 函数体基准 dtype |
| `pto.oplib.seed.support_dtypes` | `array<string>` | 是 | 允许实例化的 dtype 列表 |
| `pto.oplib.seed.support_ops` | `array<string>` | 是 | 允许实例化的 op 列表（可含 4 个二元逐元素 op） |
| `pto.oplib.seed.core_slot` | `string` | 否 | 默认 `binary_ewise_core` |

约束：

1. `variant_id` 在同一 `op+kind` 下必须唯一。
2. `seed_id` 在同一 `kind` 下必须唯一。
3. `rows/cols/fractal` 仅允许正整数或 `-1`。
4. `cost`、`priority` 必须为整数。
5. `seed.support_ops` 仅允许 `tmul/tdiv/tadd/tsub`（本规范首批范围）。

### 2.5 Seed 的 core 指令槽位约定

为支持“同一计算骨架覆盖多个 op”，`seed` 函数体定义一个**唯一 core 槽位**：

1. seed 体内必须存在且仅存在一个二元算术 core op。
2. core op 需带属性：`pto.oplib.core_slot = "binary_ewise_core"`（或与 `seed.core_slot` 一致）。
3. 实例化时按目标 `op` 改写该 core op：
   - `tadd -> arith.addf`
   - `tsub -> arith.subf`
   - `tmul -> arith.mulf`
   - `tdiv -> arith.divf`

说明：

1. 其余循环/访存/predicate 骨架保持不变。
2. 这就是“除核心计算指令外，其他模式相同”的共享点。

### 2.6 命名建议（非强制）

建议：

```text
variant: @__pto_oplib_l3_<op>_<variant_id>
seed:    @__pto_oplib_l3_seed_<seed_id>
```

例如：

```text
@__pto_oplib_l3_seed_vec_bin_2d
@__pto_oplib_l3_tmul_f32_r32c32_fast
```

## 3. 多版本匹配、实例化与选择算法

### 3.1 总体规则

对于单个目标 OP 实例，算法分三步：

1. 收集 `variant` 候选（直接匹配）。
2. 收集 `seed` 候选并按 `op + dtype` 自动实例化为临时 `variant`。
3. 在“直接候选 + seed实例化候选”合并集合上做确定性选择。

### 3.2 匹配规则

目标元信息主键为 `op + dtype + shape + layout`：

1. `op`：`tmul/tdiv/tadd/tsub`。
2. `dtype`：`variant.match.dtype` 精确匹配；`seed.support_dtypes` 集合匹配。
3. `shape/layout`：`rows/cols/fractal` 精确或 `-1`，`blayout/slayout` 精确或 `any`。

### 3.3 Seed 实例化规则

当 `seed` 命中匹配条件时：

1. 先验证 `target.op in seed.support_ops`。
2. 再验证 `target.dtype in seed.support_dtypes`。
3. 克隆 seed 函数体并做两类改写：
   - dtype 改写：`seed_dtype -> target.dtype`（签名与体内一致替换）。
   - op 改写：`core_slot` 所在算术 core 改写为目标 op 对应指令。
4. 生成临时 `variant_id`：
   - `__seed__<seed_id>__<op>__<dtype>`
5. 临时实例继承 seed 的 `match.* / cost / priority / sync`。

### 3.4 确定性选择规则

在合并候选集合中，按以下顺序选择：

1. `cost` 最小优先。
2. `cost` 并列时 `priority` 最大优先。
3. 仍并列时 `variant_id` 字典序升序优先。

### 3.5 伪代码

```text
selectVariant(target, entries):
  variants = []

  direct = filter(entries, kind == l3_binary_elementwise_template
                          && entry_role == variant)
  direct = filter(direct, op == target.op)
  direct = filter(direct, dtype == target.dtype)
  direct = filter(direct, shapeLayoutMatch(target))
  variants += direct

  seeds = filter(entries, kind == l3_binary_elementwise_template
                         && entry_role == seed)
  seeds = filter(seeds, shapeLayoutMatch(target))
  for s in seeds:
    if target.op not in s.support_ops:
      continue
    if target.dtype not in s.support_dtypes:
      continue
    inst = instantiateSeed(s, target.op, target.dtype)
    variants += inst

  if variants is empty:
    fail hard with diagnostic

  sort variants by:
    cost ascending,
    priority descending,
    variant_id lexicographically ascending

  return variants[0]
```

### 3.6 选择策略边界

1. 第一版不提供用户手工 override。
2. 不使用“精确优先”的隐式偏置规则，统一由 `cost/priority` 决定。
3. 无匹配必须硬失败，不允许 silent fallback。
4. 显式 `variant` 与 seed 实例化结果同池竞争，不设隐式优先级。

## 4. 同步语义与融合约束

### 4.1 模板内同步支持

允许模板函数体包含显式同步语义（如 event/flag/barrier 等低层同步操作）。

### 4.2 `sync=true` 版本的融合策略

当选中版本具有 `pto.oplib.sync = true` 时，默认进入保守策略：

1. 禁止跨模板的 loop fusion。
2. 禁止跨模板重排（reorder/hoist/sink）。
3. 仅允许在单模板内部做局部规范化且不改变同步顺序。

## 5. OP 实现规范与建议

### 5.1 模板体风格

推荐结构化 MLIR 风格，以 `scf` 组织循环骨架，便于映射到 CCEC 风格代码（如 `vlds/vsts/predicate` 结构）。

建议骨架（示意）：

```text
for i in validRows:
  for j in repeatTimes:
    load src0/src1
    build predicate
    compute op core
    store dst
```

### 5.2 允许与禁止（最小约束）

允许（建议）：

1. 结构化循环与基础算术控制流。
2. 明确的数据加载、计算、存储阶段。
3. 与目标 OP 对应的核心算术语义。

禁止（建议）：

1. 与 OP 语义无关的跨阶段副作用操作。
2. 不可判定的隐式外部依赖。
3. 破坏 `src0/src1/dst` 角色契约的参数重解释。

### 5.3 二元逐元素核心语义要求

1. `tadd`：核心为逐元素加法。
2. `tsub`：核心为逐元素减法。
3. `tmul`：核心为逐元素乘法。
4. `tdiv`：核心为逐元素除法。

说明：

1. 可包含向量化/分块/predicate 处理。
2. 不得改变观测语义（除目标架构允许的浮点细节差异）。

### 5.4 开发建议

1. 优先提交一个高复用 `seed`（覆盖 `tadd/tsub/tmul/tdiv` 与常见 dtype）。
2. 再增补高优性能特化 `variant`，通过 `cost/priority` 覆盖 seed 默认选择。
3. 对含同步版本显式打 `sync=true`，避免误融合。

## 6. 错误模型与诊断建议

推荐诊断类别：

1. `E_OPLIB_NO_MATCH`：无匹配版本（硬失败）。
2. `E_OPLIB_MISSING_ATTR`：缺失必需属性。
3. `E_OPLIB_INVALID_ATTR`：属性值非法（如 `rows=0`）。
4. `E_OPLIB_DUP_VARIANT_ID`：同 `op+kind` 下 `variant_id` 冲突。
5. `E_OPLIB_DUP_SEED_ID`：同 `kind` 下 `seed_id` 冲突。
6. `E_OPLIB_DTYPE_MISMATCH`：属性 dtype 与签名 dtype 不一致。
7. `E_OPLIB_SIGNATURE_INVALID`：参数个数/顺序不符合二元模板契约。
8. `E_OPLIB_SEED_UNSUPPORTED_OP`：目标 op 不在 `seed.support_ops`。
9. `E_OPLIB_SEED_UNSUPPORTED_DTYPE`：目标 dtype 不在 `seed.support_dtypes`。
10. `E_OPLIB_SEED_CORE_SLOT_INVALID`：core 槽位缺失或不唯一。
11. `E_OPLIB_SEED_INSTANTIATE_FAIL`：seed 改写/克隆失败。

诊断信息建议至少包含：

1. `op`
2. `dtype`
3. `variant_id` 或 `seed_id`
4. 触发的 tile 元信息（rows/cols/layout/fractal）
5. 失败原因与修复建议

## 7. 最小示例

### 7.1 显式 `variant` 示例（`tadd`）

```mlir
module {
  func.func @__pto_oplib_l3_tadd_f32_generic(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tadd",
        pto.oplib.variant_id = "f32_generic",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "any",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 100 : i64,
        pto.oplib.priority = 0 : i64
      } {
    return
  }
}
```

### 7.2 `seed` 示例（单 seed 覆盖 4 个 op + 2 个 dtype）

```mlir
module {
  func.func @__pto_oplib_l3_seed_vec_bin_2d(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "seed",
        pto.oplib.seed_id = "vec_bin_2d",
        pto.oplib.seed_dtype = "f32",
        pto.oplib.seed.support_dtypes = ["f16", "f32"],
        pto.oplib.seed.support_ops = ["tadd", "tsub", "tmul", "tdiv"],
        pto.oplib.seed.core_slot = "binary_ewise_core",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "any",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 90 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // 示例：当前写成 addf，仅作为 seed 基准 core；
    // 实例化到 tmul/tdiv/tsub 时会按 core_slot 改写。
    %a = "arith.constant"() <{value = 0.0 : f32}> : () -> f32
    %b = "arith.constant"() <{value = 0.0 : f32}> : () -> f32
    %v = arith.addf %a, %b {pto.oplib.core_slot = "binary_ewise_core"} : f32
    %u = arith.addf %v, %a : f32
    %w = arith.addf %u, %b : f32
    return
  }
}
```

### 7.3 多版本选择示例（`variant` 与 `seed` 同池竞争）

```mlir
module {
  // seed：通配保底（支持 tmul/f16,f32）
  func.func @__pto_oplib_l3_seed_vec_bin(...) attributes {
    pto.oplib.kind = "l3_binary_elementwise_template",
    pto.oplib.entry_role = "seed",
    pto.oplib.seed_id = "vec_bin",
    pto.oplib.seed_dtype = "f32",
    pto.oplib.seed.support_dtypes = ["f16", "f32"],
    pto.oplib.seed.support_ops = ["tmul", "tadd", "tsub", "tdiv"],
    pto.oplib.match.rows = -1 : i64,
    pto.oplib.match.cols = -1 : i64,
    pto.oplib.match.blayout = "any",
    pto.oplib.match.slayout = "any",
    pto.oplib.match.fractal = -1 : i64,
    pto.oplib.cost = 90 : i64,
    pto.oplib.priority = 0 : i64
  } { return }

  // 显式特化 variant：32x32 + f32 + row_major + none_box
  func.func @__pto_oplib_l3_tmul_f32_r32c32_fast(...) attributes {
    pto.oplib.kind = "l3_binary_elementwise_template",
    pto.oplib.entry_role = "variant",
    pto.oplib.op = "tmul",
    pto.oplib.variant_id = "f32_r32c32_fast",
    pto.oplib.match.dtype = "f32",
    pto.oplib.match.rows = 32 : i64,
    pto.oplib.match.cols = 32 : i64,
    pto.oplib.match.blayout = "row_major",
    pto.oplib.match.slayout = "none_box",
    pto.oplib.match.fractal = 512 : i64,
    pto.oplib.cost = 40 : i64,
    pto.oplib.priority = 10 : i64
  } { return }
}
```

若目标为 `tmul + f32 + 32x32 + row_major + none_box + fractal=512`，则 seed 可实例化出 `tmul/f32` 候选，但最终仍会选显式特化 variant（更低 `cost`）。

## 8. 场景验收（规范层）

1. 同 OP 两个 `variant` 都匹配时，按 `cost` 选择。
2. `cost` 相同按 `priority` 选择。
3. `shape` 精确与通配共存时，不做“精确优先”隐式偏置，仍按 `cost/priority`。
4. `dtype` 不匹配导致无候选，触发硬失败。
5. `sync=true` 版本被选中时，必须禁用跨模板 loop fusion/重排。
6. 缺失关键属性（如 `variant_id/seed_id`）时，模板判定为无效并给出诊断。
7. seed 命中但 `target.op` 不在 `support_ops`，必须拒绝实例化并继续尝试其他候选。
8. seed 命中但 `target.dtype` 不在 `support_dtypes`，必须拒绝实例化并继续尝试其他候选。
9. seed 缺失唯一 core_slot 时，必须给出 `E_OPLIB_SEED_CORE_SLOT_INVALID`。

## 9. 默认假设与约束（本轮锁定）

1. 本规范仅定义接口，不含实现与回归测试落地。
2. 与现有 V1 机制关系为“仅文档先行”。
3. 首批仅二元逐元素 OP：`tmul/tdiv/tadd/tsub`。
4. 匹配主键固定为 `op + dtype + shape + layout`，shape 仅支持“精确+通配（-1）”。
5. 第一版仅自动选择，不提供手工 override。
