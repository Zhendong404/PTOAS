# Level-3 OP-Lib IR 接口规范（仅接口文档）

- 状态：Draft v0.1
- 生效范围：`docs` 规范层，不要求当前代码立即实现
- 目标读者：PTOAS OP Fusion 维护者、OP-Lib 开发者

## 1. 设计目标与非目标

### 1.1 目标

本文定义 Level-3 OP-Lib 与 PTOAS 之间的接口契约，满足以下能力：

1. 输入输出以 `tile_buf` 语义建模。
2. 每个 OP 支持多版本并存与注册。
3. 通过模板函数属性提供融合选择信息（静态 cost model）。
4. 版本选择行为确定性、可复现。
5. 对 OP-Lib 开发者友好，尽量模板化与约定化。

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

### 2.2 注册单位

注册单位为 `func.func`，一个版本对应一个函数。

1. 同一 OP 允许多个版本并存。
2. 不依赖函数名编码语义。
3. 必须依赖函数属性携带匹配与选择信息。

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

所有 Level-3 模板函数使用 `pto.oplib.*` 前缀属性。必需与可选字段如下：

| 字段 | 类型 | 必需 | 说明 |
|---|---|---|---|
| `pto.oplib.op` | `string` | 是 | OP 名，取值：`tmul/tdiv/tadd/tsub` |
| `pto.oplib.kind` | `string` | 是 | 固定为 `l3_binary_elementwise_template` |
| `pto.oplib.variant_id` | `string` | 是 | 版本唯一标识 |
| `pto.oplib.match.dtype` | `string` | 是 | 精确 dtype，如 `f16`/`f32` |
| `pto.oplib.match.rows` | `i64` | 是 | 行数精确匹配；`-1` 表示通配 |
| `pto.oplib.match.cols` | `i64` | 是 | 列数精确匹配；`-1` 表示通配 |
| `pto.oplib.match.blayout` | `string` | 是 | `row_major/col_major/any` |
| `pto.oplib.match.slayout` | `string` | 是 | `none_box/row_major/col_major/any` |
| `pto.oplib.match.fractal` | `i64` | 是 | 精确匹配；`-1` 表示通配 |
| `pto.oplib.cost` | `i64` | 是 | 静态代价，数值越小越优 |
| `pto.oplib.priority` | `i64` | 是 | 同 cost 时优先级，数值越大越优 |
| `pto.oplib.sync` | `bool` | 否 | 是否包含显式同步语义，默认 `false` |

约束：

1. `variant_id` 在同一个 `op+kind` 作用域内必须唯一。
2. `dtype` 必须与函数参数中的 `tile_buf.dtype` 一致。
3. `rows/cols/fractal` 仅允许正整数或 `-1`。
4. `cost`、`priority` 必须为整数。

### 2.5 命名建议（非强制）

建议使用：

```text
@__pto_oplib_l3_<op>_<variant_id>
```

例如：

```text
@__pto_oplib_l3_tmul_f32_r32c32_rm_nb_fast
```

## 3. 多版本匹配与选择算法

### 3.1 确定性选择规则

对于单个目标 OP 实例，选择算法按以下顺序执行：

1. 过滤 `pto.oplib.kind == "l3_binary_elementwise_template"`。
2. 过滤 `pto.oplib.op == <target-op>`。
3. 按 `dtype+shape+layout` 规则匹配：
   - `dtype` 必须精确匹配。
   - `rows/cols/fractal`：候选值为精确值或 `-1`。
   - `blayout/slayout`：候选值为精确值或 `any`。
4. 在匹配集合中选 `cost` 最小项。
5. 若 `cost` 并列，选 `priority` 最大项。
6. 若仍并列，按 `variant_id` 字典序升序选第一项。

### 3.2 伪代码

```text
selectVariant(op, tileMeta, candidates):
  pool = filter(candidates, kind == "l3_binary_elementwise_template")
  pool = filter(pool, oplib.op == op)
  pool = filter(pool, dtypeExactMatch(tileMeta.dtype))
  pool = filter(pool, rowsMatch(tileMeta.rows, match.rows))
  pool = filter(pool, colsMatch(tileMeta.cols, match.cols))
  pool = filter(pool, blayoutMatch(tileMeta.blayout, match.blayout))
  pool = filter(pool, slayoutMatch(tileMeta.slayout, match.slayout))
  pool = filter(pool, fractalMatch(tileMeta.fractal, match.fractal))

  if pool is empty:
    fail hard with diagnostic

  sort pool by:
    cost ascending,
    priority descending,
    variant_id lexicographically ascending

  return pool[0]
```

### 3.3 选择策略边界

1. 第一版不提供用户手工 override。
2. 不使用“精确优先”的隐式偏置规则，统一由 `cost/priority` 决定。
3. 匹配失败必须硬失败，不允许 silent fallback。

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

1. 先提交一个“通配保底版本”（`rows/cols/fractal = -1`，layout=`any`）。
2. 再增补高优性能特化版本，使用更低 `cost` 或更高 `priority` 控制选择。
3. 对含同步版本显式打 `sync=true`，避免误融合。

## 6. 错误模型与诊断建议

推荐诊断类别：

1. `E_OPLIB_NO_MATCH`：无匹配版本（硬失败）。
2. `E_OPLIB_MISSING_ATTR`：缺失必需属性。
3. `E_OPLIB_INVALID_ATTR`：属性值非法（如 `rows=0`）。
4. `E_OPLIB_DUP_VARIANT_ID`：同 `op+kind` 下 `variant_id` 冲突。
5. `E_OPLIB_DTYPE_MISMATCH`：属性 dtype 与签名 dtype 不一致。
6. `E_OPLIB_SIGNATURE_INVALID`：参数个数/顺序不符合二元模板契约。

诊断信息建议至少包含：

1. `op`
2. `variant_id`（若有）
3. 触发的 tile 元信息（dtype/rows/cols/layout/fractal）
4. 失败原因与修复建议

## 7. 最小示例

### 7.1 单版本模板示例（`tadd`）

```mlir
module {
  func.func @__pto_oplib_l3_tadd_f32_generic(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst:  !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.op = "tadd",
        pto.oplib.kind = "l3_binary_elementwise_template",
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
    // 伪代码示意：读取 src0/src1，逐元素相加后写回 dst。
    // 实现中推荐使用结构化循环骨架，便于映射 CCEC 代码形态。
    return
  }
}
```

### 7.2 其他首批 OP 最小声明（`tmul/tdiv/tsub`）

```mlir
module {
  func.func @__pto_oplib_l3_tmul_f16_generic(...) attributes {
    pto.oplib.op = "tmul",
    pto.oplib.kind = "l3_binary_elementwise_template",
    pto.oplib.variant_id = "f16_generic",
    pto.oplib.match.dtype = "f16",
    pto.oplib.match.rows = -1 : i64,
    pto.oplib.match.cols = -1 : i64,
    pto.oplib.match.blayout = "any",
    pto.oplib.match.slayout = "any",
    pto.oplib.match.fractal = -1 : i64,
    pto.oplib.cost = 120 : i64,
    pto.oplib.priority = 0 : i64
  } { return }

  func.func @__pto_oplib_l3_tdiv_f32_generic(...) attributes {
    pto.oplib.op = "tdiv",
    pto.oplib.kind = "l3_binary_elementwise_template",
    pto.oplib.variant_id = "f32_generic",
    pto.oplib.match.dtype = "f32",
    pto.oplib.match.rows = -1 : i64,
    pto.oplib.match.cols = -1 : i64,
    pto.oplib.match.blayout = "any",
    pto.oplib.match.slayout = "any",
    pto.oplib.match.fractal = -1 : i64,
    pto.oplib.cost = 130 : i64,
    pto.oplib.priority = 0 : i64
  } { return }

  func.func @__pto_oplib_l3_tsub_f32_generic(...) attributes {
    pto.oplib.op = "tsub",
    pto.oplib.kind = "l3_binary_elementwise_template",
    pto.oplib.variant_id = "f32_generic",
    pto.oplib.match.dtype = "f32",
    pto.oplib.match.rows = -1 : i64,
    pto.oplib.match.cols = -1 : i64,
    pto.oplib.match.blayout = "any",
    pto.oplib.match.slayout = "any",
    pto.oplib.match.fractal = -1 : i64,
    pto.oplib.cost = 110 : i64,
    pto.oplib.priority = 0 : i64
  } { return }
}
```

### 7.3 多版本选择示例（`tmul`）

```mlir
module {
  // 版本 A：通配保底
  func.func @__pto_oplib_l3_tmul_f32_generic(...) attributes {
    pto.oplib.op = "tmul",
    pto.oplib.kind = "l3_binary_elementwise_template",
    pto.oplib.variant_id = "f32_generic",
    pto.oplib.match.dtype = "f32",
    pto.oplib.match.rows = -1 : i64,
    pto.oplib.match.cols = -1 : i64,
    pto.oplib.match.blayout = "any",
    pto.oplib.match.slayout = "any",
    pto.oplib.match.fractal = -1 : i64,
    pto.oplib.cost = 100 : i64,
    pto.oplib.priority = 0 : i64
  } { return }

  // 版本 B：32x32 + row_major + none_box 特化
  func.func @__pto_oplib_l3_tmul_f32_r32c32_fast(...) attributes {
    pto.oplib.op = "tmul",
    pto.oplib.kind = "l3_binary_elementwise_template",
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

若目标 tile 元信息为 `f32 + 32x32 + row_major + none_box + fractal=512`，则命中 A/B 两个版本并选择 B（更低 `cost`）。

## 8. 场景验收（规范层）

1. 同 OP 两个版本都匹配时，按 `cost` 选择。
2. `cost` 相同按 `priority` 选择。
3. `shape` 精确与通配共存时，不做“精确优先”隐式偏置，仍按 `cost/priority`。
4. `dtype` 不匹配导致无候选，触发硬失败。
5. `sync=true` 版本被选中时，必须禁用跨模板 loop fusion/重排。
6. 缺失关键属性（如 `variant_id`）时，模板判定为无效并给出诊断。

## 9. 默认假设与约束（本轮锁定）

1. 本规范仅定义接口，不含实现与回归测试落地。
2. 与现有 V1 机制关系为“仅文档先行”。
3. 首批仅二元逐元素 OP：`tmul/tdiv/tadd/tsub`。
4. 匹配主键固定为 `dtype+shape+layout`，shape 仅支持“精确+通配（-1）”。
5. 第一版仅自动选择，不提供手工 override。
