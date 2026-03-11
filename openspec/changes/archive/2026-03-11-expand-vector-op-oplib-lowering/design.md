# Design: 4.4 Vector Arithmetic OP-Lib Lowering

## 范围与依赖

本设计只覆盖 `docs/PTO_IR_manual.md` 第 4.4 节。
设计、实现和测试项都必须限定在 4.4，不向 4.5-4.8 扩张。

它显式依赖 `generalize-oplib-template-capabilities` 已完成以下基础设施：

1. 多 `pto.oplib.kind` family 模型
2. 可变参数模板签名校验
3. `argN.*` 匹配元数据
4. 模板体 `math.exp/log/sqrt/rsqrt` 白名单
5. A5 OP-Lib vector 对 float unary / math unary 的 EmitC 支持

本 change 把 `generalize-oplib-template-capabilities` 视为已合入主干的硬依赖，不提供任何降级 fallback，也不重复定义这些基础设施；这里只在其之上定义 4.4 family 如何分组、匹配和测试。

## 当前状态

仓库中已有的 4.4 相关基础非常有限：

1. `oplib/level3/binary_templates.mlir` 只提供 `tadd/tsub/tmul/tdiv/tmax/tmin` 的 binary seed 模板。
2. `PTOLowerToOpLibCalls.cpp` 现有逻辑围绕 binary tile-tile 接口构建。
3. 现有 lit 用例主要保护 binary float chain，对 4.4 其余家族没有系统覆盖。

## Family 划分

### Family A: `l3_float_binary_elementwise_template`

覆盖：

- `tadd`
- `tsub`
- `tmul`
- `tdiv`
- `tmax`
- `tmin`

策略：

1. 复用当前 binary seed-template 骨架。
2. 继续使用单一 core-slot 改写。
3. `trem` 不放进该 seed family。

### Family B: `l3_float_partial_binary_template`

覆盖：

- `tpartadd`
- `tpartmax`
- `tpartmin`

策略：

1. 仍使用 `(src0, src1, dst)` ABI。
2. loop 上界取 `src0/src1/dst` 的 valid-region 交集。
3. 允许 seed 覆盖这 3 个 op，但骨架必须独立于普通 binary family。

### Family C: `l3_float_binary_special_template`

覆盖：

- `tprelu`

策略：

1. 使用独立 variant-only family。
2. 模板内部允许比较、乘法和 select 组合。
3. 不复用 binary seed core-slot 模型。

### Family D: `l3_float_tile_scalar_template`

覆盖：

- `tadds`
- `tsubs`
- `tmuls`
- `tmaxs`
- `tmins`

以及额外的 variant：

- `tdivs`
- `trems`

策略：

1. `(tile, scalar, dst)` 是标准模板签名。
2. `tadds/tsubs/tmuls/tmaxs/tmins` 允许用 seed 共享骨架。
3. `tdivs` 拆成两个 variant：`tile/scalar` 与 `scalar/tile`。
4. `trems` 走 variant-only。

### Family E: `l3_float_ternary_tile_template`

覆盖：

- `taddc`
- `tsubc`

策略：

1. 使用 `(src0, src1, src2, dst)` 签名。
2. variant-only。
3. 不要求 core-slot 改写。

### Family F: `l3_float_ternary_tile_scalar_template`

覆盖：

- `taddsc`
- `tsubsc`

策略：

1. 使用 `(src0, scalar, src1, dst)` 签名。
2. variant-only。
3. 以共享 skeleton 组织模板。

### Family G: `l3_float_unary_template`

覆盖：

- `tabs`
- `tneg`
- `trecip`
- `trelu`

策略：

1. 使用 `(src, dst)` 签名。
2. variant-only。
3. 模板体只依赖 float unary 和简单 compare/select 语义。

### Family H: `l3_float_unary_math_template`

覆盖：

- `texp`
- `tlog`
- `tsqrt`
- `trsqrt`

策略：

1. 使用 `(src, dst)` 签名。
2. variant-only。
3. 模板体允许最小子集 `math.exp/log/sqrt/rsqrt`。

### Family I: `l3_float_unary_scalar_template`

覆盖：

- `tlrelu`

策略：

1. 使用 `(src, scalar, dst)` 签名。
2. variant-only。
3. 不与 tile-scalar arithmetic seed family 共享 metadata。

## Matching 与实例化

4.4 change 对 matcher 的要求如下：

1. 先按 op name 归属 family，再读取 family 需要的 `argN.*` 和 attr 匹配元数据。
2. `tdivs` 的操作数顺序必须作为 variant 选择条件，而不是在命中后再静默猜测。
3. partial family 只在 valid-region 语义被 metadata 声明为 partial 时可命中。
4. seed 改写仅用于 `l3_float_binary_elementwise_template` 和 `l3_float_partial_binary_template` 的单 core-slot 模板。

## 动态 shape 约束

本 change 中的“动态 shape”统一指动态 valid shape：

1. `rows/cols` 仍为静态 physical tile shape。
2. `v_row/v_col` 可以是动态值。
3. 不扩展到动态 physical tile bridge，因为当前 OP-Lib tile bridge 仍依赖静态 physical shape。

## 测试设计

测试目录独立为 4.4 专属套件，不继续堆进现有 binary smoke 文件。

必须覆盖：

1. 每个 op 的静态 valid shape 命中
2. 每个 op 的动态 valid shape 命中
3. `tdivs` 的双顺序
4. partial family 的 mixed valid-region
5. family 级 EmitC 终态检查：`trem`、`tprelu`、`tlrelu`、`texp`、`tsqrt`

现有 binary regression 用例继续保留，作为 4.4 扩展后的回归保护。
