# Design: 4.8-4.9 Compare 与 Bitwise OP-Lib Lowering

## 范围与依赖

本设计覆盖：

1. 4.8 Compare & Select
2. 4.9 Bitwise Operations

依赖 `generalize-oplib-template-capabilities` 已支持：

1. vector compare/select 模板体与 EmitC 支持
2. integer vector load/store/bitwise/shift
3. `cmp_mode` 等 family attr 匹配
4. `(mask, src0, src1, dst)` 与 `(src0, src1, selectMode, dst)` 等签名

## Family 划分

### Family A: `l3_cmp_tile_tile_template`

覆盖：

- `tcmp`

策略：

1. 使用 `(src0, src1, dst)` 签名。
2. variant 由 `cmp_mode` 区分 `EQ/NE/LT/LE/GT/GE`。
3. 模板内部统一生成 `vector<i1>` 比较结果，再写回 mask tile。

### Family B: `l3_cmp_tile_scalar_template`

覆盖：

- `tcmps`

策略：

1. 使用 `(src, scalar, dst)` 签名。
2. variant 由 `cmp_mode` 区分 6 种模式。

### Family C: `l3_select_mask_template`

覆盖：

- `tsel`

策略：

1. 使用 `(mask, src0, src1, dst)` 签名。
2. variant-only。

### Family D: `l3_select_scalar_template`

覆盖：

- `tsels`

策略：

1. 使用 `(src0, src1, selectMode, dst)` 签名。
2. variant-only。

### Family E: `l3_int_binary_elementwise_template`

覆盖：

- `tand`
- `tor`
- `txor`
- `tshl`
- `tshr`

策略：

1. 使用 `(src0, src1, dst)` 签名。
2. variant-only。
3. 不引入 seed，避免过早把 shift 语义绑定到固定 core-slot 模型。

### Family F: `l3_int_tile_scalar_elementwise_template`

覆盖：

- `tands`
- `tors`
- `txors`
- `tshls`
- `tshrs`

策略：

1. 使用 `(src, scalar, dst)` 签名。
2. variant-only。

### Family G: `l3_int_unary_template`

覆盖：

- `tnot`

策略：

1. 使用 `(src, dst)` 签名。
2. variant-only。

## 模板语义

compare family 的外部 ABI 继续保持 PTO tile 语义：

1. 输入输出仍是 `!pto.tile_buf`
2. 内部比较可用 `vector<i1>`
3. 最终按现有 PTO 语义 materialize 到 mask tile

bitwise family 主要针对整数 dtype，首版默认以 `i32` 为主测试，额外补一个较小宽度 smoke。

## 测试

必须覆盖：

1. `tcmp/tcmps` 的 6 个 `cmpMode`
2. `tsel` 与 `tsels`
3. bitwise / shift family 的静态 valid shape 与动态 valid shape
4. 至少 1 个较小整数宽度 smoke
5. family 级 EmitC 终态检查：`tcmp(LT)`、`tsel`、`tand`、`tshrs`、`tnot`
