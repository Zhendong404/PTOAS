## 1. 范围与前置

- [x] 1.1 在 proposal/design 中将本 change 收敛为 `PTO_IR_manual.md` 第 4.4 节，并显式声明依赖 `generalize-oplib-template-capabilities`

## 2. 4.4 Family 模板实现

- [x] 2.1 在 `oplib/level3/` 下补齐 `l3_float_binary_elementwise_template`
- [x] 2.2 在 `oplib/level3/` 下补齐 `l3_float_partial_binary_template`
- [x] 2.3 在 `oplib/level3/` 下补齐 `l3_float_binary_special_template`
- [x] 2.4 在 `oplib/level3/` 下补齐 `l3_float_tile_scalar_template`
- [x] 2.5 在 `oplib/level3/` 下补齐 `l3_float_ternary_tile_template`
- [x] 2.6 在 `oplib/level3/` 下补齐 `l3_float_ternary_tile_scalar_template`
- [x] 2.7 在 `oplib/level3/` 下补齐 `l3_float_unary_template`
- [x] 2.8 在 `oplib/level3/` 下补齐 `l3_float_unary_math_template`
- [x] 2.9 在 `oplib/level3/` 下补齐 `l3_float_unary_scalar_template`

## 3. Lowering 接入

- [x] 3.1 在 `PTOInstantiateAndLowerToLibCallPass` 中接入 4.4 family 的匹配与实例化逻辑
- [x] 3.2 保持 seed 改写只作用于单 core-slot family，不把 `trem`、`tdivs`、`tprelu`、ternary、math unary 纳入 seed

## 4. 测试与验证

- [x] 4.1 为 4.4 的 31 个 op 新增 IR级 lit测试 和 python 测试 （参考test/samples），覆盖静态 valid shape 与动态 valid shape
- [x] 4.2 为 `tdivs` 增加 `tile/scalar` 与 `scalar/tile` 双顺序测试
- [x] 4.3 为 partial family 增加 mixed valid-region 覆盖
- [x] 4.4 增加 4.4 family 的代表性 EmitC 终态测试，覆盖 `trem`、`tprelu`、`tlrelu`、`texp`、`tsqrt`
- [x] 4.5 验证现有 binary OP-Lib regression 不回归
