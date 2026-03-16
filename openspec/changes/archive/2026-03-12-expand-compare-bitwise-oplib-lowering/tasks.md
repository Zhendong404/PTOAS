## 1. 范围与前置

- [x] 1.1 在 proposal/design 中固定 4.8-4.9 范围并声明依赖 `generalize-oplib-template-capabilities`

## 2. 4.8-4.9 Family 模板实现(必须采用simd编程范式)
- [x] 2.1 在 `oplib/level3/` 下补齐 `l3_cmp_tile_tile_template`(必须采用simd编程范式)
- [x] 2.2 在 `oplib/level3/` 下补齐 `l3_cmp_tile_scalar_template`(必须采用simd编程范式)
- [x] 2.3 在 `oplib/level3/` 下补齐 `l3_select_mask_template`(必须采用simd编程范式)
- [x] 2.4 在 `oplib/level3/` 下补齐 `l3_select_scalar_template`(必须采用simd编程范式)
- [x] 2.5 在 `oplib/level3/` 下补齐 `l3_int_binary_elementwise_template`(必须采用simd编程范式)
- [x] 2.6 在 `oplib/level3/` 下补齐 `l3_int_tile_scalar_elementwise_template`(必须采用simd编程范式)
- [x] 2.7 在 `oplib/level3/` 下补齐 `l3_int_unary_template`(必须采用simd编程范式)

## 3. Lowering 接入

- [x] 3.1 在 matcher / instantiation 中接入 4.8-4.9 family
- [x] 3.2 为 `tcmp` 与 `tcmps` 接入 6 个 `cmpMode` 覆盖

## 4. 测试与验证

- [x] 4.1 为 `tsel` 与 `tsels` 增加独立 IR 测试
- [x] 4.2 为 integer bitwise / shift family 增加静态 valid shape、动态 valid shape 和较小整数宽度 smoke
- [x] 4.3 增加 family 级 EmitC 终态测试，覆盖 `tcmp(LT)`、`tsel`、`tand`、`tshrs`、`tnot`
- [x] 4.4 验证 compare / bitwise 新增 family 与既有 OP-Lib regression 一致通过
