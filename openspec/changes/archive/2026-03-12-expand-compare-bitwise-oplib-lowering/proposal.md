# Proposal: 扩展 4.8-4.9 Compare 与 Bitwise 的 OP-Lib Lowering

## 概述

本变更覆盖 `docs/PTO_IR_manual.md` 第 4.8 节和第 4.9 节，共 15 个 op：

1. Compare / Select：`tcmp`、`tcmps`、`tsel`、`tsels`
2. Bitwise tile-tile：`tand`、`tor`、`txor`、`tshl`、`tshr`
3. Bitwise tile-scalar：`tands`、`tors`、`txors`、`tshls`、`tshrs`
4. Unary bitwise：`tnot`

本 change 依赖 `generalize-oplib-template-capabilities` 先扩好 integer vector、vector compare/select 和 family-specific attr 匹配。

## 背景与动机

4.8-4.9 相比 4.4-4.6 有额外复杂性：

1. `tcmp` / `tcmps` 带 6 种 `cmpMode`
2. `tsel` 需要 mask tile 参与
3. `tsels` 需要 scalar select mode
4. bitwise / shift family 主要使用整数 dtype，而当前 A5 OP-Lib vector lowering 仍是 float-only

如果不单独拆分，这类差异会把 4.4 的 float family 设计污染得过于复杂。

## 目标

1. 让 4.8-4.9 的 15 个 op 全部进入 OP-Lib lowering。
2. 用 family 方式吸收 mask、cmp mode 和 integer bitwise 语义。
3. 为 compare / select / integer bitwise 建立完整 IR 测试与代表性终态检查。

## 非目标

1. 不覆盖 4.4 的 float arithmetic。
2. 不覆盖 4.5-4.6 的 reduction / broadcast。
3. 不在首版 compare / bitwise family 中引入 seed core-slot 改写。

## 成功标准

1. `tcmp`、`tcmps` 覆盖 6 个 `cmpMode`。
2. `tsel`、`tsels` 分别覆盖 mask 路径和 scalar-mode 路径。
3. integer bitwise/shift family 覆盖静态 valid shape、动态 valid shape 和至少一个较小整数宽度 smoke。
