# Proposal: 扩展 4.4 Vector Arithmetic 的 OP-Lib Lowering 覆盖范围

## 概述

本变更现在只覆盖 `docs/PTO_IR_manual.md` 第 4.4 节 `Vector Arithmetic Operations`。
实现与测试范围必须严格收敛在 4.4，不能再把 4.5-4.8 的 reduction、broadcast、compare、bitwise 需求带回本 change。

原先把 4.4-4.8 一次性塞进单个 change 的做法已经取消，新的拆分是：

1. `generalize-oplib-template-capabilities` 负责多 family OP-Lib 基础设施
2. `expand-vector-op-oplib-lowering` 只负责 4.4
3. `expand-reduction-broadcast-oplib-lowering` 负责 4.5-4.6
4. `expand-compare-bitwise-oplib-lowering` 负责 4.7-4.8

因此，本 change 以前置 change `generalize-oplib-template-capabilities` 已完成并可用为硬前提，再在其提供的多 family OP-Lib 基础设施之上补齐 4.4 的 31 个 op。

## 背景与动机

当前仓库的 Level-3 OP-Lib 路径只稳定覆盖少量 binary tile-tile float op，现状明显落后于第 4.4 节定义的完整向量算术范围：

- Binary tile-tile：`tadd`、`tsub`、`tmul`、`tdiv`、`tmax`、`tmin`、`trem`
- Partial binary：`tpartadd`、`tpartmax`、`tpartmin`
- Special binary：`tprelu`
- Tile-scalar：`tadds`、`tsubs`、`tmuls`、`tdivs`、`tmaxs`、`tmins`、`trems`
- Ternary：`taddc`、`tsubc`、`taddsc`、`tsubsc`
- Unary：`tabs`、`tneg`、`texp`、`tlog`、`tsqrt`、`trsqrt`、`trecip`、`trelu`、`tlrelu`

这会带来三类问题：

1. 4.4 中语义相近的 op 分散在 OP-Lib 和非 OP-Lib 两条 lowering 路径，维护成本高。
2. 现有 seed-template 工作流只覆盖一小撮 op，新增 4.4 op 时容易退化成一次性特判。
3. 当前测试没有系统保护 4.4 全量 op 在静态 shape、动态 valid shape 下的 OP-Lib 命中与实例化行为。

## 目标

1. 让第 4.4 节列出的 31 个 op 全部支持 OP-Lib lowering。
2. 按共享计算骨架组织 family，最大化模板复用，而不是按单个 op 平铺模板。
3. 为 4.4 建立独立测试套，覆盖 IR 命中、实例化、call rewrite 和代表性 EmitC 终态检查。

## 非目标

1. 本 change 不覆盖第 4.5-4.8 节的 reduction、broadcast、compare、bitwise op。
2. 本 change 不负责设计多 family OP-Lib 基础设施本身；那部分由 `generalize-oplib-template-capabilities` 承担。
3. 本 change 不重复交付 `generalize-oplib-template-capabilities` 已落地的 family-aware 注册、签名校验、`argN.*` 元数据、模板白名单和 EmitC 基础能力。
4. 本 change 不引入基于特殊 shape、特殊 dtype、特殊硬件 corner case 的模板特化。
5. 本 change 不修改外部 OP-Lib ABI 的基本原则：模板、实例和调用点仍以 `!pto.tile_buf` 为主。

## 预期结果

完成后应达到以下状态：

1. 第 4.4 节的全部 op 能在 `PTOInstantiateAndLowerToLibCallPass` 中匹配到对应 family。
2. float arithmetic、partial、tile-scalar、ternary、unary、special-form 等 4.4 family 的边界清晰且互不混淆。
3. 现有 binary OP-Lib 路径不回归，新增 4.4 family 受到独立 lit 测试保护。

## 成功标准

1. 第 4.4 节每个 op 至少具备 1 个静态 valid shape 用例和 1 个动态 valid shape 用例。
2. `tdivs` 必须覆盖 `tile/scalar` 与 `scalar/tile` 两种形式。
3. partial family 必须覆盖 mixed valid-region 场景。
4. 代表性终态检查至少覆盖 `trem`、`tprelu`、`tlrelu`、`texp`、`tsqrt`。
