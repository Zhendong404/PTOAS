# PTOAS OP Fusion V1 设计与落地方案（已实现）

## 1. 状态与目标

本文档描述 **当前主线已落地实现**，不再是仅设计态草案。

当前 OP Fusion/OP-Lib 链路目标：

1. 在 PTOAS 中自动识别可融合 OP group。
2. 基于 OP-Lib 模板完成实例化与 `pto op -> libcall` 改写。
3. 在 helper 函数内部进行 inline 与低层 loop 级融合。
4. 继续复用 `EmitPTOManual -> emitc -> C++` 主流程。

核心约束：**OP-Lib 对外接口（模板/实例/调用点）必须保持 `!pto.tile_buf`。**

---

## 2. 当前实际 Pipeline

以下以 `tools/ptoas/ptoas.cpp` 为准。

### 2.1 Stage-1（前置 lowering）

1. `LoweringSyncToPipe`（`func::FuncOp`）
2. `PTOViewToMemref`（`ModuleOp`）

### 2.2 Stage-2（优化与 OP-Lib 主链路）

1. `InferPTOLayout`（可选）
2. `PlanMemory`（仅 `level1/level2`）
3. `PTOInsertSync`（可选；`level3` 忽略）
4. `PTOMemrefToTileBuf`
5. `PTOCreateFusionGroupsPass`
6. `PTOOutlineFusionGroupsPass`
7. `PTOInstantiateAndLowerToLibCallPass`
8. `PTOInlineLibCallPass`
9. `PTOTileBufToMemref`
10. `Canonicalizer -> CSE`
11. `PTOLowLevelLoopFusionPass`
12. `Canonicalizer -> CSE`

### 2.3 Stage-3（代码生成）

1. `CSE`
2. `EmitPTOManual`（按 `--pto-arch=a3|a5`）
3. `emitc::FormExpressions`
4. `CSE`
5. C++ 文本后处理

### 2.4 Dump 截断点

1. `--dump-ir-after-oplib-lowering`：在 `PTOInstantiateAndLowerToLibCallPass` 后直接输出并退出。
2. `--dump-ir-after-op-fusion`：在 `Inline + TileBuf2Memref + LoopFusion` 后输出并退出。

---

## 3. Pass 语义（V1）

### 3.1 `PTOCreateFusionGroupsPass`

1. 在同一 basic block 内识别连续可融合链。
2. 给 PTO 二元逐元素 op 打组属性（例如 group/order）。

### 3.2 `PTOOutlineFusionGroupsPass`

1. 输入是“带 group 属性的 PTO op 链”，不是旧的 call 链。
2. 生成 `@__pto_fused_group_*` helper。
3. helper 边界参数保持 `tile_buf` 类型。
4. caller 仅保留 helper 调用边界，不提前扩散到 memref。

### 3.3 `PTOInstantiateAndLowerToLibCallPass`

1. 从 `--op-lib-dir` 读取模板，做候选匹配与选择。
2. 对组内 OP 与单 OP 都执行 lower（单 OP 无匹配则 warning 回退）。
3. 将 PTO op 改写为实例函数 `func.call`。
4. 严格校验模板签名为 `(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`。

### 3.4 `PTOInlineLibCallPass`

1. 物化实例函数体并执行 inline。
2. 清理死 cast / 冗余调用边界。
3. 允许在函数体内部引入局部 `unrealized_conversion_cast(tile_buf <-> memref)`，但函数对外签名不变。

### 3.5 `PTOTileBufToMemref` 与 `PTOLowLevelLoopFusionPass`

1. 在 inline 之后统一收口到 memref。
2. 低层 loop 融合默认执行（除非在更早 dump 截断）。

---

## 4. OP-Lib 接口约束

### 4.1 强制签名

首批二元逐元素模板函数签名：

`(!pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>) -> ()`

### 4.2 强制规则

1. 模板函数参数/结果必须是 `tile_buf`，禁止模板层 `memref` ABI。
2. 实例函数签名必须延续 `tile_buf`。
3. 调用点（caller 中 `func.call`）参数必须是 `tile_buf`。
4. `memref` 只允许在 inline 后桥接，且只作为内部 lowering 手段。

### 4.3 失配与错误

1. 单 OP 无候选：warning，保留原 PTO op。
2. 组内有失配：warning，整组回退。
3. 模板签名非法（例如 memref ABI）：直接报错并终止。

---

## 5. CLI 语义（当前）

1. `--op-lib-dir`：必填，缺失直接报错。
2. `--enable-op-fusion`：兼容开关，当前为 no-op（主链路总是执行）。
3. `--op-fusion-debug`：输出 group/outline/lower/inline/loop-fusion 调试日志。
4. `--disable-oplib-lowering`：已移除，不再支持。

---

## 6. Level 约束

1. `level1/level2`：执行 `PlanMemory`，可选 `InsertSync`。
2. `level3`：跳过 `PlanMemory`，忽略 `--enable-insert-sync`（保持现状），并要求 `alloc_tile` 显式 `addr`。
3. OP-Lib 主链路在三个 level 均执行。

---

## 7. 测试覆盖建议

建议至少覆盖以下类别：

1. pipeline 顺序与 dump 截断点检查。
2. outline helper 生成（`@__pto_fused_group_*`）与调用边界检查。
3. OP-Lib 接口形态检查（模板/实例/调用均为 `tile_buf`）。
4. memref 签名模板负例（必须报错）。
5. 单 OP 路径回归（无 group 也能 lower + inline，失配 warning 回退）。
6. level3 回归（InsertSync 忽略，其余链路正常）。

---

## 8. 相关文档

1. OP-Lib 模板规范：`docs/tile_fusion/oplib_ir_spec.md`
2. Pipeline 实现入口：`tools/ptoas/ptoas.cpp`
