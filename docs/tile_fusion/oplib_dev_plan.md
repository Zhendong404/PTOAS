## OP-Lib 开发计划（已实现后的迁移说明）

### Summary

本文档原用于 Stage-1 设计评审。当前主线已完成重构，以下内容以现实现为准：

1. OP-Lib 主链路默认总是执行，不再受 `--enable-op-fusion` 控制。
2. `--op-lib-dir` 必填，缺失直接失败。
3. 外部 OP-Lib ABI 强制 `tile_buf`，不接受 `memref` 模板签名。
4. Lower 与 Inline 已拆分为两个 pass：
   - `PTOInstantiateAndLowerToLibCallPass`
   - `PTOInlineLibCallPass`

### 1. 当前 Pipeline（实现态）

1. `LoweringSyncToPipe -> PTOViewToMemref`
2. `InferPTOLayout/PlanMemory/InsertSync`（按现有 level 规则）
3. `PTOMemrefToTileBuf`
4. `PTOCreateFusionGroupsPass`
5. `PTOOutlineFusionGroupsPass`
6. `PTOInstantiateAndLowerToLibCallPass`
7. `PTOInlineLibCallPass`
8. `PTOTileBufToMemref`
9. `Canonicalizer/CSE -> PTOLowLevelLoopFusionPass -> Canonicalizer/CSE`

### 2. 已移除或变更的旧语义

1. `--disable-oplib-lowering`：已移除（传入会报 unknown argument）。
2. `--enable-op-fusion`：保留为兼容 no-op，不再作为 gating。
3. 旧 pass 名称：
   - `PTOLowerToOpLibCallsPass`（对外 API 已替换）
   - `PTOInstantiateAndInlineOpLibPass`（对外 API 已替换）

### 3. OP-Lib 接口约束（强制）

1. 模板函数签名：`(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`。
2. 实例函数签名保持 `tile_buf`。
3. 调用点参数保持 `tile_buf`。
4. Inline 阶段仅允许在函数体内部局部桥接 `tile_buf <-> memref`。

### 4. 失败与回退策略

1. 单 OP 无匹配：warning，保留原 OP。
2. 组内任一 OP 失配：warning，整组回退。
3. 模板签名非法（memref ABI 等）：报错终止。

### 5. 参考

1. 主方案：`docs/tile_fusion/tile_fusion_plan.md`
2. OP-Lib 规范：`docs/tile_fusion/oplib_ir_spec.md`
3. Pipeline 实现入口：`tools/ptoas/ptoas.cpp`
