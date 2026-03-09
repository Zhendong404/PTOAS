# Tile Fusion Memref World 重构方案（V1）

## 1. 背景

现有 A5 OP-Lib 链路包含 `memref -> tile_buf -> memref` 双向桥接：

1. `legacy memref-to-tile_buf bridge pass`
2. `legacy tile_buf-to-memref bridge pass`

该设计在 `InsertSync` 之后引入额外类型转换与 cast 清理，导致：

1. pipeline 复杂度升高
2. 易出现桥接相关回归
3. OP-Lib 路径与主 memref world 语义割裂

本次重构目标是删除桥接 pass，统一 `InsertSync` 后主链路工作在 memref world，同时保持 OP-Lib 外部模板 ABI 不变。

## 2. 目标与约束

### 2.1 目标

1. 删除 `TileBufBridge` 双 pass。
2. `CreateFusionGroups/Outline/Instantiate/Inline/LoopFusion` 全链路适配 memref。
3. 在 `PTOViewToMemref` 前导入 OP-Lib 模板，使其与业务函数一起完成 ViewToMemref。
4. `PTOValidateSimdIR` 前移到 `PTOViewToMemref` 前执行模板预校验。

### 2.2 保持不变

1. `--op-lib-dir` 在 A5 仍为必填。
2. OP-Lib 模板文件签名仍强制：
   `(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`
3. 组内失败/单 OP 失败的 warning + fallback 语义保持不变。

## 3. 新 pipeline（A5）

### 3.1 Stage-1（Pre）

1. 工具层预导入 OP-Lib 模板（从 `--op-lib-dir` 导入到当前 module）
2. `LoweringSyncToPipe`
3. `PTOValidateSimdIR`
4. `PTOViewToMemref`

### 3.2 Stage-2（优化与 OP-Lib 主链路）

1. `InferPTOLayout`（可选）
2. `PlanMemory`（level1/level2）
3. `PTOInsertSync`（可选，level3 忽略）
4. `PTOCreateFusionGroupsPass`（`--enable-op-fusion`）
5. `PTOOutlineFusionGroupsPass`（`--enable-op-fusion`）
6. `PTOInstantiateAndLowerToLibCallPass`
7. `PTOInlineLibCallPass`
8. `Canonicalizer -> CSE`
9. `PTOLowLevelLoopFusionPass`（`--enable-op-fusion`）
10. `Canonicalizer -> CSE`

## 4. 实现要点

### 4.1 OP-Lib 预导入入口

新增公共入口：

`mlir::pto::importPTOOpLibTemplates(ModuleOp module, StringRef opLibDir, bool debug)`

行为：

1. 复用既有 OP-Lib 文件扫描与模板导入逻辑
2. 导入时继续执行模板 ABI/元数据/模板体硬校验
3. 导入符号使用 `__pto_oplib_entry_*`

### 4.2 InstantiateAndLower 的 memref world 匹配

1. 候选类型从 `TileBufType-only` 扩展为 `MemRefType + TileBufType`。
2. `rows/cols/dtype` 从当前 concrete 操作数类型提取（rank-2 f16/f32）。
3. 对 memref 目标值回溯 `bind_tile/pointer_cast/subview/reinterpret_cast/cast`
   获取 tile config；缺失时使用默认 config。
4. `blayout/slayout/fractal` 匹配键来自上述 config。
5. 实例函数参数类型使用 concrete 类型（memref 主路径）。

### 4.3 删除 TileBufBridge 组件

1. 删除 `lib/PTO/Transforms/PTOTileBufBridge.cpp`
2. 从 `PTOTransforms` 构建列表移除
3. 删除 `createlegacy memref-to-tile_buf bridge pass/createlegacy tile_buf-to-memref bridge pass` 声明与调用

## 5. 测试与验收

1. 更新 `tile_fusion` lit 用例，移除对 bridge pass dump 标签依赖。
2. 增加顺序校验：`PTOValidateSimdIR` 必须出现在 `PTOViewToMemref` 之前。
3. 保持负例：memref 签名模板仍报 `invalid OP-Lib signature`。
4. 保持 `oplib_tile_to_memref_lowering`：最终 IR 不残留 `pto.simd.tile_to_memref`。

推荐回归：

`llvm-lit -sv -j 1 test/tile_fusion`

