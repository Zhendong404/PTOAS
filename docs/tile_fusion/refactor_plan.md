### PTOAS OP-Lib Pipeline 全量重构（按你指定顺序）

**Summary**
- 目标是把主链路重构为你指定的核心顺序：`InsertSync -> Memref2Tilebuf -> createFusionGroups -> OutlineFusionGroups -> InstantiateAndLowerToLibCall -> InlineLibCall -> Tilebuf2Memref`。
- 采用“顺序优先 + 完全替换”策略：删除旧组合语义。OP-Lib 实例化/降级/内联主链路默认执行，`CreateFusionGroups/Outline/LowLevelLoopFusion` 由 `--enable-op-fusion` 控制。
- 保持 `level3` 的 InsertSync 现状不变（仍按当前规则忽略），`--op-lib-dir` 继续强制必传，单 OP 仍走 Lower+Inline（无匹配时 warning 回退）。

**Public API / Pipeline 变更**
- 新增并对外暴露两个 pass：
- `PTOInstantiateAndLowerToLibCallPass`（替代旧 `PTOLowerToOpLibCallsPass` 在流程中的职责）
- `PTOInlineLibCallPass`（替代旧 `PTOInstantiateAndInlineOpLibPass` 的 inline 职责）
- 完全移除旧 pass 对外接口与实现入口：
- `createPTOLowerToOpLibCallsPass(...)`
- `createPTOInstantiateAndInlineOpLibPass(...)`
- `ptoas` pipeline 中 OP-Lib 实例化/降级/内联主链路默认执行。
- `--enable-op-fusion` 控制 `PTOCreateFusionGroupsPass`、`PTOOutlineFusionGroupsPass`、`PTOLowLevelLoopFusionPass`。
- `--op-lib-dir` 保持 mandatory，缺失直接报错（全量链路下不允许 silent fallback）。
- 保留 `--op-fusion-debug` 与 IR dump 开关，但其触发点会按新顺序重定位。

**Implementation Changes**
- `PTOOutlineFusionGroupsPass` 语义重写为“处理带 group 属性的 PTO 二元算子链”，不再要求输入是 `func.call` 链。
- 该 pass 在 caller 中把组内 PTO op 链 outline 成 `@__pto_fused_group_*`，保证顺序与 side effect 保持。
- 该 pass 输出的 fused helper 继续使用 tilebuf 接口，给后续 InstantiateAndLower 阶段消费。
- `PTOInstantiateAndLowerToLibCallPass` 重构为“实例化 + 改写到 LibCall（不 inline）”：
- 复用现有 OP-Lib 目录扫描、模板匹配、variant/seed 选择策略。
- 覆盖对象为“outlined helper 内的组内 op + 非组单 OP”。
- 将支持的 PTO op 改写为 `func.call @__pto_oplib_inst_*`。
- 实例函数类型统一生成为 memref 形态；调用点按需插入桥接 cast 以匹配签名。
- 组内失配策略保持可观测 fallback（warning + 保留原语义），单 OP 失配同样 warning 回退。
- `PTOInlineLibCallPass` 只做两件事：实例体 materialize + call inline。
- 对外部 instance 函数生成 low-level loop body（沿用现有 fake body 机制与 op->arith 映射）。
- 在普通函数与 fused helper 内联所有 instance call，并清理死 cast/死调用。
- `PTOTileBufToMemrefPass` 移到 Inline 之后执行，作为该段尾部转换。
- 在 Inline 与 Tilebuf2Memref 之间固定插入 `Canonicalizer + CSE`，然后执行 `PTOLowLevelLoopFusionPass`（由 `--enable-op-fusion` 控制），再做一轮 `Canonicalizer + CSE`。
- `ptoas` 入口逻辑同步更新：
- `enable-op-fusion` 分支编排调整为仅控制 `Create/Outline/LowLevelLoopFusion` 三个 pass。
- 重排 dump 时机：`dump-ir-after-oplib-lowering` 在 InstantiateAndLower 后截断；`dump-ir-after-op-fusion` 在 Inline+LoopFusion+Tilebuf2Memref 后截断。
- Pass 注册与构建系统同步：
- 更新 Passes.td/Passes.h 的 pass 定义与 constructor。
- 替换 transforms 库中的旧源码编译单元为新 pass 文件。
- 文档同步：
- 更新项目内 pipeline 描述与 tile_fusion 文档，确保与新顺序、开关语义、默认行为一致。

**Test Plan**
- 更新并扩充 `tile_fusion` lit 用例，覆盖 `--enable-op-fusion` 开关开启/关闭两条链路。
- 新增顺序验证用例：检查 outline 发生在 libcall 改写之前（通过 IR dump/日志锚点验证）。
- 新增行为验证用例：组内与单 OP 都会触发 InstantiateAndLower + Inline。
- 新增回退用例：单 OP 无匹配 warning 回退；组内失配整组回退且语义不变。
- 保留并更新错误用例：未传 `--op-lib-dir` 必须失败。
- 回归 `level3` 行为：InsertSync 仍按既有规则处理，不因本次重构改变。
- 端到端回归：`dump-ir-after-oplib-lowering`、`dump-ir-after-op-fusion`、最终 EmitC/C++ 产物可生成并通过现有关键检查。

**Assumptions / Defaults（已按你确认锁定）**
- 严格按你给的顺序执行该段 pipeline（顺序优先）。
- 完全替换旧 pass 形态，不保留旧 pass API 兼容层。
- OP-Lib 实例化/降级/内联主链路默认启用。
- `CreateFusionGroups/Outline/LowLevelLoopFusion` 由 `--enable-op-fusion` 控制。
- `--op-lib-dir` 必传，缺失即失败。
- 非组单 OP 也执行 Lower + Inline。
- `level3` 的 InsertSync 行为保持现状。
