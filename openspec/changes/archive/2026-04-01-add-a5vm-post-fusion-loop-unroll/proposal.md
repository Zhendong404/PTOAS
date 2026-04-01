# Proposal: 增加 A5VM 融合后循环展开契约

## 概述

当前 A5 tile fusion 主线已经具备 `PTOLowLevelLoopFusion -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion` 的 post-lowering cleanup 链路，但还没有对“融合后自动循环展开”建立正式契约。随着低层循环融合和链内访存清理逐步稳定，`pto.fusion_region` 内经常会形成单个较短的 hardware-facing carrier loop；如果继续直接 flatten 并进入发射，loop body 可能过短，无法充分利用硬件取指与乱序窗口。

本 change 计划补齐一个 `v1` 级别的 OpenSpec 契约：把 `PTOPostFusionLoopUnroll` 明确定义为位于 `PTOFusionLoadStoreElision` 与 `PTOFlattenFusionRegion` 之间的保守 backend-side 插点，并规定其 cost model、动态/静态 valid shape 处理和与前后 pass 的边界。

## 背景与动机

当前主线里，真正接近 backend-ready 形态的 carrier loop 只有在 `PTOLowLevelLoopFusion`、`Canonicalizer/CSE`、`PTOFusionPredicateElision` 和 `PTOFusionLoadStoreElision` 之后才会稳定下来。在这个时点之前：

- planner / pre-lowering 阶段看到的仍是 tile-level PTO op，而不是最终 `scf.for + a5vm.*` body；
- 低层循环融合、谓词复用和链内 store/load round-trip 消除都会显著改变 loop body 密度与 live value 形态；
- 如果过早决策 unroll，实际很容易基于不稳定 IR 高估收益或低估 tail/predicate 成本。

相反，如果等到 `PTOFlattenFusionRegion` 之后再做决策：

- 会丢失 `pto.fusion_region` / `pto.yield` 提供的结构化边界；
- residual non-fused A5VM op 与 fusion-local loop 会混在普通父 block 中；
- 该阶段的职责也会开始与更通用的 backend loop optimization 交叠，不利于 tile fusion 主线维持清晰契约。

因此，需要一个明确的 spec-level 结论：`PTOPostFusionLoopUnroll` 的 cost model 只在 post-fusion / pre-flatten 阶段做保守 gatekeeping，而不是承担最终、依赖机器细节的全量性能裁判。

## 目标

- 在 A5 fusion mainline 中新增 `PTOPostFusionLoopUnroll` 的正式 pipeline 插点，固定其位于 `PTOFusionLoadStoreElision` 与 `PTOFlattenFusionRegion` 之间。
- 定义一个 `v1` 级别、保守的 post-lowering unroll cost model 契约，允许它依赖：
  - loop body 指令数
  - trip count / valid shape 可证性
  - live value / loop-carried value 粗估
  - tail 比例与 predicate 复杂度
  - 粗粒度寄存器压力估计
- 明确静态 shape 与动态 valid shape 都可进入该阶段，但动态 shape 下必须保留显式 tail 语义。
- 规定该阶段必须保持 `pto.fusion_region` frontier、不得重新引入 fusion-local round-trip 访存，并保证 `PTOFlattenFusionRegion` 仍可按既有契约工作。
- 通过新增 capability spec 与更新 `a5vm-backend-pipeline`，把“为什么 cost model 放在这个阶段”固化为可归档、可回归的行为契约。

## 非目标

- 不在本 change 中实现 `PTOPostFusionLoopUnroll` 的完整代码接线或 backend emission 级二次校正。
- 不定义依赖最终寄存器分配、指令选择或机器调度结果的精细性能模型。
- 不新增用户可见 CLI flag，不要求用户显式传入 unroll 因子。
- 不引入 strip-mining、software pipelining、跨 `pto.fusion_region` 的 loop reshaping，或更激进的 backend 通用循环优化。
- 不改变 EmitC 路径、A3 路径或 A5 之外架构的行为。

## 预期结果

- OpenSpec 将明确规定：A5 fusion mainline 在 `PTOFusionLoadStoreElision` 之后、`PTOFlattenFusionRegion` 之前预留 `PTOPostFusionLoopUnroll` 阶段。
- `PTOPostFusionLoopUnroll` 将被定义为仅作用于 `pto.fusion_region` 内 backend-ready `scf.for + a5vm.*` carrier loop 的保守变换，不会扩张到 wrapper 外的 residual non-fused A5VM op。
- `v1` cost model 将以 `skip / x2 / x4` 这类小而稳的因子集合作为正式决策域；当收益不可证、tail 代价过高或寄存器压力风险过大时，必须保守 no-op。
- 动态 valid shape 将被正式允许进入该阶段，但 transform 不得依赖“动态 trip count 恰好整除”的隐式假设，必须显式保留等价 tail 处理。
- 后续实现可以直接以该 change 中的 capability 和 pipeline delta 为依据接线，不需要再重新定义插点与职责边界。

## 成功标准

- 新增 OpenSpec change `add-a5vm-post-fusion-loop-unroll`，包含 `proposal.md`、`design.md`、`tasks.md` 和对应 specs delta。
- 新增 capability `tile-fusion-post-lowering-unroll`，明确：
  - 仅在 `pto.fusion_region` 内工作；
  - 正式输入是 post-fusion post-cleanup carrier loop；
  - cost model 只做保守 gatekeeping；
  - 动态 valid shape 必须保留显式 tail 语义；
  - 不得重新引入 fusion-local round-trip memory traffic。
- 修改 `a5vm-backend-pipeline` 的正式顺序，把 cleanup 链从 `... -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion -> CSE` 更新为 `... -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion -> CSE`。
- proposal/design/tasks 中明确记录：`PTOPostFusionLoopUnroll` 当前仍是 planned stage，本 change 只锁定契约，不承诺本次立即落代码实现。

## Capabilities

### New Capabilities

- `tile-fusion-post-lowering-unroll`: 定义 A5 tile fusion 主线中，`PTOFusionLoadStoreElision` 之后、`PTOFlattenFusionRegion` 之前的 fusion-local 循环展开契约，包括 cost model 决策边界、动态/静态 valid shape 语义和与 flatten/frontier 的兼容性。

### Modified Capabilities

- `a5vm-backend-pipeline`: 更新 A5 fusion mainline 的固定 backend cleanup 顺序，把 `PTOPostFusionLoopUnroll` 纳入正式 pipeline 契约，并强调该阶段仍保持 memref-first 地址语义直到进入发射边界。

## Impact

- 受影响 specs：
  - `openspec/specs/a5vm-backend-pipeline/spec.md`
  - 新增 capability `tile-fusion-post-lowering-unroll`
- 预期受影响实现区域：
  - `tools/ptoas/ptoas.cpp`
  - `include/PTO/Transforms/Passes.td`
  - `include/PTO/Transforms/Passes.h`
  - `lib/PTO/Transforms/TileFusion/`
- 预期受影响验证区域：
  - `test/tile_fusion/`
  - `--pto-backend=a5vm --a5vm-print-ir` 相关 pipeline 观察用例
