## Context

### 范围

本 design 只覆盖 A5/A5VM tile fusion mainline 中的 post-lowering backend-side loop unroll 插点，不覆盖：

- EmitC backend
- A3 或非 A5 架构
- pre-lowering planner / scheduling 阶段
- flatten 之后的通用 backend loop optimization

讨论对象限定为：

- `pto.fusion_region` 内部的 backend-ready `scf.for + a5vm.*` carrier loop
- 其所处固定阶段：`PTOLowLevelLoopFusion -> Canonicalizer/CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion`

### 当前状态

当前仓库已经明确接通的 A5 fusion backend cleanup 顺序是：

`PTOLowLevelLoopFusion -> Canonicalizer -> CSE -> PTOFusionPredicateElision -> PTOFusionLoadStoreElision -> PTOFlattenFusionRegion -> CSE`

文档和代码都还没有正式接入 `PTOPostFusionLoopUnroll`，但需求已经出现：

- `PTOLowLevelLoopFusion` 会把相邻 stage 聚合成单个 carrier loop；
- `PTOFusionPredicateElision` 和 `PTOFusionLoadStoreElision` 会进一步压缩 loop body 中的噪声；
- 清理后的 body 有时会变成很短的 steady-state kernel loop，性能上存在合理的后融合 unroll 空间。

同时，这个需求也有明显边界：

- planner / pre-lowering 阶段看不到最终 `scf.for + a5vm.*` body，无法稳定评估展开收益；
- flatten 之后再做，则会失去 `pto.fusion_region` frontier 和 fusion-local 边界；
- emission/backend allocator 层面又掌握了本阶段不可见的最终机器细节，因此本阶段不能承担“最终性能裁判”的职责。

### 实现约束

- 不新增用户可见 CLI flag。
- 不改变 `pto.fusion_region` / `pto.yield` 作为 flatten 前正式 frontier 的语义。
- 不重新引入 fusion-local store/load round-trip。
- 不把 `PTOPostFusionLoopUnroll` 的正式输入扩大到 wrapper 外 residual non-fused A5VM op。
- 继续保持 A5 fusion cleanup 阶段的 memref-first 地址契约，直到进入发射边界。

## Goals / Non-Goals

**Goals:**

- 明确 `PTOPostFusionLoopUnroll` 的正式插点和前后 pass 契约。
- 定义一个 `v1`、保守的 backend-side unroll cost model 契约。
- 让静态 shape 与动态 valid shape 都有清晰的规范语义。
- 约束后续实现只能做 fusion-local、pre-flatten 的 loop unroll，不越界承担其它 backend 职责。

**Non-Goals:**

- 不在本 design 中承诺完整实现。
- 不定义依赖最终寄存器分配、机器调度、发射器内部 lowering 细节的精细性能模型。
- 不在本阶段引入更复杂的 strip-mining、software pipelining 或 cross-region loop restructuring。
- 不引入大范围 factor 搜索；`v1` 只讨论小而稳的因子集合。

## Decisions

### 1. `PTOPostFusionLoopUnroll` 放在 post-fusion / pre-flatten 阶段

选择该插点，而不是 planner/pre-lowering 或 flatten 之后，原因如下：

- 在 `PTOFusionLoadStoreElision` 之后，loop body 指令数、predication 形态和链内访存噪声已经接近真实 steady-state。
- 在 `PTOFlattenFusionRegion` 之前，`pto.fusion_region` / `pto.yield` 仍然提供明确 frontier，可以限制 pass 只工作在 fusion-local loop 上。
- 该阶段可以直接观察 `scf.for + a5vm.*` 低层结构，不再需要从 tile-level op 图反推硬件 loop 形态。

备选方案与拒绝理由：

- 提前到 planner/pre-lowering：
  - 可见信息过于抽象，无法稳定评估 tail/predicate 与低层 body 密度。
- 推迟到 flatten 之后：
  - 失去结构化 region 边界，容易与更通用的 backend loop optimization 职责混淆。

### 2. cost model 只做 coarse-grained gatekeeping

`v1` cost model 只负责回答：

- 当前 loop 是否值得展开；
- 若值得展开，允许的保守因子是否为 `x2` 或 `x4`。

它不负责：

- 推导全局最优因子；
- 预测最终寄存器分配结果；
- 代替 emission/backend allocator 做最终机器级收益裁决。

这样设计的原因是：

- 本阶段能看到足够多的 fusion-local 结构信息；
- 但仍看不到最终 spill、指令选择和机器调度等细节；
- 用保守 gate 比用激进“全知模型”更符合当前主线成熟度。

### 3. 决策信号固定为局部、可解释的结构指标

`v1` cost model 的正式输入信号固定为：

- loop body 指令数
- trip count / valid shape 可证性
- live value / loop-carried value 的粗粒度统计
- tail 比例与 predicate 处理复杂度
- 寄存器压力的粗估结果

后续实现必须遵守以下原则：

- 这些信号来自 post-fusion post-cleanup IR，而不是来自 planner 阶段缓存或外部 profile。
- 当某项信号无法稳定计算时，默认转向更保守的决策，而不是乐观猜测。

### 4. 因子集合限制为 `skip / x2 / x4`

`v1` 只允许三档决策：

- 不展开
- 展开 2 倍
- 展开 4 倍

不开放更大因子，原因是：

- tile fusion post-cleanup loop 的 live value / predicate 成本可能增长很快；
- 大因子更容易放大 tail 成本与寄存器压力；
- 当前目标是先建立稳定、可验收的 contract，而不是做激进自动调优。

### 5. 动态 valid shape 允许部分 unroll，但必须保留显式 tail 语义

静态与动态 shape 的契约分别为：

- 静态可整除 trip count：
  - 可以做无额外 tail 的部分 unroll；
- 静态小 trip count：
  - 允许退化为小规模 full unroll；
- 动态 valid shape：
  - 允许做部分 unroll；
  - 但必须显式保留等价 tail 处理，通常是残余 loop 或等价 predicate 路径；
  - 不能依赖“运行时恰好整除”的隐式假设。

这个决策确保：

- 动态 shape 不会被一刀切排除；
- 但 tail 正确性必须成为 formal contract，而不是实现细节。

### 6. 展开后必须继续满足 flatten 与 cleanup 契约

后续实现必须保证：

- 不重新引入中间 `a5vm.vsts -> a5vm.vlds` round-trip；
- 不破坏 `PTOFusionPredicateElision` 已建立的等价复用关系；
- 不绕过 `pto.yield` / region result frontier；
- `PTOFlattenFusionRegion` 仍然可以按既有 contract 直接 splice body。

## Testing Strategy

OpenSpec 需要预留以下验证方向，供后续实现直接落测试：

- 正向用例：
  - 静态可整除 loop 做 `x2` / `x4` unroll；
  - 动态 valid shape 做 partial unroll，并保留显式 tail；
- 负向用例：
  - loop body 太短但 tail 成本过高时保持 no-op；
  - live value / loop-carried value 过多、寄存器压力高时保持 no-op；
  - 复杂 `scf.if` / 多出口控制流不进入当前 pass；
- 集成验证：
  - `--pto-backend=a5vm --a5vm-print-ir` 可以观察 unroll 插点前后 loop 形态；
  - flatten 后不残留 `pto.fusion_region`，且不存在重新引入的 round-trip 访存。

## Risks / Trade-offs

- [Risk] 本阶段看不到最终寄存器分配和机器调度，可能误判展开收益
  - Mitigation：把 cost model 限定为保守 gatekeeping，并限制因子集合为 `skip / x2 / x4`。

- [Risk] 动态 valid shape 下 tail/predicate 成本可能吃掉 steady-state 收益
  - Mitigation：把显式 tail 保持写入正式契约，并要求当成本不可证时保守 no-op。

- [Risk] 展开后可能破坏 `pto.fusion_region` 的 frontier 假设，影响 flatten
  - Mitigation：明确规定该 pass 不得改写 `pto.yield` / region result 的正式边界。

- [Risk] 后续实现把该 pass 扩张成普通 backend loop optimization，导致职责漂移
  - Mitigation：spec 中严格限定其只作用于 `pto.fusion_region` 内 post-cleanup carrier loop。

## Migration Plan

本 change 为纯契约增量，无外部 ABI 迁移，也无数据迁移需求。

推荐后续实现顺序：

1. 先补 pass 声明与 pipeline 接线。
2. 再实现候选 loop 识别与保守 cost model。
3. 最后补动态 shape tail 处理和回归测试。

若后续实现阶段发现 `v1` cost model 仍需更靠近 emission 的二次校正，应作为独立 change 追加，而不是回滚本 change 对插点与职责边界的定义。

## Open Questions

当前无阻塞本 change 的开放问题。

后续实现时如需增加：

- factor `x8` 或更大集合
- target-specific tuning 常量
- emission/backend allocator 反馈回路

应单独发起新 change，而不是直接扩大本次 `v1` 契约范围。
