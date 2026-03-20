## Context

### 范围

本 design 只覆盖 5.4 `OpSchedulingPass`。

前置条件固定为：

- `FusionPlanPass` 已存在
- 目标 op 已带 `pto.fusion.group_id` 和 `pto.fusion.order`
- 输入仍在 `tile_buf world`

后置条件固定为：

- 同组成员在 basic block 内形成连续片段
- CFG 不变
- 组内物理顺序与 `pto.fusion.order` 一致

### 当前状态

当前仓库还没有与 tile fusion planning 解耦的调度阶段。现有前端分组逻辑把“谁成组”和“天然是否相邻”混在一起，这在 DAG 分组后会立刻失效。

### 约束

- 只在单个 basic block 内重排
- 不跨 region 移动
- 不跨 SSA 定义点
- 不跨 side-effect / barrier / 外部 call
- `treshape` 不属于任何 fusion group

## Goals / Non-Goals

**Goals:**

- 实现独立的 `OpSchedulingPass`
- 消费现有 group metadata 做物理聚拢
- 明确 `treshape` 的调度语义

**Non-Goals:**

- 不改变 group 划分结果
- 不做 CFG 变换
- 不做 5.5+ 后续阶段

## Decisions

### 决策 1：调度采用 block-local 稳定拓扑压缩

`OpSchedulingPass` 只在 basic block 内移动指令，目标是让每个 group 变成一个连续片段。调度顺序固定为：

- 先满足 `pto.fusion.order`
- 再尽量保持原始相对顺序稳定

采用该方案的原因：

- 这能最大限度减少调度阶段重新决策的空间。
- 后续 `fusion_region` 封装只需要扫描连续片段。

### 决策 2：`treshape` 不是调度屏障，允许无关 group 跨过它

`treshape` 在调度层的语义固定为：

- 不属于 group
- 若与某个 group 无依赖关系，group 成员允许跨过它移动
- 若某次移动会因为 `treshape` 所在依赖链破坏 SSA 合法性，则该移动仍然被禁止

采用该方案的原因：

- 这与当前用户对 `treshape` 的边界定义完全一致。
- 也避免把 `treshape` 升级成不必要的全局屏障。

### 决策 3：hard boundary 和 local boundary 在调度层明确区分

调度层区分两类边界：

- `HardBoundary`
  - barrier
  - side-effect op
  - 外部 call
  - region / block 边界
- `LocalBoundary`
  - 当前只包括 `treshape`

规则如下：

- 不可跨 `HardBoundary`
- 可在合法条件下跨 `LocalBoundary`

采用该方案的原因：

- 这让 `OpSchedulingPass` 能直接复用 analysis / planning 阶段对边界的分类结果。

## Risks / Trade-offs

- [Risk] 调度实现若只看 group metadata，不校验 SSA 依赖，可能引入错误重排。
  → Mitigation：移动前必须显式校验 SSA 定义点与 side-effect 边界。

- [Risk] 允许跨 `treshape` 移动会让实现比“把所有非 group op 都当屏障”更复杂。
  → Mitigation：只允许跨无依赖的 `treshape`，并用专门回归锁住语义。

- [Risk] 物理重排会放大 nondeterministic 顺序风险。
  → Mitigation：明确使用稳定顺序策略，不允许依赖迭代容器偶然顺序。

## Migration Plan

1. 声明并实现 `OpSchedulingPass`。
2. 让其直接消费 `FusionPlanPass` 输出的 group metadata。
3. 在 pipeline 中将其放到 `FusionPlanPass` 之后。
4. 用 focused lit 和 `online_update` driver sample 验证连续化与合法性边界。

## Open Questions

- 实现时是以“锚点插入 + 成员搬移”的方式压缩 group，还是先提取成员列表再集中重排，更利于保证稳定顺序？
