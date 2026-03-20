# Tile Fusion Scheduling Specification

## ADDED Requirements

### Requirement: OpSchedulingPass MUST compact group members into contiguous block-local spans

`OpSchedulingPass` MUST 在 basic block 内将同一融合组的成员压缩成连续运行片段。

#### Scenario: Group members become contiguous in one block

- **WHEN** 一组目标 op 已拥有相同的 `pto.fusion.group_id`
- **THEN** `OpSchedulingPass` MUST 在所属 basic block 内将它们重排为连续片段
- **AND** 组内物理顺序 MUST 与 `pto.fusion.order` 一致

### Requirement: OpSchedulingPass MUST preserve legality boundaries

调度不得破坏 SSA、side-effect、barrier 或 region / block 合法性。

#### Scenario: Scheduler does not cross hard boundaries

- **WHEN** 某次移动会跨越 barrier、side-effect op、外部 call，或跨出原有 region / block
- **THEN** `OpSchedulingPass` MUST 禁止该移动

#### Scenario: Scheduler does not move an op before its SSA definition

- **WHEN** 某次移动会让 op 出现在其某个 SSA operand 定义之前
- **THEN** `OpSchedulingPass` MUST 禁止该移动

### Requirement: Unrelated treshape MUST NOT act as a global scheduling barrier

`treshape` 不属于 fusion group，但对与其无依赖关系的 group 成员，不得被当作全局调度屏障。

#### Scenario: Group can move across an unrelated treshape

- **WHEN** 一个 group 成员与给定 `pto.treshape` 没有数据依赖关系，且移动不会违反其它合法性约束
- **THEN** `OpSchedulingPass` MAY 让该 group 成员跨过该 `pto.treshape` 进行聚拢

#### Scenario: treshape still blocks moves that would violate SSA legality

- **WHEN** 某次跨 `pto.treshape` 的移动会破坏 SSA 定义顺序或其它合法性边界
- **THEN** `OpSchedulingPass` MUST 禁止该移动
