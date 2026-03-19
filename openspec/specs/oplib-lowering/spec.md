# OpLib Lowering Specification

## Purpose

定义 PTO IR 到 OP-Lib 调用的 lowering 契约，覆盖模板匹配、实例化与强制门禁行为，确保新增算子和既有算子都能以可维护、可验证的方式接入 OP-Lib 流水线。

## Requirements

### Requirement: 基础算术和激活算子的 OpLib 映射支持

PTO OpLib 降低（Lowering）流水线 MUST 支持将 `trem`、`trems`、`tprelu` 和 `tlrelu` 算子映射到其对应的 OpLib 模板。

#### Scenario: 将 PTO 算子降低为 OpLib 调用

- `TRemOp` (`trem`) 应当被降低为使用 `trem` 模板的 OpLib 调用。
- `TRemSOp` (`trems`) 应当被降低为使用 `trems` 模板的 OpLib 调用。
- `TPReluOp` (`tprelu`) 应当被降低为使用 `tprelu` 模板的 OpLib 调用。
- `TLReluOp` (`tlrelu`) 应当被降低为使用 `tlrelu` 模板的 OpLib 调用，并将 `slope` 属性正确传递给模板。
- 所有降低后的调用必须根据原始 PTO 算子的具体 Tile 形状（Shape）和元素类型（Type）进行正确的实例化。
- 降低过程应当在开启 A5 架构支持且使用 OpLib 流水线时自动执行。

### Requirement: Interface-based OpLib lowering

The OpLib lowering mechanism SHALL be refactored to use an interface-driven approach to improve extensibility and maintainability.

#### Scenario: Replace hardcoded matching with interfaces

- A new `OpLibOpInterface` SHALL be defined to provide OpLib template information.
- The `PTOLowerToOpLibCalls` pass SHALL be refactored to use `OpLibOpInterface` instead of hardcoded operator matching.
- Existing PTO operators SHALL implement `OpLibOpInterface` to maintain functionality.

#### Scenario: Interface descriptor fields preserve legacy matching semantics

- `OpLibOpInterface` descriptors SHALL carry enough data to reconstruct `MatchRequest`, including `kind/opName`, operand order and roles, and optional fields `scalarPos/cmpMode/isBinary/requiredVariantId`.
- Ops that require temporary tile operands, for example `tprelu`, `trowsum`, `trowmax`, `trowmin`, and `tcolsum`, SHALL expose `tmp` in descriptor operands with correct role and order.
- Interface-driven matching SHALL preserve legacy constraints for active compare/select lowering (`tcmp`, `tcmps`, `tsel`) and MUST reject or bypass mismatched select-scalar paths such as `tsels` when the PTO IR contract does not match the available templates.
- Interface-driven matching SHALL preserve `tdivs` variant restriction via `requiredVariantId` (`tile_scalar` or `scalar_tile`).
- Interface-driven matching SHALL preserve `tcolsum` branch selection via `isBinary`.
- Interface-driven matching SHALL preserve `tprelu` tmp element type (`i8`) and `trem`/`trems` float-only restrictions.

### Requirement: 4.5~4.9 target ops must not silently fallback

For ops in PTO IR manual sections 4.5~4.9 (current lowering target set), OP-Lib lowering SHALL be mandatory once `PTOInstantiateAndLowerToLibCall` runs.

#### Scenario: Mandatory lowering for target op family

- During OP-Lib lowering, if a target op fails at descriptor-to-`MatchRequest`, candidate selection, instance creation, or call rewrite, the pass SHALL emit error and fail.
- The pass SHALL NOT keep original target ops via warning-based fallback for these families.

#### Scenario: Post-pass no-residual check

- After lowering, non-OPLib user functions SHALL NOT contain remaining target PTO ops.
- If any residual target op exists, the pass SHALL emit error and fail.
