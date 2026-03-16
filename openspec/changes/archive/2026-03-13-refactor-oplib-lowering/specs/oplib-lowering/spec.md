# OpLib Lowering Specification

## ADDED Requirements

### Requirement: Interface-based OpLib lowering
The OpLib lowering mechanism SHALL be refactored to use an interface-driven approach to improve extensibility and maintainability.

#### Scenario: Replace hardcoded matching with interfaces
- A new `OpLibOpInterface` SHALL be defined to provide OpLib template information.
- The `PTOLowerToOpLibCalls` pass SHALL be refactored to use `OpLibOpInterface` instead of hardcoded operator matching.
- Existing PTO operators SHALL implement `OpLibOpInterface` to maintain functionality.

#### Scenario: Interface descriptor fields preserve legacy matching semantics
- `OpLibOpInterface` descriptors SHALL carry enough data to reconstruct `MatchRequest`, including `kind/opName`, operand order and roles, and optional fields `scalarPos/cmpMode/isBinary/requiredVariantId`.
- Ops that require temporary tile operands (for example `tprelu`, `trowsum/trowmax/trowmin`, `tcolsum`) SHALL expose `tmp` in descriptor operands with correct role/order.
- Interface-driven matching SHALL preserve legacy constraints for special cases, including:
  - compare/select family (`tcmp/tcmps/tsel/tsels`) dtype and mask/select-mode checks;
  - `tdivs` variant restriction via `requiredVariantId` (`tile_scalar` or `scalar_tile`);
  - `tcolsum` branch selection via `isBinary`;
  - `tprelu` tmp element type (`i8`) and `trem/trems` float-only restrictions.

### Requirement: 4.5~4.9 target ops must not silently fallback
For ops in PTO IR manual sections 4.5~4.9 (current lowering target set), OP-Lib lowering SHALL be mandatory once `PTOInstantiateAndLowerToLibCall` runs.

#### Scenario: Mandatory lowering for target op family
- During OP-Lib lowering, if a target op fails at descriptor->match-request, candidate selection, instance creation, or call rewrite, the pass SHALL emit error and fail.
- The pass SHALL NOT keep original target ops via warning-based fallback for these families.

#### Scenario: Post-pass no-residual check
- After lowering, non-OPLib user functions SHALL NOT contain remaining target PTO ops.
- If any residual target op exists, the pass SHALL emit error and fail.
