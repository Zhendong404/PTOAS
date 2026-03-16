# Compare and Bitwise Lowering Specification

## ADDED Requirements

### Requirement: Support Compare and Bitwise OpLib Lowering
The OpLib lowering pipeline SHALL be expanded to support 15 operators from the Compare and Bitwise families, handling integer types and specific comparison/selection modes.

#### Scenario: Lower Compare and Select operators
- The `tcmp`, `tcmps`, `tsel`, and `tsels` operators SHALL be lowered to OpLib calls.
- Comparison modes SHALL be correctly mapped to OpLib templates.
- Mask tiles and scalar select modes SHALL be handled for selection operators.

#### Scenario: Lower Bitwise and Shift operators
- Bitwise operators (`tand`, `tor`, `txor`, `tnot`) and shift operators (`tshl`, `tshr`) SHALL be lowered to OpLib calls.
- Both tile-tile and tile-scalar variants SHALL be supported.
- Integer data types SHALL be supported in the OpLib lowering flow for these operators.
