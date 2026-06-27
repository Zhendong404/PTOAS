## 1. Section Inference Semantics

- [x] 1.1 Extend uncovered-section normalization to classify frontend pipe
      initialization ops as Vector or Cube ownership signals.
- [x] 1.2 Add or update module-kind inspection so the same init ops participate
      in mixed-kernel kind inference where needed.
- [x] 1.3 Add focused lit coverage for valid inference and ambiguous-ownership
      failures involving uncovered frontend pipe initialization.

## 2. VPTO Split Integration

- [x] 2.1 Confirm the VPTO split pipeline consumes materialized
      `pto.section.vector` / `pto.section.cube` from logical mixed kernels
      without requiring user-authored helper kernels.
- [x] 2.2 Add pipeline regression coverage proving that a sectioned logical
      mixed kernel becomes physical Vector/Cube compile units before backend
      emission.
- [x] 2.3 Keep strict failure coverage for illegal cross-section SSA dependence
      or duplicate/invalid section structure.

## 3. PTODSL Surface And Verification

- [x] 3.1 Update PTODSL mixed-kernel examples and tests to use the hidden-section
      surface without explicit user-authored Vector/Cube decorators.
- [x] 3.2 Adjust PTODSL frontend verification expectations so the cv-split flow
      stays a single logical kernel while still showing inferred section
      normalization.
- [ ] 3.3 Run and fix `check-dsl` coverage for the new hidden-section flow.
