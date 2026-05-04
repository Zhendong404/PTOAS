# Tile Fusion Design Specification

## 1. Introduction

### 1.1 Background & Motivation

PTO instructions on the Davinci architecture operate on data blocks (Tiles) resident in the Unified Buffer (UB). While this model is expressive, it introduces two categories of overhead when multiple PTO instructions execute in sequence:

- **Memory Access Overhead**: Each PTO instruction (e.g., `pto.add`) reads its inputs from UB into vector registers and writes its output back to UB. When instructions are chained, intermediate results make a round-trip through UB — written by the producer, then re-read by the consumer — incurring redundant bandwidth consumption.
- **Control Overhead**: Each PTO instruction expands into a hardware loop parameterized by tile shape. Multiple PTO instructions mean multiple sets of loop initialization, branch prediction, and instruction dispatch.

**Example: `D = Relu(Add(A, B))`**

*Non-fused execution:*

1. **Loop 1 (Add)**: Read tiles A, B from UB → compute addition → write result tile C to UB.
2. **Loop 2 (Relu)**: Read tile C from UB → compute ReLU → write result tile D to UB.

*Bottleneck*: Tile C is written to UB and immediately read back — a completely redundant round-trip. Two independent loop controllers are set up and torn down.

*Fused execution:*

1. **Single Loop**: Read tiles A, B from UB → compute addition → **register-level handoff** → compute ReLU → write result tile D to UB.

*Benefit*: Eliminates the UB read/write of intermediate tile C. Merges two loop controllers into one, halving loop management overhead.

**Dataflow Comparison:**

```text
      [Non-Fused]                              [Fused]
      (Loop 1: Add)                         (Single Loop)
    ┌───────────────┐                     ┌───────────────┐
    │ Read A, B     │                     │ Read A, B     │
    │      │        │                     │      │        │
    │ Compute Add   │                     │      │        │
    │      │        │                     │      │ (Reg)  │
    │ Write C to UB │──┐ (UB Latency)     │ Compute Relu  │
    └───────────────┘  │                  │      │        │
    (Loop 2: Relu)     │                  │ Write D to UB │
    ┌───────────────┐  │                  └───────────────┘
    │ Read C fr UB  │◀─┘
    │      │        │               >> Eliminates intermediate write/read
    │ Compute Relu  │               >> Merges loop control logic
    │      │        │
    │ Write D to UB │
    └───────────────┘
```

**Summary of current bottlenecks**: Frequent UB reads and writes between chained PTO instructions lead to bandwidth contention, excessive loop control instruction overhead, and idle compute units while waiting on memory access.

### 1.2 Scope & Design Goals

#### 1.2.1 Fusion Scope

This feature targets all PTO instructions that map to the **Vector Pipeline (`PIPE_V`)** as documented in `docs/PTO_IR_manual.md`.

**Supported operations:**

| Category | Operations |
|---|---|
| Vector arithmetic (elementwise) | `pto.tadd`, `pto.tsub`, `pto.tmul`, `pto.tdiv`, `pto.tmax`, `pto.tprelu` |
| Vector-scalar arithmetic | `pto.tadds`, `pto.tsubs`, `pto.tmuls`, `pto.tmaxs` |
| Vector reduction & broadcast | `pto.trowsum`, `pto.tcolmax`, `pto.tbcast` |
| Vector bitwise | `pto.tand`, `pto.tor`, `pto.txor` |
| Local data movement | `pto.tmov` (ACC ↔ VEC domain), `pto.ttrans` (transpose) |

**Out of scope:** Matrix Pipeline (`PIPE_M`) matrix multiply instructions and DMA Pipeline (`PIPE_MTE`) data transfer instructions are excluded from internal fusion logic, though they may serve as endpoints of a fusion chain.

#### 1.2.2 Fusion Criteria

Not every pair of `PIPE_V` instructions is fusible. The following criteria must be satisfied:

##### Criterion 1: Iteration Space Consistency

Fused operations must execute within the same logical iteration domain. For PTO tiles, this means their logical compute shape (valid shape, i.e., `v_row` and `v_col`) must align to a single loop space.

- **Physical vs. logical decoupling**: Physical shape governs memory allocation and may differ between tiles; logical shape governs the compute range and must match.
- **Fusible examples**:
  - *Case A (exact match)*: OP1 and OP2 both have logical shape `16×128`. Fusion produces a single `16 × 128` hardware loop.
  - *Case B (physical layout differs)*: OP1's tile is physically `32×128` but valid region is `v_row=16, v_col=128`; OP2's tile is `16×128` with valid region `16×128`. Both iterate over the same logical domain — fusion is valid.
  - *Case C (heterogeneous output shape, same domain)*: `OP1: C = Add(A, B)` (output tile 16×128) followed by `OP2: d = RowSum(C)` (output vector 16×1). The iteration domain is compatible.
- **Non-fusible examples**:
  - *Case D (logical shape conflict)*: OP1 logical shape `16×128`, OP2 logical shape `8×128`. Mismatched iteration bounds would produce incorrect loop counts. Non-aligned fusion is not supported in the initial release.
  - *Case E (unprovable dynamic shapes)*: OP1's `v_row` is dynamic `%M1`, OP2's `v_row` is dynamic `%M2`. If the compiler cannot statically prove `%M1 == %M2` (e.g., via symbolic analysis tracing both to the same `pto.get_tensor_view_dim`), fusion is conservatively rejected.

##### Criterion 2: Data Dependence & Mapping Rules

- **Universality of elementwise ops**: Elementwise ops (e.g., `Add`, `Mul`, `Relu`) preserve data index mapping, making them the most flexible building blocks for fusion chains.
- **Weak dependence rule**: Fusion does not strictly require a data-flow dependence between ops. Independent parallel ops within the same iteration space can still benefit from "control-flow fusion" — merging loop logic to reduce overhead.
- **Deep fusion rule**: When ops have a producer-consumer relationship with a point-to-point (1:1) mapping, "memory-eliminating fusion" is triggered, using register-level handoff to eliminate intermediate tile UB traffic entirely.
- **Reduction adaptation rule**: Under A5 hardware support, elementwise + reduction op combinations follow the same deep fusion rules, enabling in-register accumulation with memory elimination.
- **Examples**:
  - *Case F (parallel, no dependence)*: `OP1: C = Add(A, B)` and `OP2: F = Mul(D, E)`. No data dependence, but merging into one loop reduces control overhead and improves compute unit utilization.
  - *Case G (elementwise deep fusion, 1:1 mapping)*: `OP1: C = Add(A, B)`, `OP2: D = Relu(C)`. The canonical 1:1 case. `C` is consumed by `Relu` directly in registers, completely eliminating the intermediate tile's physical memory allocation.
  - *Case H (cross-pattern fusion, elementwise + reduction)*: `OP1: C = Add(A, B)`, `OP2: d = RowSum(C)`. On A5, the `RowSum` instruction can perform cross-column accumulation within vector registers while the addition is still in-flight. Fusion merges loops and eliminates the UB storage and reload of the large tile `C`.

##### Criterion 3: Memory Layout Compatibility & Flexible Adaptation

**Standard rule (layout consistency)**: Input and output tiles participating in fusion must be fully compatible in metadata and hardware access patterns:

- **Metadata alignment**: Tile base layout (`blayout`, e.g., row-major / column-major), secondary layout (`slayout`), fractal size (`fractal`), and data type (`dtype`) must be identical.
- **Memory mapping consistency**: At the hardware level, this means the element at logical index `(i, j)` must occupy the same physical position in the register sequence regardless of which op is accessing it.
- **Consequence of mismatch**: If layouts differ (e.g., OP1 produces row-major, OP2 expects column-major), elements will be "misaligned" in registers even if the iteration space is identical. The standard rule therefore requires physical layout alignment for zero-overhead register-level handoff.

**Special-case adaptations:**

- **Virtual layout transform**: If the intervening op is a pure layout transform (e.g., `pto.ttrans`), the compiler may absorb the layout conversion into the compute loop's index mapping, adjusting vector register access strides or offsets to eliminate the physical transpose overhead.
- **Hardware shuffle acceleration**: Leveraging A5 hardware vector register shuffle/permute instructions to reorder data on-the-fly during computation, enabling fusion across different layout conventions.

**Example:**

- *Case I (transpose-eliminating fusion)*: `OP1: B = Transpose(A)` (via `pto.ttrans`), `OP2: C = Relu(B)`. Executed separately, `Transpose` requires a physical UB shuffle. With fusion, the compiler can generate a "column-traversal" `Relu` kernel that reads tile `A` directly. The logical intermediate `B` disappears, along with the redundant physical transpose memory traffic.

##### Criterion 4: Register & Hardware Parameter Budget

- **Physical register limit**: The Davinci A5 has 32 vector registers (`V0`–`V31`). The total number of live variables in a fusion chain (inputs, intermediates, temporaries) must not exceed this threshold. Exceeding it causes register spilling, forcing data back to UB/L1 and severely degrading performance.
- **VF parameter limit**: Vector Function (VF) hardware loop parameter lists (tile physical addresses, strides, shapes, and other metadata) are capped at 32 entries. If a fused mega-loop references too many distinct tile objects, VF invocation will exceed this parameter budget.
- **Example**:
  - *Case J (dual parameter & register overflow)*: Attempting to fuse a complex chain with 18 distinct input tiles (e.g., 18-way tile addition).
    - *Analysis*: (1) **Parameter spill**: 18 input tile addresses + 1 output tile address + corresponding stride metadata can easily hit the 32-entry VF parameter ceiling. (2) **Register spill**: If loop unrolling causes more than 32 128-bit vector values to be simultaneously live, the hardware cannot sustain register-level handoff without writing back to memory.
    - *Decision*: The compiler must split the fusion chain at an intermediate point, writing the partial result to UB and executing two independent VF loops.

##### Criterion 5: Canonically Supported Patterns

- **Linear chain**: `A → B → C` continuous elementwise computation.
- **Parallel independent fusion**: Multiple ops in the same iteration space with no direct data dependence, combined into one hardware loop to reduce control overhead.
- **Broadcast fusion**: `pto.tbcast` immediately followed by elementwise ops.
- **Terminal reduction fusion**: A sequence of elementwise ops followed by `pto.trowsum` or `pto.tcolmax`.
- **Reduction + elementwise**: A reduction output vector directly consumed as input to a subsequent elementwise op.
- **Elementwise + broadcast**: An elementwise result tile directly consumed as broadcast input, expanding at the register level.
- **Reduction + broadcast**: Common in Softmax patterns. Row/column extrema from reduction are broadcast back to tile shape for subsequent computation.

### 1.3 Core Design Principle

The central technique is to fuse the loop bodies of multiple PTO instructions at the MLIR level, passing intermediate data directly through vector registers. This achieves:

- **Reduced UB traffic**: Register-level data pipelining replaces physical tile write/read round-trips, eliminating redundant bandwidth consumption.
- **Reduced control overhead**: Multiple hardware loops are merged into a single loop structure, reducing loop prologue/epilogue, branch prediction, and instruction dispatch overhead.
- **Improved compute efficiency**: Instruction reordering and unrolling keep vector compute units as close to saturation as possible.
- **Transparency**: The fusion boundary is explicit and predictable, so upstream compiler frameworks can anticipate which instruction combinations will receive register-level acceleration.

---

## 2. Hardware Architecture & Constraints

### 2.1 Davinci A5 Memory Hierarchy & Access Costs

- **Global Memory (GM)**: Hundreds of GB/s bandwidth; high latency (hundreds of cycles).
- **Unified Buffer (UB)**: TB/s bandwidth; moderate latency. This is the default input/output medium for PTO instructions.
- **Vector Registers**: Single-cycle access at instruction granularity.

**Fusion motivation**: Passing data between PTO instructions through UB creates a bandwidth bottleneck. Routing data through vector registers instead elevates the data path to register-level bandwidth and eliminates physical access latency.

### 2.2 Vector Compute Pipeline & Execution Model

- **Dual-issue pipeline**: Davinci supports two vector compute pipelines (Pipe V). Intelligent instruction scheduling can hide compute latency.
- **Hardware loop (Vector Function)**: The built-in loop controller has non-trivial initialization overhead. A fused "mega-loop" amortizes prologue and epilogue cycles across more compute work.
- **Vector mask (VMSK)**: Used for handling misaligned boundaries. Fusion must ensure that mask logic from different ops is composable within a single loop iteration.

### 2.3 Physical Alignment & Tile Layout Constraints

- **32B/512B alignment**: The Davinci architecture enforces strict alignment on UB addresses and tile widths.
- **Padding**: When the logical (valid) shape is smaller than the physical shape, the hardware fills the physical tile edges with invalid data. Fusion must strictly bound computation to the valid region to avoid contaminating results with padding values.
- **Fractal layout**: Specialized layout for matrix/fractal operations. Vector ops working with fractal layouts require additional shuffle or stride calculations.

---

## 3. Design Challenges

### 3.1 Register Pressure

- **Spill risk**: Fusing multiple ops significantly increases the number of live variables. If the required vector registers exceed the hardware budget, spilling occurs, writing data back to UB or L1 and negating the memory benefit of fusion.
- **Lifetime overlap**: The fused long loop extends intermediate value lifetimes, increasing the complexity of register allocation.

### 3.2 Memory Layout & Alignment

- **Layout mismatch**: A producer op may output row-major data while the consumer expects column-major input. Fusing such ops may require inserting expensive transpose or shuffle instructions that offset the memory savings.
- **Non-elementwise ops**: Reduction or data-movement op fusion involves complex index transformations and synchronization logic.

### 3.3 Loop Control & Synchronization

- **Loop structure divergence**: Ops with different iteration ranges or strides require careful loop peeling or padding when fused.
- **Fine-grained synchronization**: In certain pipeline architectures, fused long instruction sequences may cause hardware pipeline deadlocks or violate data dependence ordering.

### 3.4 Cost Model & Boundary Decisions

- **Greedy strategy limitations**: Over-aggressive fusion can lead to code bloat and instruction cache misses.
- **Adaptive decision-making**: Determining the optimal fusion boundary under varying tile shapes and hardware configurations is an NP-hard problem in the general case.

---

## 4. IR Representation & Pipeline Integration

### 4.1 PTO Dialect Context

The fusion system operates on the PTO tile-level dialect, which includes core instructions such as `pto.alloc_tile`, `pto.load_tile`, `pto.store_tile`, and the compute ops listed in §1.2.1. The existing lowering path for these ops provides the foundation on which fusion is layered.

### 4.2 Fusion Pipeline Position

#### 4.2.1 Activation Conditions

Fusion is active only in the A5 VPTO backend mainline in `tools/ptoas/ptoas.cpp`. All of the following must hold:

- `--pto-backend=vpto`
- `--pto-arch=a5`
- `--enable-op-fusion` explicitly passed

#### 4.2.2 Input Level Support

| Level | Status | Notes |
|---|---|---|
| `level2` | Supported | Fusion adapters run before `PlanMemory`. |
| `level3` | Supported | Fusion adapters operate directly on manual-address tile-native IR. |
| `level1` | N/A | No viable input surface in the current migration scope. |

#### 4.2.3 Mainline Pass Sequence

1. **Shared tile-native pre-processing**:
   ```
   PTOAssignDefaultFrontendPipeId → PTOLowerFrontendPipeOps →
   PTOInferValidatePipeInit → LoweringSyncToPipe → InferPTOLayout →
   PTOA5NormalizeTMov
   ```

2. **Fusion core** (inserted before the `PlanMemory` decision point, when A5 VPTO fusion conditions are met):
   ```
   FusionPlan → OpScheduling → PTOFusionRegionGen
   ```

3. **Level-specific adapters**:
   - `level2`: `shared fusion core → PlanMemory → PTOResolveReservedBuffers → (optional) PTOInsertSync`
   - `level3`: `shared fusion core → skip PlanMemory → PTOResolveReservedBuffers`
     (If `--enable-insert-sync` is passed under `level3`, a warning is emitted and the flag is ignored.)

4. **ExpandTileOp seam** (transition to VPTO backend lowering):
   ```
   ExpandTileOp → PTOInlineLibCall → FoldTileBufIntrinsics → SCCP → Canonicalizer
   ```

5. **Post-lowering fusion lifecycle** (only for fused A5 VPTO path):
   ```
   PTOLowLevelLoopFusion → Canonicalizer → CSE →
   PTOFusionPredicateElision → PTOFusionLoadStoreElision →
   PTOFlattenFusionRegion → CSE
   ```

6. **VPTO emission preparation**:
   ```
   Canonicalizer/CSE → VPTOPtrNormalize → VPTOPtrCastCleanup →
   ReconcileUnrealizedCasts → PTOVPTOExpandBridgeOps →
   PTOInferVPTOVecScope → Canonicalizer → CSE →
   PTOValidateVPTOEmissionIR
   ```

#### 4.2.4 Non-target Paths

- The EmitC backend ignores `--enable-op-fusion`.
- When `--enable-op-fusion` is not set, the standard VPTO path does not form `pto.fusion_region` and does not enter the post-lowering fusion lifecycle.
- The backend seam is fixed at `ExpandTileOp`; the legacy `View2Memref` / `PTOToA5VM` mainline has been removed.

---

## 5. Detailed Pass Design

### 5.0 Fusion Pipeline Overview

```text
    [tile-native PTO IR]
               │
               ▼
    5.1  PreFusionAnalysis (analysis-only)
               │
               ▼
    5.2  Iteration Domain Proof (embedded in pre-analysis; no standalone ShapeInferencePass)
               │
               ▼
    5.3  FusionPlan
               │
               ▼
    5.4  OpScheduling (intra-group physical clustering)
               │
               ▼
    5.5  PTOFusionRegionGen (region encapsulation)
               │
               ▼
    5.6  Level-specific Adapters (before ExpandTileOp)
         level2: PlanMemory → ResolveReservedBuffers → optional InsertSync
         level3: skip PlanMemory → ResolveReservedBuffers
               │
               ▼
    5.7  VPTO Backend Seam
         (ExpandTileOp → InlineLibCall → FoldTileBufIntrinsics →
          SCCP → Canonicalizer)
               │
               ▼
    5.8  PTOLowLevelLoopFusion
               │
               ▼
    5.9  Post-fusion Normalization & Predicate Elision
         (Canonicalizer → CSE → PTOFusionPredicateElision)
               │
               ▼
    5.10 PTOFusionLoadStoreElision (intra-region load/store elimination)
               │
               ▼
    5.11 PTOFlattenFusionRegion → CSE (region unwrapping)
               │
               ▼
    5.12 VPTO Emission Preparation
         (ptr normalize / bridge expand / vecscope infer / emission IR validate)
               │
               ▼
    5.13 VPTO Backend Emission
         (VPTO text / LLVM emission)
```

### 5.1 PreFusionAnalysis (Analysis-Only)

**Motivation.** Provides `FusionPlan` with reusable block-local analysis results without mutating IR. All decisions about which ops can participate in tile fusion, which values escape across local/hard boundaries, and which ops share an iteration domain are consolidated into a single analysis pass, consumed by planning, region formation, and subsequent region-local cleanup.

**Pipeline Position.**
- **Pre-conditions**: IR is still in the tile-native `tile_buf` world, before the `ExpandTileOp` seam.
- **Post-conditions**: Produces a `PreFusionAnalysis` result object. This pass is not inserted standalone in the default mainline; `FusionPlanPass` consumes it via `getAnalysis<pto::PreFusionAnalysis>()`.

**I/O Specification.**
- **Input**: `func::FuncOp` containing PTO tile-level IR.
- **Output**: Per-basic-block `FusionBlockAnalysis`, containing:
  - `computeNodes`: Compute nodes eligible for pre-fusion modeling.
  - `edges`: Producer/consumer dependence edges.
  - `valueLiveness` / `writeInstances`: Value and write-instance liveness categories and escape classes.
  - `iterationDomainClasses`: Iteration domain equivalence classes keyed by `(v_row, v_col)` proof results.

**Logic & Constraints.**
- **Op semantic classification**: `FusionOpSemantics` categorizes ops as `Compute`, `LocalBoundary`, or `HardBoundary`.
  - `treshape` is treated as `LocalBoundary` and blocks cross-boundary local planning and scheduling.
  - Ops lacking the `OpLibOpInterface`, or carrying regions, calls, or unknown side effects, are classified as `HardBoundary`.
- **Recognized compute families**:
  - `Elementwise`: `tadd/tsub/tmul/tdiv/tmax/tmin` and corresponding scalar variants, `texp`
  - `ScalarExpand`: `texpands`
  - `RowBroadcastBinary`: `trowexpandmul`, `trowexpanddiv`
  - `ReduceRow/ReduceCol`: `trowsum/trowmax/trowmin`, `tcolsum/tcolmax/tcolmin`
- **Dependence & liveness modeling**:
  - Tile inputs/outputs are collected via SSA use-def chains and DPS output normalization.
  - Per-block tracking of consumers, local boundary users, hard boundary users, and out-of-block escapes.
  - Write instances are further classified as `Internal`, `LocalBoundaryExternal`, or `HardExternal`, used for region output/frontier computation.
- **Analysis scope**: Strictly limited to a single basic block. Cross-block or cross-CFG planning is not performed.

**Invariants & Side Effects.**
- Pure analysis; no IR mutation.
- Debugging and lit inspection are available via `pto-pre-fusion-analysis` and `pto-print-pre-fusion-analysis`, but these are not part of the default backend mainline.

### 5.2 Iteration Domain Proof

**Motivation.** Shape inference has not yet been extracted into a standalone pass. The mainline instead relies on a conservative iteration domain proof embedded in `FusionAnalysis.cpp`: planning proceeds only when the anchor tile's effective rank-2 shape can be statically proven consistent at compile time.

**Pipeline Position.**
- **Pre-conditions**: Depends on op semantics from §5.1 and tile type / `pto.bind_tile` metadata.
- **Post-conditions**: No IR mutation. Analysis results receive `IterationDomainInfo` annotated as `Proven` or `Unproven` with a failure reason.

**I/O Specification.**
- **Input**: Compute op tile inputs/outputs and their valid shape information.
- **Output**: `IterationDomainInfo { vRow, vCol, proof, unprovenReason }`.

**Logic & Constraints.**
- **Proof sources (current)**:
  - Preferentially reads `TileBufType::validShape`.
  - If the value originates from `pto.bind_tile`, its constant `validRow/validCol` overrides the type-level static shape.
  - `Elementwise` aggregates across inputs and outputs; `ScalarExpand/RowBroadcastBinary` uses the output domain; `ReduceRow/ReduceCol` uses the input domain.
- **Failure conditions**:
  - Any critical dimension is dynamic.
  - Anchor group members have inconsistent `(v_row, v_col)`.
  - Recoverable tile domain information is absent.
- **Practical boundary**: The implementation explicitly does not attempt to prove equivalence of dynamic symbols. Chains with dynamic `v_row/v_col` remain `Unproven` and are conservatively rejected during planning (§5.3).

**Invariants & Side Effects.**
- No new attributes, no type rewriting, no shape solver is introduced.

### 5.3 FusionPlanPass

**Motivation.** Consumes analysis results from §5.1 and §5.2 to assign stable group metadata to ops within a block, forming the single input contract for subsequent scheduling and region formation. The current implementation prioritizes conservative correctness over an open pluggable strategy framework.

**Pipeline Position.**
- **Pre-conditions**: `PreFusionAnalysis` is valid; ops are still at the tile-level PTO IR.
- **Post-conditions**: Accepted group members receive:
  - `pto.fusion.group_id`
  - `pto.fusion.order`

**I/O Specification.**
- **Input**: `FusionBlockAnalysis`.
- **Output**: Original PTO IR with planning metadata. Only groups of size ≥ 2 are materialized.

**Logic & Constraints.**
- **Current strategy**:
  - `ConservativeDAGGreedyStrategyEngine`
  - `ConservativeDAGGreedyCostModel`
- **Plannable op subset** (narrower than the analyzable compute families in §5.1):
  - `tadd/tsub/tmul/tdiv/tmax/tmin`
  - `tadds/tsubs/tmuls/tdivs/tmaxs/tmins`
  - `texp`
  - `texpands`
  - `trowexpandmul`
  - `trowexpanddiv`
- **Seed conditions**:
  - Op must belong to the plannable set above.
  - Its iteration domain class must be `Proven`.
- **Append conditions**:
  - Candidate shares the same `iterationDomainClass` as the group's first member.
  - Candidate has at least one direct data-flow edge to the current group.
  - Cost model score: `dependencyBenefit + loopMergeBenefit - liveTilePenalty - vfParameterPenalty > 0`.
- **Cost model parameters (current)**:
  - `dependencyBenefit = 4 × connectionCount`
  - `loopMergeBenefit = 4`
  - Penalty begins when `liveTileCount > 10`
  - Penalty begins when `vfParameterCount > 12`
- **Result ordering**:
  - Intra-group order is stable-sorted by `blockOrder/id`.
  - `group_id` is assigned in stable block order of group leaders.

**Invariants & Side Effects.**
- Metadata only; ops are not moved.
- The strategy interface is not currently exposed as a pluggable configuration point. ML/AI-driven decision interfaces remain future work.

### 5.4 OpSchedulingPass

**Motivation.** Compresses the groups planned in §5.3 into contiguous spans within the block, providing the structural prerequisite for `PTOFusionRegionGen` (one group → one contiguous interval).

**Pipeline Position.**
- **Pre-conditions**: Ops carry complete `pto.fusion.group_id/order` metadata.
- **Post-conditions**: Each group forms a single contiguous span in the block; group membership is unchanged.

**I/O Specification.**
- **Input**: PTO IR with planning metadata.
- **Output**: Physically reordered PTO IR.

**Logic & Constraints.**
- **Barrier classification**:
  - `Movable`: Ordinary movable compute op.
  - `LocalBoundary`: e.g., `treshape`; may be crossed when no tile dependence conflict exists.
  - `HardBoundary`: Calls, region ops, ops with unknown side effects, or otherwise unsafe-to-move boundaries.
- **Scheduling strategy**:
  - Group members are collected by `group_id` and sorted by `pto.fusion.order`.
  - For each subsequent member, the scheduler first attempts to move it forward to just after the current `placement`.
  - If forward movement is blocked, it attempts the reverse — pushing `placement` backward — provided no later-crossing constraints are violated.
- **Legality checks**:
  - Must not cross the definition point of any operand.
  - Must not move a producer past any of its consumers.
  - Must not cross a hard boundary.
  - When crossing a local boundary, both sides must not share tile input/output dependencies.

**Invariants & Side Effects.**
- Rewrites intra-block op order.
- Does not modify group metadata or the CFG.

### 5.5 PTOFusionRegionGenPass

**Motivation.** Wraps each contiguous span into an explicit `pto.fusion_region`, providing a container boundary for the VPTO post-lowering stages: region-local loop fusion, predicate cleanup, and load/store elimination.

**Pipeline Position.**
- **Pre-conditions**: Each `group_id` in the block is a single contiguous span.
- **Post-conditions**: Each span is wrapped in a `pto.fusion_region` with an explicit `pto.yield` declaring the externally visible frontier.

**I/O Specification.**
- **Input**: Block-local group spans after scheduling.
- **Output**: PTO IR with `pto.fusion_region` wrappers.
  - Regions do not model explicit input block arguments.
  - Region bodies implicitly capture parent-scope SSA values.
  - `pto.yield` returns only values that are genuinely visible outside the region.

**Logic & Constraints.**
- **Span identification constraints**:
  - A single `group_id` must appear in exactly one contiguous span per block; otherwise an error is raised.
  - `pto.fusion.order` must be strictly increasing within each group.
- **Frontier computation**:
  - `PTOFusionRegionGen` combines use-def analysis with `PreFusionAnalysis` write-instance escape information to identify escaping values that must appear in the region result list.
  - Values used only within the region, or with external uses that cannot be legally replaced by a region result, are not blindly promoted.
- **Structural constraints**:
  - One group → one region.
  - No extra region input operands are generated.
  - Empty results / empty `pto.yield` are allowed.

**Invariants & Side Effects.**
- Introduces nested regions, significantly altering IR structure.
- This is the boundary where a logical fusion group is concretized into a form recognizable by the downstream backend.

### 5.6 Level-Specific Adapters (Before `ExpandTileOp`)

**Motivation.** Rather than maintaining a separate fusion implementation per level, differences are confined to adapters: the shared fusion core is fixed before `ExpandTileOp`, and `level2`/`level3` differ only in the constraints applied before and after it.

**Pipeline Position.**
- **Pre-conditions**: `pto.fusion_region` exists; body still contains PTO tile ops.
- **Post-conditions**:
  - `level2`: `PlanMemory → PTOResolveReservedBuffers → optional PTOInsertSync` completed, with `pto.fusion_region` preserved.
  - `level3`: `PlanMemory` skipped; `PTOResolveReservedBuffers` completed; explicit-address / manual-sync contract preserved.

**Logic & Constraints.**
- The shared adapter insertion point is fixed before the `PlanMemory` decision point.
- `level2` contract:
  - `PlanMemory` may rewrite `alloc_tile` (inside and outside regions) into `pto.pointer_cast`.
  - `PTOInsertSync` may append tail barriers after regions.
- `level3` contract:
  - `PlanMemory` is not re-entered.
  - If `--enable-insert-sync` is explicitly passed, a warning is emitted and the flag is ignored to preserve the manual sync contract.
- `level1`:
  - Architecturally, it should also follow "shared fusion core before PlanMemory."
  - No viable input surface exists in the current branch; implementation and test coverage remain N/A.

### 5.7 VPTO Backend Seam

**Motivation.** `ExpandTileOp` is the hard boundary from PTO tile IR to VPTO authoring IR. Tile fusion must complete group and region modeling before this seam, while ensuring `pto.fusion_region` survives across it for post-lowering cleanup.

**Pipeline Position.**
- **Pre-conditions**: Input is tile-native PTO IR; `pto.fusion_region` may already be present.
- **Post-conditions**:
  - Tile-level PTO ops are replaced by TileLang helper `call`/`func.call`.
  - Helper bodies are inlined into the parent function.
  - `pto.tile_buf_addr` / `pto.tile_valid_rows` / `pto.tile_valid_cols` intrinsics are folded into VPTO-side `memref`/`ptr`/constants.
  - `pto.fusion_region` is preserved on the fused path.

**Logic & Constraints.**
- Fixed sequence:
  1. `ExpandTileOp`
  2. `PTOInlineLibCall`
  3. `FoldTileBufIntrinsics`
  4. `SCCP`
  5. `Canonicalizer`
- Practical boundary:
  - The legacy `PTOViewToMemref` bridge has been removed and is not a valid seam precondition.
  - `FoldTileBufIntrinsics` depends on native tile metadata; inputs that are not native producers (e.g., block arguments lacking address/valid-shape metadata) will cause a conservative failure downstream.

### 5.8 PTOLowLevelLoopFusion

**Motivation.** The actual loop fusion now occurs after `ExpandTileOp`, operating on the current VPTO post-lowering `scf.for + memref/pto.v*` structure within `pto.fusion_region`.

**Pipeline Position.**
- **Pre-conditions**: Input contract is a VPTO post-lowering loop nest preserved inside `pto.fusion_region`.
- **Post-conditions**: Adjacent fusible loop stages are merged into a single carrier loop with a shared loop header.

**Logic & Constraints.**
- **Stage identification**:
  - Each stage consists of setup ops, one or more layers of isomorphic `scf.for`, and leaf `pto.vlds/vsts/vadd/...` or related pure ops.
  - Only adjacent stages with equivalent loop headers and safely reorderable preludes/setup are considered.
- **Legality conditions**:
  - `sameForHeader(lhs, rhs)`: Lower bound, upper bound, step, and loop attributes must be exactly equivalent.
  - Prelude/setup ops must be side-effect-free or analyzable memory preludes.
  - Moving a prelude across stages must not create potential alias conflicts with the preceding stage's memory roots.
- **Practical boundary**:
  - This is a conservative matcher; no aggressive loop normalization is performed.
  - Only adjacent stages are fused; cross-region or cross-control-flow global loop stitching is not attempted.

### 5.9 Post-Fusion Normalization & Predicate Elision

**Motivation.** After low-level loop fusion, a standard `Canonicalizer + CSE` pass cleans up the IR, followed by targeted elimination of redundant VPTO predicate materialization inside fusion regions, reducing noise for subsequent load/store elimination.

**Pipeline Position.**
- **Pre-conditions**: `PTOLowLevelLoopFusion` completed.
- **Post-conditions**: Duplicate `pto.plt_*` computations are compressed; the VPTO loop body seen by subsequent memory elimination is cleaner.

**Logic & Constraints.**
- The specialized `PTOVPTOIfCanonicalize` / `A5VMIfCanonicalize` passes from earlier branches have been retired; only the shared `Canonicalizer` is used.
- `PTOFusionPredicateElision` focuses on `pto::PltB8/B16/B32Op`.
- Conservative value equivalence checks are performed within fusion regions, including:
  - Pure op result equivalence.
  - Direct recursive equivalence of loop-carried `iter_arg` and `plt.scalar_out`.
- Predicates are reused only when equivalence is provable; side effects and complex loop recursion are not crossed.

### 5.10 PTOFusionLoadStoreElision

**Motivation.** Once a stable VPTO carrier loop is formed, eliminate intra-region store/load round-trips that serve only as temporary handoffs, contracting the region-local data path toward register/vector-value direct transfer.

**Pipeline Position.**
- **Pre-conditions**: Low-level loop fusion, global canonicalization, and predicate elision are complete.
- **Post-conditions**: Local `pto.vsts → pto.vlds` round-trips are eliminated; non-escaping tail stores may also be removed.

**Logic & Constraints.**
- **Input contract**: Operates on VPTO post-lowering loop bodies within `pto.fusion_region`, typically of the form `scf.for + memref.subview + memref.cast + pto.vlds/vsts/vadd/...`.
- **Core behavior**:
  - Normalizes tracked memref root values, penetrating `bind_tile`, `memref.cast`, `reinterpret_cast`, `transpose`, and similar wrappers.
  - Uses `pto.yield` / region results as the external visibility frontier to avoid deleting write-backs that must escape the region.
  - Conservatively eliminates matched store/load pairs and performs frontier-aware tail-store cleanup.
- **Practical boundary**:
  - Memref effects that cannot be resolved via alias analysis trigger conservative bail-out.
  - Only fusion-region-local patterns are handled; this is not a general-purpose DSE pass.

### 5.11 PTOFlattenFusionRegion

**Motivation.** Once all region-local backend optimizations are complete, the `pto.fusion_region` structural container is no longer needed and must be explicitly flattened back into the parent block, restoring flat VPTO IR that the backend emitter can consume.

**Pipeline Position.**
- **Pre-conditions**: After §5.10, the remaining ops inside the region are in final backend-ready form.
- **Post-conditions**: `pto.fusion_region` and `pto.yield` are deleted; the parent block contains only ordinary low-level VPTO ops.

**Logic & Constraints.**
- All ops in the region body except the terminator are moved before the wrapper.
- `pto.fusion_region` results are replaced with the corresponding `pto.yield` operands.
- The `pto.yield` and wrapper op are erased.
- A final `CSE` pass runs after flattening to remove redundant values whose lifetimes ended with the wrapper.

### 5.12 VPTO Emission Preparation

**Motivation.** After the post-lowering fusion lifecycle completes, the IR must pass through a final stage of VPTO emission preparation — strongly coupled to the backend emitter — before it becomes exportable as VPTO text or LLVM output.

**Pipeline Position.**
- **Pre-conditions**: IR is flattened, backend-ready VPTO form.
- **Post-conditions**: Pointer normalization, bridge expansion, vecscope inference, and emission legality validation are complete.

**Logic & Constraints.**
- Fixed sequence:
  1. `Canonicalizer + CSE`
  2. `VPTOPtrNormalize`
  3. `VPTOPtrCastCleanup`
  4. `ReconcileUnrealizedCasts`
  5. `PTOVPTOExpandBridgeOps`
  6. `CSE`
  7. `PTOInferVPTOVecScope`
  8. `Canonicalizer + CSE`
  9. `PTOValidateVPTOEmissionIR`
- This stage is no longer part of tile fusion proper, but it determines whether the VPTO IR produced by the preceding pipeline can be emitted successfully.

### 5.13 VPTO Backend Emission

**Motivation.** The ultimate target of the tile fusion mainline is the existing VPTO backend emitter: either output cleaned-up VPTO text, or continue through LLVM emission for downstream toolchain consumption.

**Pipeline Position.**
- **Pre-conditions**: IR has passed emission preparation and validation.
- **Post-conditions**: VPTO text output or LLVM-level backend artifact is produced.

**Logic & Constraints.**
- `--emit-vpto` outputs the prepared VPTO IR directly.
- `--vpto-emit-hivm-llvm` / `--vpto-emit-hivm-bc` continue through LLVM/HIVM export.
- From the tile fusion perspective, the concern here is whether the preceding passes produced structurally correct VPTO IR — not the internal details of the backend emitter itself.

---

## 6. Key Algorithms

- **Pre-analysis-driven block-local DFG modeling**: Extracts compute/local-boundary/hard-boundary classifications, value liveness, and write escape classes within the `tile_buf` world, serving as the unified foundation for all subsequent decisions.
- **Conservative DAG greedy planning**: The current default strategy is `ConservativeDAGGreedyStrategyEngine + ConservativeDAGGreedyCostModel` — not an open plugin system.
- **Span-compression scheduling**: Uses barrier classification and bidirectional movement rules to compact discrete groups into contiguous spans while preserving SSA and boundary legality.
- **Post-lowering stage matcher**: `PTOLowLevelLoopFusion` matches same-header loops and reorderable preludes in the `scf.for + memref/pto.v*` VPTO layer, performing conservative adjacent-stage fusion.
- **Frontier-aware cleanup**: Both `PTOFusionPredicateElision` and `PTOFusionLoadStoreElision` rely on the fusion-region frontier to avoid eliminating values that must remain externally visible.

---

## 7. Cross-Module Interactions

- **Shared pre-backend normalization**: The first half of the tile fusion pipeline shares `LoweringSyncToPipe`, `InferPTOLayout`, `PlanMemory`, `PTOResolveReservedBuffers`, and optional `PTOInsertSync` with the VPTO backend. `level3` explicitly skips `PlanMemory`.
- **ExpandTileOp seam contract**: `ExpandTileOp` is the hard PTO → VPTO boundary. Tile fusion must form `pto.fusion_region` before this seam and ensure the wrapper survives `InlineLibCall` / `FoldTileBufIntrinsics`.
- **Backend emitter contract**: Upstream passes must converge the IR to a form accepted by `prepareVPTOForEmission()`; otherwise backend emission will fail.
- **Testing & OpenSpec**: Current behavioral constraints are encoded in `test/lit/pto/op_fusion_*`, `test/basic/expand_tile_op_tilelang_tadds.pto`, `test/vpto/auto_vecscope_infer_boundary.pto`, and `openspec/changes/reintroduce-vpto-tile-fusion/*`. The legacy LibCall/CCE design is no longer the source of truth.

---

## 8. Verification & Testing

### 8.1 Functional Verification

- Primary regression coverage is in `test/lit/pto/op_fusion_*` and `fusion_region_*`. Legacy `test/tile_fusion/*.mlir` and test-only CLI paths are deprecated.
- To inspect frontend group/schedule/region results, use:
  - `--mlir-print-ir-after=pto-fusion-plan`
  - `--mlir-print-ir-after=pto-op-scheduling`
  - `--mlir-print-ir-after=pto-fusion-region-gen`
- To verify backend mainline IR shape, use:
  - `--mlir-print-ir-after=pto-expand-tile-op`
  - `--mlir-print-ir-after=pto-low-level-loop-fusion`
  - `--mlir-print-ir-after=pto-fusion-predicate-elision`
  - `--mlir-print-ir-after=pto-fusion-load-store-elision`
  - `--mlir-print-ir-after=pto-flatten-fusion-region`
- Negative test cases should cover: dynamic shapes, local boundaries, hard boundaries, and ops with immovable side effects.

### 8.2 Performance Metrics

- **Structural metrics**:
  - `pto.fusion.group_id/order` correctness.
  - `pto.fusion_region` wrapping exactly one contiguous span.
  - `pto.fusion_region` survival across `PlanMemory`, `ResolveReservedBuffers`, optional `PTOInsertSync`, and the `ExpandTileOp` seam.
- **Low-level metrics**:
  - Reduction in redundant `pto.plt_*` predicates.
  - Reduction in intra-region `vlds/vsts` round-trips.
- **End-to-end test commands**:
  - `llvm-lit -sv test/lit/pto/op_fusion_*`
  - `llvm-lit -sv test/lit/pto/fusion_region_*`
  - Representative fused A5 VPTO samples: `test/lit/pto/op_fusion_adapter_placement_level2_tadd.pto` and `test/lit/pto/op_fusion_adapter_placement_level3_tadd.pto`
  - When needed, cross-reference key pipeline stages with `--mlir-print-ir-after=<pass>`.

---

## 9. Current Boundaries & Future Directions

### 9.1 Current Boundaries

- Mainline covers A5 VPTO backend only; EmitC is excluded.
- Planning is limited to conservative block-local groups; cross-basic-block fusion is not supported.
- Dynamic iteration domains cannot currently be proven; such chains are rejected.
- `PreFusionAnalysis` recognizes a broader set of compute families than `FusionPlan` currently permits for planning.
- `level1` has no viable input surface in the current migration scope; only a design position is reserved, with no implementation or regression coverage.

### 9.2 Future Directions

- Promote dynamic `v_row/v_col` proof to a standalone shape inference mainline pass.
- Broaden planner support for reduce/broadcast combination patterns.
- Extend post-lowering loop fusion and load/store elimination coverage while preserving the stable `pto.fusion_region` contract.
