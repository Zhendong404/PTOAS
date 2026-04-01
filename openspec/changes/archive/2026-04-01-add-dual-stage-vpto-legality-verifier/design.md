## Context

### 范围

本 design 只覆盖 `--pto-backend=vpto` 路径中的 VPTO IR legality 验证，不改变 EmitC backend、非 VPTO backend、NPU runtime 验证或底层 LLVM IR 合法性规则。

目标对象分成两个阶段：

1. authoring stage

- 输入是 backend mainline 结束后的 VPTO IR；
- 该形态仍允许 memref-first 地址表达，同时也允许 hand-written ptr-form VPTO IR；
- 这是库开发者和高级用户最直接接触的 VPTO authoring surface。

2. emission stage

- 输入是 `PTOVPTOPtrBoundary` 之后的 ptr-form VPTO IR；
- 该形态作为 `--emit-vpto`、text emission、LLVM emission 的正式发射前契约；
- 该阶段的职责是验证“优化后是否合法、boundary 后是否合法”。

### 当前状态

当前仓库已经具备三块相关基础能力：

1. op/type verifier 已覆盖大量局部规则

- `lib/PTO/IR/VPTO.cpp` 已验证 `!pto.vreg` 的 2048-bit 宽度、多数 load/store/copy 的地址空间约束、ptr-only family、predicate token 与部分 `dist` / `mode` 枚举。
- `lib/PTO/IR/PTO.cpp` 已验证 `get_buf` / `rls_buf` 的 attr 形态和取值范围。

2. VPTO backend 主线已经相对稳定

- `tools/ptoas/ptoas.cpp` 中 VPTO mainline 已固定为：
  - shared pre-backend
  - `PTOVPTOVersionSelection`
  - `PTOToVPTO`
  - `PTOVPTOIfCanonicalize`
  - `PTOLowLevelLoopFusion`
  - `Canonicalizer`
  - `PTOVPTOTrivialLoopCanonicalize`
  - `CSE`
  - `PTOFusionPredicateElision`
  - `PTOFusionLoadStoreElision`
  - `PTOFlattenFusionRegion`
  - `CSE`
  - `PTOVPTOExpandBridgeOps`

3. emission boundary 已有 ptr canonicalization

- CLI 路径会 clone module，再运行 `PTOVPTOPtrBoundary`；
- LLVM emission wrapper 也会在 clone 上运行 `convertVPTOEmissionBoundaryToPtr`；
- 但当前没有明确的“ptr-form emission legality verifier”。

4. 完整版 `docs/vpto-spec.md` 已经给出更完整的 VPTO contract 基线

- `llvm.loop.aivector_scope` 的语义边界已经定义为 VF 级 execution scope；
- `!pto.mask<b8|b16|b32>` 已明确为“256-bit 谓词寄存器 + granularity 视图”，而不是 lane-count vector；
- stateless / predicate family 的 `buf_like` 已定义为 `memref<...>` 或 `!pto.ptr<...>`；
- stateful `%base/%base_out` family 已定义为 pointer-only，不接受 memref authoring。

当前缺口集中在两类规则：

- 跨 op、跨 vec scope 的结构性合法性；
- 优化后、boundary canonicalization 后的最终 emission contract。

### 实现约束

- 不能重复实现已有单-op verifier 已覆盖的规则。
- `!pto.mask` 需要升级为 `!pto.mask<b8|b16|b32>`，并保持“256-bit 谓词寄存器 + granularity 视图”的语义，而不是退化为 lane-count vector mask。
- 两阶段 verifier 必须职责分离：
  - 第一阶段服务于 authoring legality；
  - 第二阶段服务于 emission legality。
- CLI 与 direct emission API 不得在第二阶段 contract 上出现分叉。
- v1 不实现 sync/hazard 分析，不将问题扩大为调度正确性验证。

## Goals / Non-Goals

**Goals:**

- 新增两个 VPTO legality pass，并用共享 helper 实现双阶段验证。
- 在 VPTO 类型系统中引入 `!pto.mask<b8|b16|b32>`。
- 在 `ptoas` 的 VPTO backend 中默认接线第一阶段 verifier。
- 在 emission boundary 中默认接线第二阶段 verifier。
- 将 vec scope 和 typed-mask 粒度规则形式化为 authoring legality contract。
- 将 ptr-form 边界约束形式化为 emission legality contract。

**Non-Goals:**

- 不改变 `PTOVPTOPtrBoundary` 的 canonicalization 语义，只在其后增加 legality recheck。
- 不采用 `!pto.mask<64xi1>` 这类 lane-count mask 设计。
- 不在 v1 追踪复杂的跨 region mask provenance；新方案优先依赖 typed mask 做直接校验。
- 不为 `get_buf` / `rls_buf`、`set_flag` / `wait_flag` 建立 CFG 级配对或时序分析。

## Design

### 1. 共享 legality 核心：按阶段拆分规则，而不是按调用入口拆分

本 change 采用“共享规则骨架 + 两个阶段入口”的设计，而不是让 CLI、pass 和 emitter 各自维护一套判断逻辑。

共享逻辑集中在一个新的实现单元中，建议落在：

- `lib/PTO/Transforms/PTOValidateVPTOIR.cpp`

该文件内实现三组共享 helper：

1. VPTO op / value 分类 helper

- 判断 op 是否属于需要 vec scope 约束的 VPTO 向量 / 谓词 / align 家族；
- 判断一个 SSA value 是否携带 `!pto.vreg` / `!pto.mask<...>` / `!pto.align`；
- 区分可继续出现在 vec scope 外的 config / sync / copy / pointer-building op。
- 区分 `copy family`、`buffer-like family`、`ptr-only family`，并让 emission-stage 只对 `buffer-like family` 追加“不得残留 memref-form”检查，而不是误把 pointer-only 规则重新实现一遍。

2. mask granularity helper

- 从 `!pto.mask<b8|b16|b32>` 直接抽取 granularity；
- 从向量元素类型推导期望 granularity：
  - `f32/i32 -> b32`
  - `f16/bf16/i16 -> b16`
  - 8-bit 元素家族 -> b8
- 对名字显式编码 granularity 的 predicate family，提供 family 后缀到类型参数的一致性检查。
- 对 compare family、carry family、mask-only family、predicate movement family 提供统一的“必须保持同一 `G`”检查。

3. 阶段化验证 helper

- `validateVPTOAuthoringIR(ModuleOp, llvm::raw_ostream *)`
- `validateVPTOEmissionIR(ModuleOp, llvm::raw_ostream *)`

其中第二个 helper 复用第一个 helper 的 authoring 规则，再叠加 emission-boundary ptr-form 规则。

### 2. 第一阶段：authoring legality verifier

新增 pass：

- `PTOValidateVPTOIR`
- pass 名：`pto-validate-vpto-ir`
- pass 类型：`ModuleOp`

该 pass 负责 authoring-stage 验证，规则固定为：

1. vec scope 结构约束

- 所有 VPTO 向量 / 谓词 / align 相关 op 必须位于某个带 `llvm.loop.aivector_scope` attr 的 `scf.for` 内。
- 一个需要 vec scope 的 op 不允许落在该作用域外。
- 带 `llvm.loop.aivector_scope` 的 `scf.for` 不允许嵌套另一个同样带该 attr 的 `scf.for`。
- 普通 `arith` / `scf` 标量与控制流骨架、pointer-building、copy programming、sync programming 仍允许位于 vec scope 外；这一点直接对齐 `docs/vpto-spec.md` 对 shared MLIR surface 的定义。

2. typed-mask granularity 约束

- 所有消费向量的 mask operand 都必须与对应向量元素类型匹配。
- 例如：
  - `!pto.vreg<64xf32>` / `!pto.vreg<64xi32>` 只能稳定对应 `!pto.mask<b32>`
  - `!pto.vreg<128xf16>` / `!pto.vreg<128xbf16>` / `!pto.vreg<128xi16>` 只能稳定对应 `!pto.mask<b16>`
  - 8-bit 元素家族只能稳定对应 `!pto.mask<b8>`
- `vcmp` / `vcmps` / carry 家族 / predicate movement 家族必须维持 family 自身定义的 granularity 契约。
- `vaddc` / `vsubc` / `vaddcs` / `vsubcs` 这类 carry family 中，`mask` / `carry` / `carry_in` 必须保持同一 `G`。
- `vcmp` / `vcmps` 中，输入 seed mask 与输出 result mask 必须保持同一 `G`，并与向量元素家族一致。
- `pnot` / `psel` / `ppack` / `punpack` 这类 mask-only family 不显式改变粒度，输入输出必须保持同一 `G`。
- `pdintlv_b8` / `pintlv_b16` 这类 predicate movement family 必须满足 family 自身编码的 granularity 契约。
- 函数参数、block 参数、region 传入值如果承载 mask，也必须显式带 granularity，不再默认按 opaque 放行。

3. 明确不纳入该 pass 的规则

- `get_buf` / `rls_buf` 配对；
- `set_flag` / `wait_flag` 顺序；
- `mem_bar` / `pipe_barrier` 充分性；
- pointer ABI / emission-boundary 专属规则。

### 3. 第二阶段：emission legality verifier

新增 pass：

- `PTOValidateVPTOEmissionIR`
- pass 名：`pto-validate-vpto-emission-ir`
- pass 类型：`ModuleOp`

该 pass 只面向 `PTOVPTOPtrBoundary` 之后的 ptr-form VPTO IR，规则分两层：

1. 继承 authoring-stage 规则

- 继续执行 vec scope 结构验证；
- 继续执行 typed-mask granularity 验证。

2. 新增 emission-boundary 规则

- function signature 中不得残留 memref argument 或 memref result；
- 进入 emission 的受支持 `buffer-like family` VPTO op 不得残留 memref-form 地址操作数；
- 这里的 `buffer-like family` 定义直接对齐 `docs/vpto-spec.md` / `docs/vpto-verify.md` 中的 stateless / predicate load-store 家族，而不是覆盖 stateful pointer-only family；
- canonicalization 完成后，不得残留仍参与正式 emission 链路的：
  - `pto.bind_tile`
  - 平凡 `pto.castptr`
  - `memref.subview`
  - `memref.reinterpret_cast`
  - `memref.memory_space_cast`

说明：

- `vlds_post` / `vsts_post` / `vldas` / `vldus` / `pstu` / `vstu` / `vstus` / `vstur` 等 pointer-only family 继续由现有单-op verifier 保证“不能是 memref”；
- emission-stage verifier 不重复实现这类局部签名规则，而是专注于“buffer-like family 在 emission form 中不得再保留 memref”这一模块级收口约束。

该 pass 不重新做 ptr canonicalization；它只确认 canonicalization 结果满足最终 emission contract。

### 4. pipeline 接线

#### 4.1 `ptoas` 的第一阶段接线

第一阶段 verifier 必须对两类输入都生效：

- 从 PTO lowering 到 VPTO 的正常 backend 路径；
- direct VPTO input 导致 `skipPreBackendPasses` / `skip backendPM` 的路径。

因此该 pass 不直接塞进 `addVPTOBackendMainlinePasses()`，而是在 `tools/ptoas/ptoas.cpp` 中以单独的 `validationPM` 运行：

- 放在 backendPM 完成之后；
- 若输入已经是 VPTO IR，则直接在当前 module 上运行；
- 总是早于 emission clone 和 `PTOVPTOPtrBoundary`。

这样可以保证：

- authoring-stage verifier 能看见完整的 post-mainline VPTO 形态；
- direct VPTO input 不会绕过第一阶段验证。

#### 4.2 `ptoas` 的第二阶段接线

CLI 的第二阶段 prepare contract 统一为：

1. clone module
2. `PTOVPTOPtrBoundary`
3. `PTOValidateVPTOEmissionIR`
4. `--emit-vpto` 直接消费该 clone；LLVM emission wrapper 复用同一 helper

这保证：

- `--emit-vpto` 输出的最终 ptr-form VPTO IR 已经过第二阶段 verifier；
- LLVM emission 看到的是统一的、再次确认过合法性的 IR；
- legacy `translateVPTOModuleToText` 不再作为本 change 的统一行为基线。

### 5. direct emission API 接线

当前 direct API 的主要不对齐点集中在 LLVM emission：

- LLVM emission wrapper 会 clone 并做 boundary canonicalization；
- `ptoas` CLI 也会在 emission 前做自己的 clone/prepare；
- legacy text emission wrapper 不再作为持续维护的统一目标。

本 change 将引入一个共享的 emission prepare helper，建议声明在：

- `include/PTO/Transforms/VPTOLowering.h`

建议接口语义为：

- 输入：source `ModuleOp`
- 行为：clone -> `convertVPTOEmissionBoundaryToPtr` -> `PTOValidateVPTOEmissionIR`
- 输出：可继续进入 CLI `--emit-vpto` 或 LLVM emission 的 clone module

CLI、`translateVPTOModuleToLLVMText`、`translateVPTOModuleToLLVMBitcode`
都必须复用这条流程。legacy `translateVPTOModuleToText` 不要求纳入该统一面。

第一阶段 verifier 不要求 direct emission API 自动执行；它的正式入口是 `ptoas --pto-backend=vpto` 的 backend pipeline。

### 6. 诊断策略

两阶段 verifier 都采用清晰的 `emitError` / `emitOpError` 文本诊断，不引入新的错误码体系。

诊断要求：

- 明确指出失败发生在 authoring stage 还是 emission stage；
- 明确指出违规 op 或 function；
- 对 vec scope 违规指出“在 vec scope 外”还是“nested vec scope”；
- 对 typed-mask granularity 违规指出实际 mask type 与期望 granularity / consumer vector element type；
- 对 emission-stage 违规指出残留的是 memref boundary、memref-form `buffer-like family` VPTO op，还是 dead scaffold 仍参与 emission。

## Testing Strategy

### 1. authoring-stage 回归

在 `test/phase2/` 新增或更新以下用例：

- 正向：合法 memref-first VPTO IR 通过 `pto-validate-vpto-ir`
- 正向：合法 ptr-form hand-written VPTO IR 通过 `pto-validate-vpto-ir`
- 负向：向量 / 谓词 / align op 位于 `llvm.loop.aivector_scope` 外时报错
- 负向：嵌套 `llvm.loop.aivector_scope` loop 报错
- 负向：`!pto.mask<b16>` 驱动 `!pto.vreg<64xf32>` 等 typed-mask granularity 不匹配时报错
- 负向：family 后缀与 mask 类型不一致，例如 `pset_b32` 产出非 `!pto.mask<b32>` 时报错
- 负向：`vcmp` / `vcmps` / carry family / `pnot` / `psel` / `ppack` / `punpack` / `p*dintlv*` 的 granularity 不一致时报错

### 2. pipeline 接线回归

- PTO 输入走 `--pto-backend=vpto --print-ir-after-all`，检查第一阶段 verifier 位于 backend mainline 结束之后、`PTOVPTOPtrBoundary` 之前。
- direct VPTO input 走 `--pto-backend=vpto`，验证不会绕过第一阶段 verifier。

### 3. emission-stage 回归

- 正向：合法 VPTO IR 经 `PTOVPTOPtrBoundary` 后通过 `pto-validate-vpto-emission-ir`
- 负向：刻意构造残留 memref boundary 或 memref-form emission op，确认第二阶段 verifier 拒绝
- 负向：刻意构造 stateless / predicate `buffer-like family` 在 emission form 中仍残留 memref 地址操作数，确认第二阶段 verifier 拒绝
- 正向：`--emit-vpto`、`--vpto-emit-hivm-text`、`--vpto-emit-hivm-llvm` 都经过同一 emission prepare 流程

### 4. direct API 回归

- `translateVPTOModuleToLLVMText` 与 `translateVPTOModuleToLLVMBitcode`
  都必须通过共享 prepare helper，避免 CLI 与 direct LLVM emission 在
  boundary + verify contract 上分叉。

## Risks / Mitigations

- [Risk] typed mask 方案会扩大迁移面，已有 hand-written IR、文档和测试都需要从 `!pto.mask` 迁移到 `!pto.mask<b8|b16|b32>`
  - Mitigation：把类型改造、verifier 改造、lowering/emitter 改造和文档/测试迁移作为同一 change 统一完成，避免新旧语义长期并存。

- [Risk] 两阶段 verifier 重复执行 authoring 规则，带来少量重复开销
  - Mitigation：规则实现共享，第二阶段只在 emission clone 上运行一次；这是有意的“优化后再确认”。

- [Risk] CLI 与 direct API 在第二阶段行为漂移
  - Mitigation：引入共享的 emission prepare helper，强制 CLI/text/LLVM 三条路径复用同一流程。

- [Risk] 某些 mask-only family 的 granularity 契约如果在设计上没有写清，会导致实现期分歧
  - Mitigation：在类型系统切换时同步补全 family-level contract，并为 `pnot` / `psel` / `ppack` / `punpack` / `p*dintlv*` 类 op 补回归。
