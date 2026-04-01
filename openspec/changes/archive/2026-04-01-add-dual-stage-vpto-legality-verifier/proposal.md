# Proposal: 为 VPTO backend 引入双阶段 IR legality verifier

## Why

当前 VPTO backend 已经形成了较清晰的主线：前段 pass 将 PTO lowering 到 VPTO，随后经过 `PTOVPTOIfCanonicalize`、`PTOLowLevelLoopFusion`、`PTOFusionPredicateElision`、`PTOFusionLoadStoreElision`、`PTOFlattenFusionRegion` 和 `PTOVPTOExpandBridgeOps` 等阶段，最后在 emission boundary 通过 `PTOVPTOPtrBoundary` 将 IR 规整到 ptr 形态并进入 text / LLVM 发射。

但这条主线目前缺少面向 VPTO IR 本身的模块级合法性验证。现有 `lib/PTO/IR/VPTO.cpp` 和 `lib/PTO/IR/PTO.cpp` 里的 verifier 主要覆盖单个 op 的签名、类型、地址空间和少量枚举属性约束，无法系统捕获以下两类问题：

- hand-written VPTO IR 或库开发者编写的 memref-first VPTO IR 是否满足 vec scope、mask 使用等跨 op 结构约束；
- VPTO lowering 和后续优化之后，IR 在进入 emission boundary 乃至 ptr canonicalization 之后是否仍满足最终发射契约。

没有双阶段 legality verifier 时，很多错误要到 text/LLVM 发射，甚至更下游阶段才暴露，诊断距离源码太远；同时 typed mask granularity 也无法在类型层直接表达并参与 fail-fast 检查。

## What Changes

本 change 计划为 `--pto-backend=vpto` 引入双阶段 legality verifier：第一阶段在 VPTO lowering / 优化主线结束后验证 authoring-form VPTO IR；第二阶段在 `PTOVPTOPtrBoundary` 之后验证 ptr-form emission IR。同时，本 change 将把 VPTO mask 类型从完全 opaque 的 `!pto.mask` 升级为带 granularity 的 `!pto.mask<b8|b16|b32>`，让 mask 合法性能够在类型层直接表达并被 verifier 直接检查。

## 概述

当前 VPTO backend 已经形成了较清晰的主线：前段 pass 将 PTO lowering 到 VPTO，随后经过 `PTOVPTOIfCanonicalize`、`PTOLowLevelLoopFusion`、`PTOFusionPredicateElision`、`PTOFusionLoadStoreElision`、`PTOFlattenFusionRegion` 和 `PTOVPTOExpandBridgeOps` 等阶段，最后在 emission boundary 通过 `PTOVPTOPtrBoundary` 将 IR 规整到 ptr 形态并进入 text / LLVM 发射。

但这条主线目前缺少面向 VPTO IR 本身的模块级合法性验证。现有 `lib/PTO/IR/VPTO.cpp` 和 `lib/PTO/IR/PTO.cpp` 里的 verifier 主要覆盖单个 op 的签名、类型、地址空间和少量枚举属性约束，无法系统捕获以下两类问题：

- hand-written VPTO IR 或库开发者编写的 memref-first VPTO IR 是否满足 vec scope、mask 使用等跨 op 结构约束；
- VPTO lowering 和后续优化之后，IR 在进入 emission boundary 乃至 ptr canonicalization 之后是否仍满足最终发射契约。

本 change 计划为 `--pto-backend=vpto` 引入双阶段 legality verifier：第一阶段在 VPTO lowering / 优化主线结束后验证 authoring-form VPTO IR；第二阶段在 `PTOVPTOPtrBoundary` 之后验证 ptr-form emission IR。同时，本 change 将把 VPTO mask 类型从完全 opaque 的 `!pto.mask` 升级为带 granularity 的 `!pto.mask<b8|b16|b32>`，让 mask 合法性能够在类型层直接表达并被 verifier 直接检查。

本 change 的 OpenSpec 契约将显式对齐完整版 [`docs/vpto-spec.md`](/home/zhangzhendong/ptoas-workspace/PTOAS/docs/vpto-spec.md) 中已经写明的 VPTO authoring surface：

- `llvm.loop.aivector_scope` 只约束产生或消费 `!pto.vreg` / `!pto.mask<...>` / `!pto.align` 的 VPTO op，不约束普通 `arith` / `scf` 标量与控制流骨架；
- mask granularity 规则不只覆盖“向量 consumer vs mask”这一类，还覆盖 compare family、carry family、mask-only family、predicate movement family；
- 地址形态按 `copy family`、`buffer-like family`、`ptr-only family` 分层，其中 stateless / predicate 的 `buf_like` authoring form 允许 memref 或 ptr，而 stateful `%base/%base_out` family 继续保持 pointer-only。

## 背景与动机

当前实现有四个直接痛点：

1. 单-op verifier 覆盖不到作者态 IR 的结构契约

- `VRegType::verify`、`VldsOp::verify`、`VstsOp::verify`、`Copy*Op::verify` 等已经能检查类型和地址空间，但不会验证向量/谓词 op 是否处于 `llvm.loop.aivector_scope` 内，也无法在模块层系统确认 mask granularity 与 consumer vector family 是否一致。

2. hand-written IR 和 lowering 产物的错误暴露过晚

- 库开发者主要接触 memref-first VPTO IR。
- 外部客户可能直接编写 ptr-form VPTO IR 做极限优化。
- 没有模块级 verifier 时，很多错误要到 text/LLVM 发射，甚至更下游阶段才暴露，诊断距离源码太远。

3. 后续优化仍可能破坏 VPTO legality

- `PTOLowLevelLoopFusion`、predicate/load-store cleanup、bridge expansion 等阶段都会继续改变 VPTO 结构。
- 即使 lowering 初始产物合法，优化后仍可能留下非法的 vec scope、mask 使用或 emission-boundary 形态。

4. emission 路径的最终契约缺少显式“再次确认”

- 当前 CLI 路径在 clone 上运行 `PTOVPTOPtrBoundary`，LLVM emission wrapper 也会在 clone 上做 pointer canonicalization。
- 但 canonicalization 之后还缺一个明确的 ptr-form legality check，来确认“优化后是否合法、boundary 之后是否合法、CLI 与 direct API 是否一致”。

## 目标

- 为 VPTO backend 引入两个职责明确的 legality verifier：
  - authoring-stage verifier：验证 memref-first / authoring-form VPTO IR；
  - emission-stage verifier：验证 `PTOVPTOPtrBoundary` 之后的 ptr-form VPTO IR。
- 明确第一阶段默认接到 `--pto-backend=vpto` 主线中，对 lowering 产物和 direct VPTO input 都生效。
- 明确第二阶段默认接到 emission boundary 中，对 `--emit-vpto`、text emission、LLVM emission 都生效。
- 让两阶段 verifier 共享同一套 VPTO legality 规则骨架，避免 CLI、pass 和 direct emission API 之间漂移。
- 在 VPTO 类型系统中引入 `!pto.mask<b8|b16|b32>`，用 granularity 而不是 lane 数表达谓词寄存器视图。
- 通过 OpenSpec 明确：
  - 哪些规则由已有 op/type verifier 负责；
  - 哪些规则由新的 legality verifier 负责；
  - 两阶段 verifier 的输入 IR 形态和 pipeline 位置。
  - 这些规则与 `docs/vpto-spec.md` 中的 vec scope、typed mask、`buf_like` / pointer-only 定义保持一致。

## 非目标

- 不采用 `!pto.mask<64xi1>` 这类 lane-count vector mask 设计；新方案使用 `!pto.mask<b8|b16|b32>` 保留“256-bit 谓词寄存器 + granularity 视图”的语义。
- 不在 v1 中实现 `get_buf` / `rls_buf` 的 acquire-release 配对验证。
- 不在 v1 中实现 `set_flag` / `wait_flag`、`pipe_barrier`、`mem_bar` 的时序充分性、hazard 正确性或 UB alias 分析。
- 不替代现有 `VPTO.cpp` / `PTO.cpp` 的单-op verifier，也不把已覆盖的地址空间、typed `!pto.ptr`、ptr-only family 规则搬到模块级 verifier 里重复实现。
- 不改变现有 VPTO lowering、优化或 emission 的功能目标；本 change 主要增加 typed mask 约束、fail-fast legality check 和相应的 pipeline 接线。
- 不重写 `docs/vpto-spec.md` 中已有的 op 语义定义；本 change 只把这些既有定义转译成 OpenSpec 下可实施、可测试的 legality contract。

## 预期结果

- `--pto-backend=vpto` 的主线会在进入 emission boundary 前，先对 authoring-form VPTO IR 做一次 fail-fast legality 验证。
- `PTOVPTOPtrBoundary` 之后会再对 ptr-form VPTO IR 做一次 emission legality 验证，确认优化后和 boundary canonicalization 后的 IR 仍满足最终发射契约。
- memref-first authoring IR 中的 vec scope、typed mask 粒度不匹配等错误，可以在更贴近源码的位置暴露。
- stateless / predicate `buf_like` family 与 stateful pointer-only family 的边界，会在 OpenSpec 与实现中使用同一套分类口径，不再由实现细节隐式决定。
- CLI 路径与 direct text / LLVM emission API 将共享一致的 emission-stage legality contract，不再依赖“调用者自己先保证正确”的隐含假设。

## 成功标准

- 新增 OpenSpec change `add-dual-stage-vpto-legality-verifier`，包含 `proposal.md`、`design.md`、`tasks.md` 和 `specs/vpto-ir-legality/spec.md`。
- `vpto-ir-legality` capability 明确定义：
  - authoring-stage verifier 的输入契约、规则边界和 pipeline 位置；
  - emission-stage verifier 的输入契约、规则边界和 pipeline 位置；
  - direct VPTO input 与 direct emission API 的适用路径。
- 任务拆解明确到可直接实施的粒度，包括 pass 注册、helper 复用、pipeline 接线和测试覆盖。
- 预期测试矩阵能显式覆盖：
  - authoring-form 正向 / 负向样例；
  - ptr-form emission 正向 / 负向样例；
  - direct VPTO input；
  - text / LLVM emission 路径。
