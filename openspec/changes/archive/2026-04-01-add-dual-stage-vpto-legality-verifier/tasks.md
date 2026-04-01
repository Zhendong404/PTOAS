## 1. OpenSpec 契约落定

- [x] 1.1 新增 `openspec/changes/add-dual-stage-vpto-legality-verifier/specs/vpto-ir-legality/spec.md`，定义 authoring-stage 和 emission-stage 两阶段 VPTO legality contract。
- [x] 1.2 在 `proposal.md` 和 `design.md` 中明确两阶段 verifier 的职责分离、pipeline 位置、共享 helper，以及 `!pto.mask<b8|b16|b32>` typed mask 方案。
- [x] 1.3 明确 spec 中哪些规则继续由现有 `VPTO.cpp` / `PTO.cpp` 单-op verifier 负责，哪些规则由新的模块级 legality verifier 负责。

## 2. typed mask 方案落地

- [x] 2.1 更新 `include/PTO/IR/VPTOTypeDefs.td` 及相关生成/实现代码，把 `!pto.mask` 升级为带参数的 `!pto.mask<b8|b16|b32>`。
- [x] 2.2 更新 `lib/PTO/IR/VPTO.cpp` 中 mask 相关 parser / printer / verifier，使其校验 granularity 参数合法且保留“256-bit 谓词寄存器视图”语义。
- [x] 2.3 更新 `include/PTO/IR/VPTOOps.td` 与相应单-op verifier，把显式 `b8/b16/b32` family 的 mask 输入输出签名收敛到 typed mask。
- [x] 2.4 更新 `lib/PTO/Transforms/PTOToVPTOLowering.cpp`、相关 bridge/cleanup/pass 路径和 emission 代码，确保所有 mask materialization / use-site 都迁移到 typed mask。
- [x] 2.5 更新 `docs/vpto-spec.md`、`docs/vpto-verify.md` 以及相关样例/说明，把文档语法从 `!pto.mask` 迁移为 `!pto.mask<b8|b16|b32>`。

## 3. legality 核心与 pass 注册

- [x] 3.1 新增 `lib/PTO/Transforms/PTOValidateVPTOIR.cpp`，实现共享的 VPTO legality helper。
- [x] 3.2 在同一实现单元中提供 `validateVPTOAuthoringIR(ModuleOp, llvm::raw_ostream *)` 与 `validateVPTOEmissionIR(ModuleOp, llvm::raw_ostream *)` 两个阶段入口。
- [x] 3.3 新增 `PTOValidateVPTOIR` pass，注册名 `pto-validate-vpto-ir`，用于 authoring-stage 验证。
- [x] 3.4 新增 `PTOValidateVPTOEmissionIR` pass，注册名 `pto-validate-vpto-emission-ir`，用于 emission-stage 验证。
- [x] 3.5 更新 `include/PTO/Transforms/Passes.h`、`include/PTO/Transforms/Passes.td` 和 `lib/PTO/Transforms/CMakeLists.txt`，完成两个新 pass 的声明、注册和编译接线。

## 4. 规则实现

- [x] 4.1 实现 vec scope 结构验证：需要 vec scope 的 VPTO 向量 / 谓词 / align op 必须位于 `scf.for` + `llvm.loop.aivector_scope` 内。
- [x] 4.2 实现 nested vec scope 拒绝规则：带 `llvm.loop.aivector_scope` 的 loop 不允许再嵌套同类 loop。
- [x] 4.3 实现 typed-mask granularity 直校验：mask type 与 consumer vector element family、carry family、compare family、mask-only family、predicate movement family 必须一致。
- [x] 4.4 实现 family 后缀和 mask type 的一致性检查，例如 `pset_b32` / `pge_b16` / `plt_b8` 的结果类型必须与后缀匹配。
- [x] 4.5 在 emission-stage verifier 中补充 ptr-form contract：按 `copy` / `buffer-like` / `ptr-only` 分类收口，拒绝残留 memref function boundary、残留 memref-form `buffer-like family` emission op，以及仍参与 emission 的 dead scaffold。

## 5. pipeline 与 emission API 接线

- [x] 5.1 修改 `tools/ptoas/ptoas.cpp`，在 backend mainline 完成后、emission clone 之前统一运行 `PTOValidateVPTOIR`，并保证 direct VPTO input 也会执行该阶段。
- [x] 5.2 修改 `tools/ptoas/ptoas.cpp` 的 emission clone 流程，在 `PTOVPTOPtrBoundary` 之后追加 `PTOValidateVPTOEmissionIR`。
- [x] 5.3 在共享 emission prepare helper 中统一封装：clone -> `convertVPTOEmissionBoundaryToPtr` -> `PTOValidateVPTOEmissionIR`，供 CLI 与 LLVM emission 复用。
- [x] 5.4 让 `translateVPTOModuleToLLVMText`、`translateVPTOModuleToLLVMBitcode` 复用该 emission prepare helper，避免 CLI 与 direct LLVM API 行为分叉；legacy `translateVPTOModuleToText` 不纳入本次统一范围。

## 6. 回归与验证

- [x] 6.1 新增 `test/phase2/` authoring-stage 正向 / 负向用例，覆盖 vec scope、nested vec scope、typed-mask mismatch、carry/compare/mask-only/predicate-movement granularity mismatch 和 family/type mismatch。
- [x] 6.2 新增 typed mask parser / printer / round-trip 用例，覆盖 `!pto.mask<b8>`、`!pto.mask<b16>`、`!pto.mask<b32>` 以及旧 `!pto.mask` 语法的拒绝行为。
- [x] 6.3 新增 `test/phase2/` emission-stage 正向 / 负向用例，覆盖 ptr-boundary 之后的最终 emission legality，以及 `buffer-like family` 残留 memref-form 地址操作数的拒绝行为。
- [x] 6.4 增加 direct VPTO input 回归，确认不经过 PTO lowering 时也会触发第一阶段 verifier。
- [x] 6.5 增加 `--emit-vpto`、`--vpto-emit-hivm-text`、`--vpto-emit-hivm-llvm` smoke，确认第二阶段 verifier 在发射前运行。
- [x] 6.6 至少运行最小相关验证命令：
  - `source scripts/ptoas_env.sh && python3 -m lit -sv test/phase2/<新增的 vpto verify 用例>`
  - `source scripts/ptoas_env.sh && build/tools/ptoas/ptoas --pto-backend=vpto --emit-vpto <case> -o /dev/null`
  - `source scripts/ptoas_env.sh && build/tools/ptoas/ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-text <case> -o /dev/null`
  - `source scripts/ptoas_env.sh && build/tools/ptoas/ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm <case> -o /dev/null`
