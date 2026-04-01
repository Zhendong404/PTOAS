## 1. OpenSpec 契约落定

- [x] 1.1 新增 `openspec/changes/add-a5vm-post-fusion-loop-unroll/specs/tile-fusion-post-lowering-unroll/spec.md`，定义 `PTOPostFusionLoopUnroll` 仅作用于 `pto.fusion_region` 内 post-cleanup carrier loop 的正式契约。
- [x] 1.2 新增 `openspec/changes/add-a5vm-post-fusion-loop-unroll/specs/a5vm-backend-pipeline/spec.md`，把 A5 fusion cleanup 顺序更新为 `... -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll -> PTOFlattenFusionRegion -> CSE`。
- [x] 1.3 在 `proposal.md` 和 `design.md` 中明确：本 change 只锁定 `v1` 契约层，不承诺 backend emission 侧二级校正实现。

## 2. Pass 与 pipeline 接线

- [x] 2.1 在 `include/PTO/Transforms/Passes.td` 与 `include/PTO/Transforms/Passes.h` 中声明 `PTOPostFusionLoopUnroll` pass 和对应构造器。
- [x] 2.2 在 `lib/PTO/Transforms/TileFusion/` 中新增 pass 实现骨架，并把正式输入约束限定为 `pto.fusion_region` 内的 post-cleanup `scf.for + a5vm.*` carrier loop。
- [x] 2.3 在 `tools/ptoas/ptoas.cpp` 中把 `PTOPostFusionLoopUnroll` 接到 `PTOFusionLoadStoreElision` 与 `PTOFlattenFusionRegion` 之间。
- [x] 2.4 更新相关 pass summary/description 或文档注释，使 cleanup pipeline 的顺序与 OpenSpec 契约一致。

## 3. 保守 unroll 决策实现

- [x] 3.1 实现候选 loop 识别，只接受 fusion-region 内 header 已稳定、无复杂多出口控制流的 carrier loop。
- [x] 3.2 实现 `v1` cost model，正式决策域限制为 `skip / x2 / x4`。
- [x] 3.3 让 cost model 至少消费这些信号：loop body 指令数、trip count / valid shape 可证性、live value / loop-carried value 粗估、tail 比例、粗粒度寄存器压力估计。
- [x] 3.4 当收益不可证、tail 成本过高、控制流过于复杂或寄存器压力风险过大时，保持保守 no-op。

## 4. 动态 shape 与 frontier 兼容性

- [x] 4.1 为静态可整除 trip count 支持无额外 tail 的部分 unroll。
- [x] 4.2 为动态 valid shape 支持部分 unroll，并保留等价 tail 处理，不依赖“运行时恰好整除”的隐式假设。
- [x] 4.3 保证展开后不重新引入 fusion-local `vsts/vlds` round-trip，也不破坏 `PTOFusionPredicateElision` 已建立的等价复用关系。
- [x] 4.4 保持 `pto.yield` / `pto.fusion_region` result frontier 不变，确保 `PTOFlattenFusionRegion` 仍可按既有契约工作。

## 5. 回归与验证

- [x] 5.1 在 `test/tile_fusion/` 新增或更新正向用例，覆盖静态 shape 的 `x2` / `x4` unroll。
- [x] 5.2 在 `test/tile_fusion/` 新增或更新正向用例，覆盖动态 valid shape 的 partial unroll + 显式 tail 保留。
- [x] 5.3 在 `test/tile_fusion/` 新增 negative 用例，覆盖高 tail 成本、高 live range / 寄存器压力风险和复杂控制流下的保守 no-op。
- [x] 5.4 使用当前仓库等效的 `--pto-backend=vpto --print-ir-after-all` 验证 `PTOPostFusionLoopUnroll` 位于 `PTOFusionLoadStoreElision` 与 `PTOFlattenFusionRegion` 之间，并检查展开前后 loop 结构。
- [x] 5.5 运行相关 lit 回归，确认 flatten 后仍无残留 `pto.fusion_region`，且没有重新引入 fusion-local round-trip 访存。
