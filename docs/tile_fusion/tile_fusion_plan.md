# PTOAS OP Fusion V1 设计与落地方案

## 1. 背景与目标

本文档定义 PTOAS 的 OP Fusion 第一版实现方案，目标是以最小复杂度打通以下端到端链路：

1. 在 PTOAS 内部自动识别可融合 OP Group（先在 Level2 语义层完成分组）。
2. 基于外部低层 OP IR 库（每个 OP 一套实现）完成低层展开。
3. 在低层 loop 级别执行融合（loop fusion）。
4. 继续复用现有 `PTOViewToMemref -> PlanMemory -> EmitC` 主流程输出 C++。

该方案优先确保可验证性与工程收敛速度，为后续扩展到 reduce / expand 等类型融合保留接口与命名空间。

---

## 2. 已对齐设计决策

### 2.1 Pass 命名（通用化）

为支持后续扩展，Pass 命名不绑定 elementwise 语义：

1. `PTOCreateFusionGroupsPass`
   - 作用：自动分组（生成 fusion group 元数据）
   - 建议 argument：`pto-create-fusion-groups`

2. `PTOLowerToOpLibCallsPass`
   - 作用：执行 `pto op -> OP-Lib call` 的匹配、实例选择与改写
   - 建议 argument：`pto-lower-to-oplib-calls`

3. `PTOOutlineFusionGroupsPass`
   - 作用：将已改写的 group call 链 outline 为 fused helper 函数
   - 建议 argument：`pto-outline-fusion-groups`

4. `PTOLowLevelLoopFusionPass`
   - 作用：低层 loop 融合
   - 建议 argument：`pto-low-level-loop-fusion`

### 2.2 分组位置与内存规划关系（更新）

`PTOCreateFusionGroupsPass` 与后续物化/loop fusion 统一放在 `PlanMemory` 与 `InsertSync` 之后  
（`level3` 跳过这两步时，融合紧跟 `InferPTOLayout` 或 Stage2 起点），以支持所有 IR level。

1. 分组与物化发生在 memref-level（`PTOViewToMemref` 之后）。
2. 分组保持严格连续性；插入的同步等中间 op 会切断链。
3. `PlanMemory` 已完成，caller 中可能已有对中间结果的规划；融合会在 IR 层移除中间值，但不重新规划内存，V1 接受这类冗余。
4. 物化仍采用 `func.call` 边界，便于低层 loop 融合与后续扩展。

### 2.3 为什么不采用“仅中间结果打标记跳过分配”（更新）

当前 `PlanMemory` 已在融合之前完成，单靠 marker 无法回收已规划的内存，且仍可能触发后续 fallback 分配。  
结论：V1 仍采用 `func.call` 封装 group 的语义改写路径，而不是 marker skip 方案。

---

## 3. V1 作用域（首版收敛）

### 3.1 支持范围

1. 同一 BasicBlock 内的连续可融合链。
2. 首批 OP 集：`pto.tmul`, `pto.tdiv`, `pto.tadd`。
3. 链长可变（连续多个可融合 OP）。
4. 多静态 shape（rank-2，静态维度）。

### 3.2 暂不支持

1. 跨分支、跨 block、跨循环边界融合。
2. 动态 shape 融合。
3. reduce / expand 融合（保留扩展位，后续接入）。

---

## 4. 外部 OP 库规范

详细模板规范见：`docs/tile_fusion/oplib_ir_spec.md`。

### 4.1 组织方式

采用“目录 + 多文件”：

1. `--op-lib-dir=<path>` 指向库目录。
2. 每个 OP 可在一个或多个 `.mlir` 文件提供多个 shape/dtype 实现。

### 4.2 承载形式

每个 OP 实现以 `func.func` 模板函数承载，不使用 fused pattern 模板。

### 4.3 命名与匹配建议（模板化）

V1 改为“函数名表达 OP，dtype/shape 解耦”：

1. 推荐模板函数名：`@__pto_oplib_<op>_template`
2. 不再在函数名编码 `dtype/shape`
3. 编译器通过函数属性匹配模板（而非纯名字解析）

示例：

`@__pto_oplib_tmul_template`

`@__pto_oplib_tdiv_template`

`@__pto_oplib_tadd_template`

### 4.4 签名原则

保持单 OP 语义，不引入融合特化签名。  
例如二元逐元素：

`(src0, src1, dst) -> ()`

V1 推荐模板签名：

`(memref<?x?xT, #pto.address_space<vec>>, memref<?x?xT, #pto.address_space<vec>>, memref<?x?xT, #pto.address_space<vec>>) -> ()`

说明：

1. shape 使用动态 `?x?`，与具体形状解耦
2. seed template 以 `T=f32` 交付
3. `PTOLowerToOpLibCallsPass` 按目标 dtype 自动实例化（`f16/f32`）

---

## 5. 编译管线插入方案

### 5.1 V1 推荐顺序

1. `LoweringSyncToPipe`
2. `PTOViewToMemref`
3. `InferPTOLayout`（可选）
4. `PlanMemory`（`level1/level2`）
5. `InsertSync`（按现有开关，`level1/level2`）
6. `PTOCreateFusionGroupsPass`
7. `PTOLowerToOpLibCallsPass`
8. `PTOOutlineFusionGroupsPass`（仅 `--enable-op-fusion`）
9. `PTOInstantiateAndInlineOpLibPass -> Canonicalizer/CSE`
10. `PTOLowLevelLoopFusionPass`
11. `EmitPTOManual -> EmitC -> C++`

说明：

1. 融合发生在 `PTOViewToMemref` 之后，统一在 memref/scf 层实现。
2. `PlanMemory/InsertSync` 在融合前执行，保证 `level1/level2/level3` 的统一支持；同步插入可能切断融合链。
3. `PlanMemory` 已完成，融合后移除的中间结果不会被重新规划，新引入的 alloc 也不会被规划（V1 接受）。
4. 当前实现以 OP-Lib `func.call` 链为主，`PTOLowLevelLoopFusionPass` 在无显式 loop body 时可能无改写（保持保守 no-op）。

### 5.2 全流程 IR 示例（`tmul -> tdiv -> tadd`）

下面用一个简化 IR 示例说明整个 tile fusion 流程。  
为突出核心逻辑，示例省略了部分 shape/stride 常量与无关属性。

#### 阶段 A：原始输入（Level2，未融合）

```mlir
module {
  func.func @kernel(%A: !pto.ptr<f32>, %B: !pto.ptr<f32>, %D: !pto.ptr<f32>, %O: !pto.ptr<f32>) {
    %a_tv = pto.make_tensor_view %A, shape=[%c32, %c32], strides=[%c32, %c1] : !pto.tensor_view<?x?xf32>
    %b_tv = pto.make_tensor_view %B, shape=[%c32, %c32], strides=[%c32, %c1] : !pto.tensor_view<?x?xf32>
    %d_tv = pto.make_tensor_view %D, shape=[%c32, %c32], strides=[%c32, %c1] : !pto.tensor_view<?x?xf32>
    %o_tv = pto.make_tensor_view %O, shape=[%c32, %c32], strides=[%c32, %c1] : !pto.tensor_view<?x?xf32>

    %a_pt = pto.partition_view %a_tv, offsets=[%c0, %c0], sizes=[%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %b_pt = pto.partition_view %b_tv, offsets=[%c0, %c0], sizes=[%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %d_pt = pto.partition_view %d_tv, offsets=[%c0, %c0], sizes=[%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %o_pt = pto.partition_view %o_tv, offsets=[%c0, %c0], sizes=[%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %d = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %t0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0> // intermediate
    %t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0> // intermediate
    %out = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%a_pt : !pto.partition_tensor_view<32x32xf32>) outs(%a : !pto.tile_buf<...>)
    pto.tload ins(%b_pt : !pto.partition_tensor_view<32x32xf32>) outs(%b : !pto.tile_buf<...>)
    pto.tload ins(%d_pt : !pto.partition_tensor_view<32x32xf32>) outs(%d : !pto.tile_buf<...>)

    pto.tmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%t0 : !pto.tile_buf<...>)
    pto.tdiv ins(%t0, %d : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%t1 : !pto.tile_buf<...>)
    pto.tadd ins(%t1, %b : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%out : !pto.tile_buf<...>)

    pto.tstore ins(%out : !pto.tile_buf<...>) outs(%o_pt : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
```

#### 阶段 B：`PlanMemory/InsertSync` 之后 + `PTOCreateFusionGroupsPass` 之后（仅分组，不改写语义）

```mlir
// 关键变化：组内 op 被打上 group/order 元数据
pto.tmul ... outs(%t0) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 0 : i64}
pto.tdiv ... outs(%t1) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 1 : i64}
pto.tadd ... outs(%out) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 2 : i64}

// 可选：中间结果标注（仅用于调试/可观测性）
%t0 = pto.alloc_tile ... {pto.fusion.role = "intermediate", pto.fusion.group_id = 0 : i64}
%t1 = pto.alloc_tile ... {pto.fusion.role = "intermediate", pto.fusion.group_id = 0 : i64}
```

#### 阶段 C：`PlanMemory/InsertSync` 之后 + `PTOLowerToOpLibCallsPass/PTOOutlineFusionGroupsPass` 之后

Caller 被改写成一个 group 调用边界，中间结果不再出现在 caller 中：

```mlir
func.func @kernel(%A: !pto.ptr<f32>, %B: !pto.ptr<f32>, %D: !pto.ptr<f32>, %O: !pto.ptr<f32>) {
  // ... tload 仍在 caller ...
  // %t0 / %t1 对应的 alloc 已从 caller 消失
  func.call @__pto_fused_group_0(%a_buf, %b_buf, %d_buf, %out_buf)
    : (memref<32x32xf32, #pto.address_space<vec>>,
       memref<32x32xf32, #pto.address_space<vec>>,
       memref<32x32xf32, #pto.address_space<vec>>,
       memref<32x32xf32, #pto.address_space<vec>>) -> ()
  // ... tstore ...
  return
}
```

生成的融合函数（示意）在 low-level 层展开组内 OP 语义：

```mlir
func.func @__pto_fused_group_0(%a: memref<32x32xf32, #pto.address_space<vec>>,
                               %b: memref<32x32xf32, #pto.address_space<vec>>,
                               %d: memref<32x32xf32, #pto.address_space<vec>>,
                               %out: memref<32x32xf32, #pto.address_space<vec>>) {
  %tmp0 = memref.alloc() : memref<32x32xf32, #pto.address_space<vec>>
  %tmp1 = memref.alloc() : memref<32x32xf32, #pto.address_space<vec>>

  // tmul low-level loop
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      %va = memref.load %a[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vb = memref.load %b[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vm = arith.mulf %va, %vb : f32
      memref.store %vm, %tmp0[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
    }
  }

  // tdiv low-level loop
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      %vt0 = memref.load %tmp0[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vd  = memref.load %d[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vv  = arith.divf %vt0, %vd : f32
      memref.store %vv, %tmp1[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
    }
  }

  // tadd low-level loop
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      %vt1 = memref.load %tmp1[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vb  = memref.load %b[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vo  = arith.addf %vt1, %vb : f32
      memref.store %vo, %out[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
    }
  }
  return
}
```

#### 阶段 D：`PTOLowLevelLoopFusionPass` 之后（融合到单一 loop nest）

```mlir
func.func @__pto_fused_group_0(%a: memref<32x32xf32, #pto.address_space<vec>>,
                               %b: memref<32x32xf32, #pto.address_space<vec>>,
                               %d: memref<32x32xf32, #pto.address_space<vec>>,
                               %out: memref<32x32xf32, #pto.address_space<vec>>) {
  // 低层 loop 融合后，不再需要 %tmp0/%tmp1
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c32 step %c1 {
      %va = memref.load %a[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vb = memref.load %b[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vd = memref.load %d[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
      %vm = arith.mulf %va, %vb : f32
      %vv = arith.divf %vm, %vd : f32
      %vo = arith.addf %vv, %vb : f32
      memref.store %vo, %out[%i, %j] : memref<32x32xf32, #pto.address_space<vec>>
    }
  }
  return
}
```

该阶段完成后，caller IR 中仅保留 `%a_buf/%b_buf/%d_buf/%out_buf` 等必要 buffer，  
但 `PlanMemory` 已在此前执行，可能已对 `%t0/%t1` 做过规划，V1 不做回收与重规划。

---

## 6. 关键实现细节

### 6.1 `PTOCreateFusionGroupsPass`

输出统一元数据（建议）：

1. `pto.fusion.group_id`：组唯一 ID
2. `pto.fusion.order`：组内顺序
3. `pto.fusion.role`：`input` / `intermediate` / `output`（可选，调试与诊断用）

### 6.2 `PTOLowerToOpLibCallsPass` + `PTOOutlineFusionGroupsPass`

核心动作：

1. `PTOLowerToOpLibCallsPass`：读取 OP-Lib 模板并完成匹配/选择，将组内或非组单 OP 改写为 OP-Lib `func.call`。
2. `PTOOutlineFusionGroupsPass`：仅对带 `group_id/order` 的 call 链做 outline，生成 `@__pto_fused_group_*` 调用边界。
3. 组内中间值由 outlined helper 管理，caller 仅保留调用边界参数。
4. 若 OP 库条目缺失：按策略执行 warning/fallback 或报错终止。

### 6.3 `PTOLowLevelLoopFusionPass`

在低层 loop 上执行保守融合：

1. 循环边界（lower/upper/step）一致才尝试融合。
2. 保持组内顺序语义不变。
3. 融合失败时可降级为“不融合但保持正确”。

---

## 7. 与 PlanMemory 的接口约定

PlanMemory 在融合之前执行，不要求其感知 fusion 中间 buffer 标记。

1. caller 侧可能已对中间结果完成规划；融合后这些中间值被移除，但不重新规划。
2. 融合函数体内新增的 `memref.alloc` 不会被 PlanMemory 处理（V1 接受）。
3. 若后续需要消除这类冗余或补规划，可考虑在融合后再执行一次 PlanMemory。

---

## 8. 错误处理与可观测性

### 8.1 错误策略

1. 分组失败：不报错，按未融合路径继续。
2. group 已建立但 OP 库缺失：报错并终止（防止 silent fallback）。
3. loop fusion 失败：降级为“仅完成物化，不做 loop fusion”。

### 8.2 调试开关建议

新增调试参数（建议）：

1. `--enable-op-fusion`
2. `--op-lib-dir=<path>`
3. `--op-fusion-debug`

`--op-fusion-debug` 建议输出：

1. 识别到的 group 与边界。
2. 每个 OP 的库匹配 key。
3. 物化前后 IR 摘要。
4. loop fusion 成功/失败统计。

---

## 9. 测试计划（V1）

1. 分组正确性（连续链 / 非连续链 / block 边界）。
2. 库匹配与缺失报错。
3. 调用边界改写后 caller 中间 alloc 消失验证。
4. `PlanMemory` 在融合前执行，融合后 caller 中间 alloc 消失但不重规划（允许规划冗余）。
5. loop fusion 正确性与失败降级路径。
6. 端到端 `pto -> cpp` 样例回归。

---

## 10. 后续扩展路线

该命名和流程可直接扩展至：

1. reduce 类融合（例如 row/col reduction 链）。
2. expand/broadcast 类融合。
3. 混合融合（elementwise + reduce/expand），通过扩展 group 规则与库签名约束接入，无需重命名 Pass。
