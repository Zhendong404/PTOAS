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

2. `PTOMaterializeFusionGroupsFromOpLibPass`
   - 作用：按外部 OP 库展开 fusion group
   - 建议 argument：`pto-materialize-fusion-groups-from-oplib`

3. `PTOLowLevelLoopFusionPass`
   - 作用：低层 loop 融合
   - 建议 argument：`pto-low-level-loop-fusion`

### 2.2 分组位置与内存规划关系

`PTOCreateFusionGroupsPass` 必须插在 `PlanMemory` 之前。  
同时为了满足“fusion 中间结果不分配内存”的诉求，采用结构化边界改写而非仅打 marker：

1. 分组发生在高层（tile_buf 语义）阶段，准确识别中间结果。
2. 物化阶段将 group 改写为 `func.call` 边界（或等效调用边界）。
3. 中间 buffer 不再出现在 caller IR 中，`PlanMemory` 天然不会为其分配地址。

### 2.3 为什么不采用“仅中间结果打标记跳过分配”

当前实现中，`PlanMemory` 主要处理 `memref.alloc`，最终还会通过 `AllocToPointerCast` 对未规划地址做 fallback。  
因此“仅 marker 跳过规划”不等价于“最终无内存分配”，且易产生后续一致性问题。

结论：V1 采用 `func.call` 封装 group 的方案更稳健。

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

### 4.1 组织方式

采用“目录 + 多文件”：

1. `--op-lib-dir=<path>` 指向库目录。
2. 每个 OP 可在一个或多个 `.mlir` 文件提供多个 shape/dtype 实现。

### 4.2 承载形式

每个 OP 实现以 `func.func` 模板函数承载，不使用 fused pattern 模板。

### 4.3 命名建议

建议使用统一命名以便索引（可调整但需稳定）：

`@__pto_oplib_<op>__<dtype>__<rows>x<cols>`

示例：

`@__pto_oplib_tmul__f32__32x32`

`@__pto_oplib_tdiv__f32__32x32`

`@__pto_oplib_tadd__f32__32x32`

### 4.4 签名原则

保持单 OP 语义，不引入融合特化签名。  
例如二元逐元素：

`(src0, src1, dst) -> ()`

---

## 5. 编译管线插入方案

### 5.1 V1 推荐顺序

1. `LoweringSyncToPipe`
2. `PTOCreateFusionGroupsPass`
3. `PTOViewToMemref`
4. `PTOMaterializeFusionGroupsFromOpLibPass`
5. `PTOLowLevelLoopFusionPass`
6. `PlanMemory`
7. `InsertSync`（按现有开关）
8. `EmitPTOManual -> EmitC -> C++`

说明：

1. 分组在 `PTOViewToMemref` 前，便于基于 tile_buf 语义判定中间结果。
2. 物化与 loop fusion 在 `PTOViewToMemref` 后，便于落在统一 memref/scf 层做实现。
3. `PlanMemory` 在调用边界改写之后执行，自然避开组内中间 buffer。

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

#### 阶段 B：`PTOCreateFusionGroupsPass` 之后（仅分组，不改写语义）

```mlir
// 关键变化：组内 op 被打上 group/order 元数据
pto.tmul ... outs(%t0) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 0 : i64}
pto.tdiv ... outs(%t1) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 1 : i64}
pto.tadd ... outs(%out) {pto.fusion.group_id = 0 : i64, pto.fusion.order = 2 : i64}

// 可选：中间结果标注（仅用于调试/可观测性）
%t0 = pto.alloc_tile ... {pto.fusion.role = "intermediate", pto.fusion.group_id = 0 : i64}
%t1 = pto.alloc_tile ... {pto.fusion.role = "intermediate", pto.fusion.group_id = 0 : i64}
```

#### 阶段 C：`PTOViewToMemref` + `PTOMaterializeFusionGroupsFromOpLibPass` 之后

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

该阶段完成后，`PlanMemory` 在 caller 侧只看到 `%a_buf/%b_buf/%d_buf/%out_buf` 等必要 buffer，  
不会再为原本的 group 中间结果 `%t0/%t1` 分配内存。

---

## 6. 关键实现细节

### 6.1 `PTOCreateFusionGroupsPass`

输出统一元数据（建议）：

1. `pto.fusion.group_id`：组唯一 ID
2. `pto.fusion.order`：组内顺序
3. `pto.fusion.role`：`input` / `intermediate` / `output`（可选，调试与诊断用）

### 6.2 `PTOMaterializeFusionGroupsFromOpLibPass`

核心动作：

1. 扫描 group，提取外部输入和最终输出。
2. 组内连续 OP 改写为一个 `func.call @__pto_fused_group_*`（或内部生成函数并调用）。
3. 组内中间值仅在被调用函数体内存在，不暴露到 caller。
4. 若 OP 库条目缺失：直接报错并终止（V1 策略）。

### 6.3 `PTOLowLevelLoopFusionPass`

在低层 loop 上执行保守融合：

1. 循环边界（lower/upper/step）一致才尝试融合。
2. 保持组内顺序语义不变。
3. 融合失败时可降级为“不融合但保持正确”。

---

## 7. 与 PlanMemory 的接口约定

本方案不要求 PlanMemory 感知 fusion 中间 buffer 标记。  
PlanMemory 只处理 caller 中实际存在的 alloc。

由于 group 被封装成调用边界：

1. caller 无中间 alloc。
2. PlanMemory 无需新增“跳过中间结果”的专门规则即可满足需求。

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
4. `PlanMemory` 后地址分配结果中无组内中间 buffer。
5. loop fusion 正确性与失败降级路径。
6. 端到端 `pto -> cpp` 样例回归。

---

## 10. 后续扩展路线

该命名和流程可直接扩展至：

1. reduce 类融合（例如 row/col reduction 链）。
2. expand/broadcast 类融合。
3. 混合融合（elementwise + reduce/expand），通过扩展 group 规则与库签名约束接入，无需重命名 Pass。
