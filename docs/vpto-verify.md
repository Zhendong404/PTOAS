# VPTO IR 合法性验证规则

## 1. 目的与适用范围

本文档给出 VPTO IR 的合法性验证基线，用来回答三个问题：

- 哪些规则已经由现有 MLIR / ODS / 单-op verifier 覆盖；
- 哪些规则应由新增的 VPTO legality verify pass 覆盖；
- 哪些规则暂时只记录为约束，但不在 v1 中实现。

本文档讨论的是静态 IR 合法性，不等同于：

- 运行时正确性；
- 性能最优性；
- sync / hazard 是否“足够”；
- NPU 板端行为是否与预期完全一致。

本文档中的“VPTO IR”包含：

- `pto` dialect 中的 VPTO 低层 op；
- 与其共同出现的 `scf`、`arith`、`memref` 等共享 MLIR dialect；
- VPTO backend 主线中的 memref-first authoring form；
- `PTOVPTOPtrBoundary` 之后的 ptr-form emission form。

## 2. 验证分层

当前和计划中的 VPTO 合法性验证分成四层：

| 层级 | 主要责任 | 当前落点 |
|---|---|---|
| MLIR 基础 verifier | SSA、region、block、terminator、函数签名、类型一致性等通用结构规则 | MLIR 自身 |
| VPTO/PTO 单-op verifier | 单个 op 的签名、类型、地址空间、局部 attr / token / mode 合法性 | `lib/PTO/IR/VPTO.cpp`、`lib/PTO/IR/PTO.cpp` |
| authoring-stage verifier | 面向库开发者和 hand-written IR 的模块级结构规则 | 计划新增 `pto-validate-vpto-ir` |
| emission-stage verifier | 面向最终发射前 ptr-form IR 的模块级收口规则 | 计划新增 `pto-validate-vpto-emission-ir` |

为了避免概念混淆，本文档统一使用下面两个阶段名：

- authoring form
  指 backend mainline 结束后、`PTOVPTOPtrBoundary` 之前的 VPTO IR。该形态允许 memref-first 地址表达，也允许直接书写 ptr-form VPTO IR。
- emission form
  指 `PTOVPTOPtrBoundary` 之后、进入 `--emit-vpto`、text emission、LLVM emission 之前的 ptr-form VPTO IR。

## 3. 规则总表

### 3.1 MLIR 通用结构规则

这些规则不由 VPTO 自己重新定义，而是直接依赖 MLIR verifier：

- 函数、block、region 必须满足 MLIR 的基本结构要求。
- SSA use 必须被正确支配。
- `scf.for`、`scf.if` 等共享 dialect op 的 region 参数、yield 和结果类型必须一致。
- op 的操作数个数、结果个数和 ODS 声明必须一致。

这部分可以视为 VPTO legality 的最低层基线，不在本文后续重复展开。

### 3.2 类型与签名规则

#### 3.2.1 `!pto.vreg`

责任方：已有，`lib/PTO/IR/VPTO.cpp`

- `!pto.vreg<NxT>` 的 `N` 必须为正数。
- `T` 必须是整数或浮点元素类型。
- `N * bitwidth(T)` 必须恰好等于 2048 bit，也就是 256 byte。

这意味着：

- `!pto.vreg<64xf32>`、`!pto.vreg<128xf16>`、`!pto.vreg<256xi8>` 是合法形态；
- 总位宽不是 2048 bit 的向量类型都不合法。

#### 3.2.2 `!pto.mask<G>`

责任方：类型系统 + 单-op verifier + legality verifier

- `!pto.mask<G>` 是专用谓词寄存器类型，不是整数向量类型。
- `G` 必须是 `b8`、`b16`、`b32` 之一。
- `G` 表达的是谓词寄存器的 granularity 视图，不是当前激活 lane 数。

这意味着：

- `!pto.mask<b32>` 对应 `f32/i32` 一类 32-bit 元素视图；
- `!pto.mask<b16>` 对应 `f16/bf16/i16` 一类 16-bit 元素视图；
- `!pto.mask<b8>` 对应 8-bit 元素视图；
- `PAT_VL32` 这类“激活前 N 个 lane”的信息属于 value / attr 层语义，不属于类型参数。

因此，新规则不再依赖“从 producer 反推 granularity”作为主路径，而是优先使用 mask type 本身做直接合法性校验。

后续各 ISA / 示例文档里如果使用抽象记法，会写成 `!pto.mask<G>`；如果 op
family 自身已经显式编码粒度，例如 `pset_b32`、`plt_b16`、`pdintlv_b8`，则
示例必须写成对应的具体 typed mask，而不是回退到旧的 `!pto.mask`。

#### 3.2.3 `!pto.align`

责任方：已有，`lib/PTO/IR/VPTO.cpp`

- `!pto.align` 只能用在 align carrier 相关 op 中；
- 不能用普通标量或向量类型替代。

#### 3.2.4 单-op 向量签名一致性

责任方：已有，`lib/PTO/IR/VPTO.cpp`

多数 VPTO 向量 op 已在单-op verifier 中检查以下规则：

- 同一 op 的输入 / 输出 `!pto.vreg` 形状必须匹配；
- 需要 `!pto.mask<G>` 的位置必须真的是某种 `!pto.mask<...>`；
- 需要 `!pto.align` 的位置必须真的是 `!pto.align`；
- vec-scalar 类 op 的标量类型必须与向量元素类型匹配；
- pair-result / carry-result / lane-select 类 op 的多个结果必须满足家族内部的一致性约束。

示例：

- `vadd` / `vsub` / `vmul` / `vdiv` / `vand` / `vor` / `vxor`：左右输入和结果必须共享同一 `!pto.vreg` 类型；
- `vabs` / `vexp` / `vln` / `vsqrt` / `vrec` / `vrelu` / `vnot`：输入和结果必须共享同一 `!pto.vreg` 类型；
- `vaddc` / `vsubc` / `vaddcs` / `vsubcs`：除向量类型匹配外，还要求 carry 相关结果和输入是同 granularity 的 `!pto.mask<G>`。

### 3.3 地址形态与地址空间规则

### 3.3.1 基本原则

责任方：已有单-op verifier + emission-stage verifier

VPTO 中的地址规则不是“所有访存 op 都接受同一种指针形态”，而是分三层：

- copy family
  只接受 typed `!pto.ptr`，并要求 GM / UB 方向匹配。
- buffer-like family
  主要接受 memref 或 `!pto.ptr`，用于 memref-first authoring IR。
- ptr-only family
  只接受 pointer buffer，不接受 memref authoring。

这里的“buffer-like”在当前实现中主要指：

- `memref<...>`
- `!pto.ptr<...>`

少数 ptr-only helper 会兼容底层 pointer buffer 表示，但这不改变 authoring 规则的主旨：不是所有 VPTO op 都允许 memref，也不是所有 VPTO op 都必须是 ptr。

### 3.3.2 copy family

责任方：已有，`lib/PTO/IR/VPTO.cpp`

`copy_gm_to_ubuf` / `copy_ubuf_to_gm` 的规则包括：

- source 和 destination 必须都是 typed `!pto.ptr`；
- GM -> UB 或 UB -> GM 的方向必须匹配；
- source 和 destination 的元素字节宽度必须已知；
- source 和 destination 的元素字节宽度必须一致。

`copy_ubuf_to_ubuf` 的规则包括：

- source 和 destination 必须是 UB-backed buffer-like 值；
- 不能直接拿 GM-backed buffer 作为该 family 的输入。

### 3.3.3 buffer-like load / store family

责任方：已有单-op verifier；emission-stage 再做 ptr-form 收口

当前以下家族按“buffer-like”处理：

- load / predicate load：
  - `vlds`
  - `uvld`
  - `plds`
  - `pld`
  - `pldi`
  - `vsld`
  - `vldx2`
  - `vsldb`
  - `vgather2`
  - `vgatherb`
  - `vgather2_bc`
- store / predicate store：
  - `vsts`
  - `pst`
  - `psti`
  - `psts`
  - `vsst`
  - `vstx2`
  - `vsstb`
  - `vsta`
  - `vstas`
  - `vstar`
  - `vscatter`
- 其他 UB buffer-like family：
  - `vbitsort`
  - `vmrgsort4`

这类 op 的共同规则是：

- 只能访问 UB-backed buffer，不能直接访问 GM；
- 对于要求 offset 的家族，offset 必须是 `index` 或该家族规定的标量类型；
- 若当前仍处于 authoring form，可以接受 memref-first 地址表达；
- 一旦进入 emission form，受支持的 family 必须被 `PTOVPTOPtrBoundary` 收敛为 ptr-form。

### 3.3.4 ptr-only family

责任方：已有单-op verifier

当前以下家族属于 ptr-only：

- `vlds_post`
- `vsts_post`
- `vldas`
- `vldus`
- `pstu`
- `vstu`
- `vstus`
- `vstur`

这类 op 的共同规则是：

- base/source/destination 必须是 pointer buffer；
- 不接受 memref authoring；
- post-update 类结果的类型必须和输入 base/source/destination 保持一致。

这部分不属于 emission-stage verifier 新增的语义，而是现有单-op verifier 已经负责的局部约束。

### 3.3.5 emission form 的额外地址规则

责任方：计划新增，emission-stage verifier

进入 emission form 后，除了继续满足单-op verifier 外，还必须满足：

- function signature 中不得残留 memref argument；
- function signature 中不得残留 memref result；
- 受支持的 buffer-like VPTO op 不得残留 memref-form 地址操作数；
- 用于桥接 memref-first IR 的中间 scaffold 不能继续挂在正式 emission 链路上。

这条规则是第二阶段 verifier 的核心职责之一。

### 3.4 局部 attr / token / mode 规则

责任方：已有，`lib/PTO/IR/VPTO.cpp` 与 `lib/PTO/IR/PTO.cpp`

这一类规则当前大多已经由单-op verifier 覆盖，后续 legality pass 不需要重复实现。

#### 3.4.1 predicate pattern

- `pset_b8` / `pset_b16` / `pset_b32`
- `pge_b8` / `pge_b16` / `pge_b32`

必须使用受支持的 `PAT_*` token，例如：

- `PAT_ALL`
- `PAT_VL1`
- `PAT_VL2`
- `PAT_VL4`
- `PAT_VL8`
- `PAT_VL16`
- `PAT_VL32`
- `PAT_VL64`
- `PAT_VL128`
- `PAT_M3`
- `PAT_M4`
- `PAT_H`
- `PAT_Q`
- `PAT_ALLF`

#### 3.4.2 distribution / stride / part / mode / compare token

已有 verifier 会分别检查不同家族的 token 合法性，例如：

- predicate load dist：
  - `NORM`
  - `US`
  - `DS`
- predicate store dist：
  - `NORM`
  - `PK`
- `vlds` 的 dist：
  - `NORM`
  - `BLK`
  - `DINTLV_B32`
  - `UNPK_B16`
- `vldx2` 的 dist：
  - `DINTLV_B8`
  - `DINTLV_B16`
  - `DINTLV_B32`
  - `BDINTLV`
- `vstx2` 的 dist：
  - `INTLV_B8`
  - `INTLV_B16`
  - `INTLV_B32`
- stride token：
  - `STRIDE_S3_B16`
  - `STRIDE_S4_B64`
  - `STRIDE_S8_B32`
  - `STRIDE_S2_B64`
  - `STRIDE_VSST_S8_B16`
- part token：
  - `LOWER`
  - `HIGHER`
- post-update mode：
  - `POST_UPDATE`
  - `NO_POST_UPDATE`
- compare mode：
  - `eq`
  - `ne`
  - `lt`
  - `le`
  - `gt`
  - `ge`

#### 3.4.3 sync / pipe / event token

责任方：已有，主要由 PTO op 定义和 parser / verifier 负责

`set_flag` / `wait_flag` / `get_buf` / `rls_buf` 相关规则与其说是“常量表达式验证”，不如说是“token / attr 形态验证”：

- `set_flag` / `wait_flag` 使用的是合法的 `PIPE_*` / `EVENT_ID*` attr；
- `get_buf` / `rls_buf` 的 `op_type` 必须能映射到具体 pipe；
- `buf_id` 必须在合法范围内；
- `mode` 必须满足当前 PTO verifier 对非负整数的要求。

因此，原先“特定参数必须是常量表达式”的说法过于宽泛，建议以后统一写成“token / attr 合法性验证”。

### 3.5 元素类型家族规则

责任方：已有，`lib/PTO/IR/VPTO.cpp`

除了“是不是 `!pto.vreg` / `!pto.mask<...>` / `!pto.align`”之外，很多 op 还有更细的元素类型家族约束：

- `vci` 的 result element type 必须是整数，index 标量类型必须与 result element type 一致；
- `vshl` / `vshr` / `vshls` / `vshrs` 要求整数向量和整数标量；
- `vbcnt` / `vcls` 要求整数向量元素；
- carry 家族当前要求 32-bit 整数向量；
- `vgatherb` / `vgather2_bc` 的 offset vector 当前要求 32-bit 整数元素；
- `vselr` / `vselrv2` 要求 selector 向量元素宽度与主数据向量元素宽度匹配；
- `vcvt` 额外受限于实现中接受的转换组合。

这部分属于 op family 自己的局部语义规则，不应交给模块级 legality pass 兜底。

### 3.6 mask / predicate 规则

### 3.6.1 已有局部规则

责任方：已有，`lib/PTO/IR/VPTO.cpp`

当前已有 verifier 已经覆盖：

- 需要 `!pto.mask<G>` 的位置必须真的是某种 `!pto.mask<...>`；
- predicate producer 的 pattern token 必须合法；
- compare / carry / predicate movement 家族的 mask 输入输出位置必须符合家族签名；
- 对名字显式编码 granularity 的 family，输入输出 mask type 必须和家族后缀一致，例如 `pset_b32 -> !pto.mask<b32>`、`pge_b16 -> !pto.mask<b16>`。

### 3.6.2 authoring-stage 的新增规则

责任方：计划新增，`pto-validate-vpto-ir`

authoring-stage verifier 改为以 typed mask 为主做强校验：

- 所有消费向量的 mask operand 都必须与向量元素家族匹配；
- `f32/i32` 对应 `!pto.mask<b32>`；
- `f16/bf16/i16` 对应 `!pto.mask<b16>`；
- 8-bit 元素家族对应 `!pto.mask<b8>`；
- `vcmp` / `vcmps` 这类 compare producer 的输入 seed mask、输出 result mask 与输入向量元素家族必须一致；
- `vaddc` / `vsubc` / `vaddcs` / `vsubcs` 的 carry mask 与主 mask 必须保持同一 granularity；
- `pnot` / `psel` / `ppack` / `punpack` 这类不显式改变粒度的 mask-only op，输入输出必须保持同一 `G`。

### 3.6.3 新方案下的边界与收益

责任方：类型系统 + authoring-stage verifier

引入 `!pto.mask<G>` 之后：

- 不再把函数参数、block 参数或 region 传入值默认视为 opaque mask 放行；
- 这些位置如果承载 mask，也必须显式写成 `!pto.mask<b8>` / `!pto.mask<b16>` / `!pto.mask<b32>`；
- legality pass 的职责从“复杂 provenance 反推”收敛为“检查类型和家族约束是否一致”。

### 3.7 VecScope 与上下文规则

### 3.7.1 authoring-stage 的主规则

责任方：计划新增，`pto-validate-vpto-ir`

所有 VPTO 向量 / 谓词 / align 相关 op 必须处于某个带 `llvm.loop.aivector_scope` 的 `scf.for` 作用域内。

这里的“向量 / 谓词 / align 相关 op”包括：

- 产生或消费 `!pto.vreg` 的 VPTO op；
- 产生或消费 `!pto.mask<...>` 的 VPTO op；
- 产生或消费 `!pto.align` 的 VPTO op。

对应地，以下类别可以位于 vec scope 外：

- 纯标量 `arith` 运算；
- `scf.for` / `scf.if` 等控制流骨架；
- pointer construction / pointer arithmetic；
- copy programming / loop programming；
- sync / pipe / event / buffer-id 相关 op；
- 其他不直接产生或消费 `!pto.vreg` / `!pto.mask<...>` / `!pto.align` 的辅助 op。

### 3.7.2 nested vec scope 禁止

责任方：计划新增，`pto-validate-vpto-ir`

一个带 `llvm.loop.aivector_scope` 的 carrier loop 内不允许再嵌套另一个同样带该 attr 的 carrier loop。

这一条的目标不是限制普通 `scf.for` 嵌套，而是保持 VPTO vector function 作用域边界唯一且明确。

### 3.7.3 当前不纳入实现的函数级规则

责任方：延后

以下规则在概念上可能合理，但当前不在 v1 范围：

- SIMD 函数只能调用 SIMD callee；
- SIMD 函数只能由特定入口或特定函数属性调用；
- global / kernel 函数必须带 aicore / host 属性且二者互斥；
- aicore 函数不能递归调用。

这些规则涉及更上层的函数属性和 call graph 约束，不适合作为第一版 VPTO legality pass 的交付范围。

### 3.8 emission-boundary 规则

责任方：计划新增，`pto-validate-vpto-emission-ir`

第二阶段 verifier 在 `PTOVPTOPtrBoundary` 之后运行，除了继承 authoring-stage 的 vec scope 和 typed-mask 规则外，还负责确认最终发射契约。

需要额外检查的内容包括：

- function boundary 中不得残留 memref argument；
- function boundary 中不得残留 memref result；
- 受支持的 buffer-like VPTO op 不得残留 memref-form 地址操作数；
- `PTOVPTOPtrBoundary` 之后，不得残留仍参与正式 emission 链路的：
  - `pto.bind_tile`
  - 平凡 `pto.castptr`
  - `memref.subview`
  - `memref.reinterpret_cast`
  - `memref.memory_space_cast`

这部分的核心语义是：第一阶段验证“作者写得是否合法”，第二阶段验证“优化完、canonicalize 完之后是否合法”。

## 4. 规则归属总结

如果只想快速判断某条规则该由谁负责，可以按下面理解：

- “是不是合法 MLIR”
  交给 MLIR verifier。
- “这个单个 VPTO/PTO op 的类型、地址空间、token、mode 对不对”
  交给 `lib/PTO/IR/VPTO.cpp` / `lib/PTO/IR/PTO.cpp`。
- “这批 op 放在一起时，vec scope、mask 来源和整体结构对不对”
  交给 authoring-stage verifier。
- “ptr-boundary 之后，最终发射前的 IR 有没有残留 memref 或桥接脚手架”
  交给 emission-stage verifier。

## 5. 当前明确延后的规则

以下规则当前只记录，不在 v1 中实现：

- `get_buf` / `rls_buf` acquire-release 配对；
- `set_flag` / `wait_flag` 的跨 block / 跨 CFG 顺序正确性；
- `mem_bar` / `pipe_barrier` 的充分性和最小性；
- UB alias、读写冲突和 hazard 证明；
- 复杂 mask provenance 追踪；
- 函数属性、调用关系、递归限制等更上层约束。

## 6. 与 OpenSpec 的关系

本文档是 VPTO legality 规则总表。  
对应的 OpenSpec change `add-dual-stage-vpto-legality-verifier` 负责定义：

- 双阶段 verifier 的正式输入契约；
- 两次 verify 的 pipeline 位置；
- authoring-stage 与 emission-stage 的责任边界；
- 需要实现的 pass、helper 和测试矩阵。

若文档规则与实现或 OpenSpec 存在冲突，以当前实现和对应 OpenSpec change 的已批准版本为准；文档应随后同步更新。
