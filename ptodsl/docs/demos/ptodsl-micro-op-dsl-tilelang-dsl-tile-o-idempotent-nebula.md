# ptodsl 语法手册重构计划（修订版 v4）

## Context

当前 `ptodsl/docs/user_guide/` 有一份 16 章的旧 DSL 手册，使用已废弃的 `@pto.vkernel`/`@pto.ckernel`。新的 ptodsl 采用 **tracing** 方式，统一使用以下抽象层级（参见 `flash_attention_sketch.py`）：

```
L0  Python wrapper          flash_attention(...)
L1  @pto.jit                编译+缓存+启动，顶层编排
L2  @pto.ukernel            micro kernel，per-block 执行编排
L3a @pto.cube               矩阵乘（Cube 单元）
L3b @pto.simd               行向量数学（Vector 单元）
L3c @pto.simt               标量控制与逐点操作
```

**核心原则**：
- `@pto.vkernel`/`@pto.ckernel` 废弃，统一为新层级
- Tile Op 和 Micro Op **统一看待为 PTO Op**，按功能分类编排
- 类型系统、控制流等 **统一章节**，不按抽象层级拆分
- Micro Op 指令的操作名和签名尽量复用旧手册内容
- ptodsl 采用 **tracing** 方式生成代码，Python 原生控制流在 trace 时执行（编译期展开），`pto.for_`/`pto.if_` 产生设备端动态控制流
- 删除 `pto.vector`、`@pto.inline_proc`、`vecscope`/`strict_vecscope`

---

## 手册结构（14 章，按 topic 组织）

### 第1章：概述
- ptodsl 定位与目标硬件（昇腾 NPU）
- 抽象层级模型：L0 wrapper → L1 `@pto.jit` → L2 `@pto.ukernel` → L3 `@pto.cube`/`@pto.simd`/`@pto.simt`
- 以 flash attention 为例展示各层职责
- Tile Op（操作 tile/view 的 op）和 Micro Op（操作 vreg/ptr 的 op）统一为 PTO Op 体系
- **tracing 执行模型简介**：核函数在编译时被 trace 一次，生成设备代码再启动执行。Python 原生控制流只在 trace 时运行（不会出现在最终设备代码中）；需要设备端动态行为时用 `pto.for_`/`pto.if_`
- 阅读指引

### 第2章：快速入门
- 最小完整 kernel：逐元素向量加法，使用 `@pto.jit` + `@pto.simd`
- 关键概念一览：`alloc_tile`、`partition_view`、`tload`、`tstore`、`pto.for_`、`vlds`、`vadd`、`vsts`
- 编译与启动：`kernel.compile(...)` → `compiled[grid, stream](args...)`
- SPMD 启动示例

### 第3章：核函数入口与子核函数
- 装饰器家族完整参考：
  - `@pto.jit(target=...)`：顶层 JIT 入口，constexpr keyword-only 参数，动态 shape，编译/启动路径，SPMD 内建函数
  - `@pto.ukernel`：micro kernel，per-block 编排，内部使用 DMA/同步指令 + 子核调用
  - `@pto.cube`：矩阵乘子核，参数类型约定
  - `@pto.simd`：向量数学子核，参数类型约定
  - `@pto.simt`：标量/逐点子核，参数类型约定
- 边界约定：vreg 不跨子核，数据通过 UB tile 传递
- `pto.constexpr`：编译期常量参数

### 第4章：类型系统与缓冲区管理（统一）
- 来源：旧第5章 + 原第6章内容整合，删除 `pto.vector`

**第一部分：类型定义**
- 标量类型（`pto.f32`、`pto.i32`、`pto.f16` 等），字面量规则
- 低精度类型（`hif8`、`f4e1m2x2`、`f4e2m1x2`、`f8e4m3`、`f8e5m2`），强调只用于存储
- Vector 寄存器类型（`pto.vreg`），`pto.vbitcast`
- 谓词/掩码类型（`mask_b8`、`mask_b16`、`mask_b32`），`pbitcast`
- 指针类型：`pto.ptr(dtype, memory_space)`，`MemorySpace` 枚举
- **TensorView**：构造、属性、切片语法、padding 模式
- **PartitionTensorView**：分区描述符
- **Tile**：完整构造函数、layout 默认值、`valid_shape`、pad 值、`TileConfig`
- 类型速查表：哪些类型出现在 `@pto.jit` 参数中，哪些出现在子核参数中

**第二部分：缓冲区管理操作**
- `make_tensor_view(tensor, shape=..., strides=...)`：创建 GM 描述符
- `select_head_view(tv, batch=..., head=..., shape=...)`：per-head 切片
- `alloc_tile(shape=[...], dtype=..., memory_space=...)`：分配 UB / cube-local tile
- `partition_view(tv, offsets=[...], sizes=[...])`：创建 GM 分区描述符
- `tile.fill(value)`：标量填充
- `tile_valid_rows(tile)`、`tile_valid_cols(tile)`：逻辑维度查询
- `tile_buf_addr(tile)`、`as_ptr()`：获取裸指针

### 第5章：控制流（统一）

核心概念说明（不引入"IR"术语，用"trace 时" vs "设备端"区分）：

**ptodsl 的 tracing 工作方式**：编写核函数时，ptodsl 会执行一次你的 Python 代码来记录所有操作（称为"trace"），然后根据记录生成可在 NPU 上运行的程序。这意味着：

- **Python 原生 `for`/`if`**：在 trace 时执行。循环体会被重复记录多次（等于展开），`if` 的条件在 trace 时求值（只保留走到的分支）。适用于编译期已知的循环次数和条件。

- **`pto.for_`/`pto.if_`**：产生设备端动态控制流。循环次数和分支条件可以是运行时的值（不同输入可能有不同执行路径）。这是实现"真正"循环和分支的方式。

用例子讲清楚：
```python
# Python for：trace 时展开 4 次，设备上始终执行 4 次
for i in range(4):
    a = vadd(a, b)

# pto.for_：设备端动态循环，N 可以是运行时值
with pto.for_(0, N, step=1) as i:
    a = vadd(a, b)
```

**`pto.for_` 详细说明**：
- 单迭代器：`with pto.for_(start, stop, step) as iv: ...`
- 携带状态：`.carry(x=...)` → `kv_loop.update(...)` → `kv_loop.final("x")`
- ping-pong 状态模式示例

**`pto.if_` 详细说明**：
- 设备端条件分支，条件可以是运行时值
- 值合并语法

**`pto.constexpr`**：标记编译期常量（配合 `@pto.jit` keyword-only 参数使用），与 tracing 行为的关系

### 第6章：标量与指针操作

**6.1 Python 标量 vs PTO 标量**（延续第5章"trace 时 vs 设备端"区分）：

- **Python 原生标量**（`x = a + b`，a/b 为 Python int/float）
  → trace 时求值，结果烘焙进设备代码。适用于编译期确定的量：
  循环边界、tile 尺寸、常量系数（如 `1.0 / sqrt(dim)`）

- **PTO 标量操作**（`pto.lds(...)`、`pto.max(...)`、`pto.exp(...)`）
  → 产生设备端运行时标量值。适用于来自设备内存或依赖运行时输入的量：
  从 tile 读取元素、运行时依赖的数学计算

- 混合使用示例（来自 flash attention）：
  ```python
  alpha * o_prev + beta * pv_val
  # ^ Python float    ^ PTO 标量    ^ PTO 标量
  # （trace 时烘焙）  （设备端读取） （设备端读取）
  ```
- 简单规则：**trace 时能确定的 → Python 原生；来自设备/依赖运行时的 → `pto.*`**

**6.2 标量存取**：`pto.lds`、`pto.sts`、`load_scalar`、`store_scalar`
**6.3 标量算术**：`pto.max`、`pto.exp` 及其他标量数学操作
**6.4 指针操作**：`castptr`、`addptr`、`as_ptr()`
**6.5 编译期查询**：`bytewidth`、`elements_per_vreg`
**6.6 tile 索引下的标量读写**：`@pto.simt` 中的典型用法，嵌套 `pto.for_` 逐元素遍历 tile

### 第7章：数据搬运操作

**Tile 级搬运**：
- `tload(partition, tile)`：GM → UB
- `tstore(tile, partition)`：UB → GM

**DMA 搬运与控制**（ukernel 内）：
- 执行操作：`mte_load(src, dst)`、`mte_store(src, dst)` 等
- DMA 配置 [Advanced]：pad fill（`set_mov_pad_val`）、stride/loop 配置（`set_loop2_stride_outtoub` 等）、执行操作（`copy_gm_to_ubuf`、`copy_ubuf_to_ubuf`、`copy_ubuf_to_gm`）

**Vector 加载/存储**（simd 内）：
- 来源：旧第9章，保留全文
- `vlds`、`vldas`、`vldus`、`vldsx2`、`vsld`、`vgather2`、`vgather2_bc`、`vgatherb`、`vsldb`
- `vsts`、`psts`、`vsst`、`vstsx2`、`vsta`、`vscatter`、`vsstb`、`vstar`、`vstas`
- 有状态 store（`pstu`、`vstu`、`vstus`、`vstur`）
- 地址生成语法糖（tile 索引）、分发枚举

**Cube 数据搬运**（cube 内）：
- `cube_load`、`cube_store`、`cube_load_frac`
- `left_load`、`right_load`、`left_load_mx`、`right_load_mx`
- `bias_load`
- `acc_store`、`acc_store_gm`、`acc_store_ub`

### 第8章：计算操作

**Tile 级计算**（直接操作 `tile_buf`，可在 `@pto.jit`/`@pto.ukernel` 体内使用）：
- 来源：PTO Tile Instruction SPEC 第5-12章
- 逐元素算术：`pto.tadd`、`pto.tsub`、`pto.tmul`、`pto.tdiv`、`pto.tmax`、`pto.tmin`
- 逐元素标量：`pto.tadds`、`pto.tsubs`、`pto.tmuls`、`pto.tdivs`、`pto.tmaxs`、`pto.tmins`
- 一元数学：`pto.tabs`、`pto.tneg`、`pto.texp`、`pto.tlog`、`pto.tsqrt`、`pto.trsqrt`、`pto.trecip`
- 激活：`pto.trelu`、`pto.tlrelu`
- 规约：`pto.trowsum`、`pto.trowmax`、`pto.trowmin`、`pto.tcolsum`、`pto.tcolmax`、`pto.tcolmin` 等
- 偏元素：`pto.tpartadd`、`pto.tpartmul`、`pto.tpartmax`、`pto.tpartmin`
- 位运算：`pto.tnot`、`pto.tand`、`pto.tor`、`pto.txor`、`pto.tshl`、`pto.tshr` 及标量形式
- 类型转换：`pto.tcvt`（含 `RoundMode` 枚举）
- 广播/扩展：`pto.texpands`、`pto.trowexpand`、`pto.tcolexpand` 及算术变体
- 填充：`pto.tfillpad`、`pto.tfillpad_expand`

**Vector 算术**（simd 内，操作 `vreg`，需在 vecscope 内使用）：
- 来源：旧第11章，保留全文
- 一元、二元、向量-标量、规约、转换（`vcvt`）、重排、比较/选择、特殊（`vaxpy`、`vmull` 等）

**Cube 矩阵乘**（cube 内）：
- 来源：旧第12章计算部分
- `mad`、`mad_acc`、`mad_bias` 及其 MX 变体
- 地址空间与数据流说明

### 第9章：谓词与掩码操作

**Tile 级谓词操作**（直接操作 `tile_buf` 形式的掩码）：
- 选择：`pto.tsel`（tile-tile 谓词选择）、`pto.tsels`（tile-scalar 谓词选择）
  - 输入掩码为 packed predicate bytes 的 `tile_buf`（`i8`/`i16`/`i32` 元素类型）
  - 根据掩码在位粒度上在两组数据之间选择

**Micro 级谓词操作**（simd 内，操作 `vreg` 和 `!pto.mask`，在 vecscope 内使用）：
- 来源：旧第10章，保留全文
- 模式掩码、尾部处理、统一 `make_mask`
- 谓词逻辑、存取、交织/解交织

### 第10章：同步操作
- 来源：旧第8章同步部分，保留全文
- 同步枚举：`BarrierType`、`Pipe`、`Event`
- 流水线同步：`set_flag`、`wait_flag`、`pipe_barrier`
- 缓冲区同步：`get_buf`、`rls_buf`
- 内存屏障：`mem_bar`
- 跨核/核内同步：`set_cross_core`、`set_intra_block`、`set_intra_core`、`wait_flag_dev`、`wait_intra_core`

### 第11章：Flash Attention 完整走读
- 逐层注解 `flash_attention_sketch.py`
- L0 → L1 → L2 → L3a/b/c 完整贯穿

### 第12章：完整示例
- 逐元素向量加法（Tile Op 级完整版）
- 带尾部处理的向量操作
- GEMM kernel（`@pto.jit` + `@pto.cube`）
- 循环携带状态示例（在线归一化）

### 第13章：从旧版迁移
- `@pto.vkernel`/`@pto.ckernel` → 新层级映射
- `TensorView`/`Tile` 参数对应关系
- `for i in range(...)` → `pto.for_` 携带状态
- 常见旧版模式等价写法对照表
- 删除的概念：`@pto.inline_proc`（tracing 下天生 inline）、`vecscope`/`strict_vecscope`（后续加回）

### 第14章：常见错误与兼容性说明
- 合并旧第14+15+16章
- 新增 Tile Op 级常见错误
- 新增 tracing 相关常见误解（Python for vs pto.for_ 混淆）
- 兼容性说明

---

## 与旧版章节对应关系

| 旧编号 | 旧标题 | 处理 |
|--------|--------|------|
| 01 | 概述 | 废弃，重写为新的第1章 |
| 02 | 快速入门 | 废弃，重写为新的第2章 |
| 03 | 核函数声明（vkernel/ckernel） | **废弃** |
| 04 | 模板核函数 | **废弃** |
| 05 | 类型系统 | → 第4章，保留全文 + 整合缓冲区管理操作 + Tile 级类型 - `pto.vector` |
| 06 | 控制流 | → 第5章，**大改**：tracing 模型 + pto.for_/pto.if_，删除 inline_proc/vecscope |
| 07 | 前端操作 | 分散入第5章（constexpr）、第6章（标量/ptr）、第4章（类型查询） |
| 08 | 同步与 DMA | → 第10章（同步部分）+ 第7章（DMA 控制部分） |
| 09 | Vector 内存操作 | → 第7章 Vector 部分，基本保留 |
| 10 | 谓词操作 | → 第9章，基本保留 |
| 11 | Vector 算术操作 | → 第8章 Vector 部分，基本保留 |
| 12 | Cube 操作 | → 第7章（搬运）+ 第8章（计算） |
| 13 | 示例 | → 第11章（flash attention）+ 第12章（其他示例） |
| 14 | 常见错误 | → 合并入第14章 |
| 15 | 兼容性说明 | → 合并入第14章 |
| 16 | 后续步骤 | → 合并入第14章 |

---

## 文件变更清单

### 新建（5 个全新章节）
- `01-introduction.md`（重写）
- `02-quick-start.md`（重写）
- `03-kernel-entry-and-subkernels.md`（全新：装饰器家族）
- `11-flash-attention-walkthrough.md`（全新：完整走读）
- `13-migration-guide.md`（全新：迁移指南）

### 修改（6 个，基于旧章节更新）
- `04-type-system-and-buffer.md`（← 旧05 + 原第6章内容整合，删除 pto.vector，新增低精度类型）
- `05-control-flow.md`（← 旧06，**大改**：tracing 模型 + pto.for_/pto.if_，删除 inline_proc/vecscope）
- `06-scalar-and-pointer-ops.md`（← 旧07 部分 + 标量算术）
- `07-data-movement-ops.md`（← 旧09 + 旧12搬运 + 旧08 DMA控制 + 新增 tload/tstore）
- `08-compute-ops.md`（← 旧11 + 旧12计算 + Tile 级计算 ops）
- `09-predicate-ops.md`（← 旧10 + 新增 Tile 级谓词）
- `12-examples.md`（← 旧13，更新示例为新写法）
- `14-common-errors-and-compatibility.md`（← 旧14+15+16 合并 + 新增 Tile Op 错误 + tracing 误解）

### 基本保留/小幅修改（1 个）
- `10-sync-ops.md`（← 旧08 同步部分，删除 DMA 控制）

### 删除（8 个旧章节）
旧 01, 02, 03, 04, 13, 14, 15, 16

---

## 控制流章节的核心叙述方案（第5章）

不引入"IR"概念，用"trace 时" vs "设备端"区分：

> ptodsl 在编译你的核函数时，会**执行一次** Python 代码来记录所有操作（这个过程叫 tracing），然后根据记录生成可在 NPU 上运行的程序。
>
> 这意味着你在核函数里写的 Python `for`/`if` 会在 trace 时运行——循环体会被展开记录多次，条件分支会在 trace 时求值（只保留命中的分支）。如果你写 `for i in range(4)`，trace 之后设备代码里就是 4 条指令，不是一个循环。
>
> 当你需要**设备端的动态控制流**（循环次数由运行时输入决定、分支条件取决于运行时数据）时，使用 `pto.for_` 和 `pto.if_`。它们会生成设备上真正的循环和分支。
>
> **简单记忆**：Python 控制流 = 编译期，`pto.*` 控制流 = 运行期。

配合代码示例说明。

---

## 验证方法

1. 15 章文件全部存在，编号连续
2. 交叉引用正确
3. `flash_attention_sketch.py` 中每个 API 在全手册中有文档覆盖
4. Micro Op 操作签名表未被误删
5. 正文无 `@pto.vkernel`/`@pto.ckernel`（仅第14章提及）
6. 正文无 `pto.vector`、`@pto.inline_proc`、`vecscope`/`strict_vecscope`
7. 控制流章的核心叙述无"IR"字样，用"trace 时"/"设备端"表述
