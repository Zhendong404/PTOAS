# Issue 417: 统一 PTODSL Kernel Entry 为 ptr + int 的分析与计划

## 背景

Issue: https://github.com/mouliangyu/PTOAS/issues/417

当前 `pto-dsl-impl` 分支的 Python DSL 主要按 A5 芯片能力设计。A5 有向量寄存器和较完整的 vreg/tile 编程模型，因此现有文档和示例大量使用：

```python
@pto.jit(target="a5")
def kernel(A: pto.tensor_spec(rank=2, dtype=pto.f32)):
    a = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
```

这个入口形式把 Python host tensor 的 ABI 信息写在 `tensor_spec(rank=..., dtype=...)` annotation 中。它适合当前 A5-oriented DSL 的开发体验，但不是跨芯片的最小共同 ABI。

A2/A3 没有 A5 的向量寄存器模型，kernel 更自然也更接近已有 C++ 写法：

```cpp
__global__ AICORE void kernel(__gm__ float *x, int32_t batch)
```

因此 issue 希望把 Python DSL 的 kernel entry 统一到 `ptr + int`：

```python
N = 1024  # static, closure

@pto.jit
def my_kernel(x_ptr: pto.ptr(pto.f32, "gm"), batch: pto.i32):
    x = pto.make_tensor_view(
        x_ptr,
        shape=[batch, N],
        strides=[N, 1],
    )
```

核心原则：

- kernel entry 只暴露裸 GM pointer 和运行时 scalar；
- 静态 shape 轴通过 Python closure / constexpr 固定；
- 动态 shape 轴作为 `pto.i32`/`pto.i64` 等 scalar 入参；
- 不在函数参数 annotation 中写 rank；
- rank 由 `make_tensor_view(..., shape=[...])` 的 `len(shape)` 推断；
- launch 侧从 torch tensor 或其他 host tensor 取 shape，再作为额外 Python int 传给 kernel；
- 入口 ABI 对齐 A2/A3/A5 的共同下限，也对齐现有 C++ kernel 的 `__gm__ T* + int32_t` 风格。

## 当前代码现状

当前代码已经具备一部分基础能力：

1. `make_tensor_view` 已经从 `shape` 推断 rank  
   位置：`ptodsl/ptodsl/_ops.py`  
   `make_tensor_view(ptr, shape=..., strides=...)` 内部使用 `rank = len(shape)`，并从 pointer element type 推 tensor view dtype。

2. runtime scalar entry 已经存在  
   位置：`ptodsl/ptodsl/_kernel_signature.py`  
   `pto.i32`、`pto.f32` 等 annotation 会被解析成 `RuntimeScalarParameterSpec`。

3. 裸 pointer entry 的下游结构已经存在  
   位置：
   - `ptodsl/ptodsl/_kernel_signature.py`: `DeviceParameterSpec`
   - `ptodsl/ptodsl/_runtime/codegen.py`: `DeviceParameterSpec` codegen 分支
   - `ptodsl/ptodsl/_runtime/launch.py`: `DeviceParameterSpec` launch marshaling 分支

4. 但 `@pto.jit` 入口目前不接受 `pto.ptr(...)`  
   `parse_jit_kernel_signature` 只接受 `pto.tensor_spec(...)` 和 runtime scalar。`pto.ptr(...)` 会走到非法 annotation 诊断路径。测试里也有“entry annotation 使用 `pto.ptr(...)` 应报错”的预期。

结论：这个 issue 不是从零实现 ptr ABI，而是要把已有的 `DeviceParameterSpec` 通路正式接到 public `@pto.jit` entry 上，并迁移测试/示例来验证跨芯片共同 ABI。

## Issue 提出的问题

当前 `tensor_spec(rank=..., dtype=...)` 入口存在几个问题：

1. rank 泄漏到 entry annotation  
   issue 明确要求省略函数参数类型里的 rank。rank 应由 kernel body 里的 `make_tensor_view` 根据 `shape` 推断。

2. 动态 shape 表达不统一  
   `tensor_spec` 会把 host tensor 的 shape/strides 自动展开为 entry ABI metadata。issue 希望动态轴显式成为 kernel 参数，例如 `batch: pto.i32`，静态轴留在 closure 中。

3. A5-oriented 设计不适合作为 A2/A3 的共同入口  
   A5 可以在 DSL 内部大量使用 vreg/tile，但 A2/A3 没有向量寄存器，入口层必须回到 pointer + scalar 这个共同模型。

4. C++ 到 Python DSL 迁移成本偏高  
   现有 C++ kernel 多数是 `__gm__ T* + int32_t`。如果 Python DSL entry 也采用 ptr + int，迁移时只需要把 body 内部用 `make_tensor_view` 建描述符，而不是先改成 `tensor_spec` 风格。

## 要解决的点

本 issue 的完成标准应包括：

1. `@pto.jit` public entry 支持 `pto.ptr(...)`
   - `x: pto.ptr(pto.f32, "gm")` 应被解析为一个 device pointer 参数；
   - IR entry 参数应是 `!pto.ptr<f32, gm>`；
   - launch wrapper 应把 Python tensor / raw pointer 转为 `void*` 或 typed pointer 后传入。

2. `@pto.jit` public entry 支持 ptr 与 runtime scalar 混排
   - 例如 `x_ptr: pto.ptr(...), batch: pto.i32, cols: pto.i32`；
   - scalar 参数在 IR entry 中保持对应整数类型；
   - launch 侧接受 Python int 并按 annotation marshaling。

3. `make_tensor_view` 是 tensor rank 的唯一推断点
   - 不需要也不允许在 ptr annotation 中携带 rank；
   - `shape=[batch, N]` 的长度决定 tensor view rank；
   - shape 可以混合 runtime scalar 和 static Python int。

4. 保持 A5 能力不被破坏
   - A5 内部仍可使用 vreg/tile/subkernel；
   - ptr entry 只是 entry ABI 统一，不要求删除 A5 内部 vreg 模型；
   - 原有 `tensor_spec` 是否保留兼容，需要单独决策。建议第一阶段保留，先把 issue 的新入口跑通。

5. 给 A2/A3 留出自然入口
   - target 为 `a2`/`a3` 的 DSL kernel 可以直接使用 ptr + int；
   - 不依赖 vreg-only 的 entry abstraction；
   - 后续 A2/A3 lowering 若受限，应在 body ops / target legality 层诊断，而不是 entry ABI 层阻塞。

## 解决方案

### 1. 接通 ptr entry parsing

修改 `ptodsl/ptodsl/_kernel_signature.py`：

- 新增 `_is_supported_device_parameter_annotation(annotation)`；
- 判断 annotation 是否是 `_PtrDescriptor` 或已解析的 PTO `PtrType`；
- 对 ptr annotation 生成 `DeviceParameterSpec`；
- 保持 runtime scalar 生成 `RuntimeScalarParameterSpec`；
- 保持 `tensor_spec` 兼容，至少第一阶段不删除。

预期解析顺序：

```text
tensor_spec(...) -> TensorSpecParameterSpec
pto.ptr(...)     -> DeviceParameterSpec
pto.i32/f32/...  -> RuntimeScalarParameterSpec
otherwise        -> diagnostic
```

### 2. 明确 ptr API 形式

当前代码已有：

```python
pto.ptr(pto.f32, "gm")
pto.ptr(pto.f32, pto.MemorySpace.GM)
```

Issue 示例写法是：

```python
pto.ptr(dtype=pto.f32)
```

建议分两步：

1. 第一阶段使用已有 API：`pto.ptr(pto.f32, "gm")`，避免扩大改动面；
2. 第二阶段补兼容糖：`pto.ptr(dtype=pto.f32, space="gm")` 或 `pto.ptr(dtype=pto.f32, address_space="gm")`。

不建议直接修改 `pto.ptr` 的全局默认 address space。当前 `pto.ptr(elem, space="ub")` 可能已被内部 UB pointer 场景依赖。entry 示例里应显式写 `"gm"`，避免因为默认值不同导致 ABI 错误。

### 3. 复用现有 launch/codegen

`DeviceParameterSpec` 在 codegen 和 launch 中已经有处理分支：

- codegen 生成 `__gm__ T *param`；
- host wrapper 接收 `T *param`；
- launch marshaling 用 `_as_void_ptr` 支持 `ctypes.c_void_p`、integer pointer、以及带 `.data_ptr()` 的 torch tensor。

因此第一阶段不需要重写 launch，只需要补测试确认：

```python
compiled[grid, stream](x_tensor, int(x_tensor.shape[0]))
```

会被 marshal 为：

```text
x_tensor.data_ptr(), c_int32(batch)
```

### 4. 保持 `make_tensor_view` 作为 body 内显式转换

现有 `make_tensor_view` 已经符合 issue 方向。要补充测试覆盖：

```python
N = 1024

@pto.jit(target="a5")
def ptr_dynamic_shape_probe(x: pto.ptr(pto.f32, "gm"), batch: pto.i32):
    x_view = pto.make_tensor_view(x, shape=[batch, N], strides=[N, 1])
```

断言重点：

- entry function 参数包含 `!pto.ptr<f32, gm>` 和 `i32`；
- `pto.make_tensor_view` 的 shape 包含 `%arg1` 和 static constant `1024`；
- tensor view rank 为 2；
- annotation 中没有 `rank=`。

### 5. 更新诊断

当前 `pto.ptr(...)` entry annotation 会被诊断为非法。需要调整：

- 删除或改写“`@pto.jit` 不支持 `pto.ptr(...)` entry”的测试；
- 新增非法指针类型诊断，例如 storage-only dtype 不允许作为 ptr element；
- 对非 GM entry pointer 是否允许做明确策略。

建议策略：

- public launch entry 第一阶段只允许 GM pointer；
- UB/MAT/LEFT/RIGHT 等非 GM pointer 仍只用于 kernel body / subkernel boundary；
- 如果用户在 public `@pto.jit` entry 写 `pto.ptr(pto.f32, "ub")`，给出清晰错误：public launch entry expects GM pointer。

这样可以避免 host 侧传入普通 tensor data pointer 却被标成 UB pointer 的错误。

### 6. 更新示例和文档

优先更新最小闭环示例：

- `ptodsl/examples/jit/tadd_launch.py`
- `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`
- 相关 docs fixture / docs-as-test

迁移原则：

旧风格：

```python
@pto.jit(target="a5")
def add(A: pto.tensor_spec(rank=2, dtype=pto.f32), O: pto.tensor_spec(rank=2, dtype=pto.f32)):
    a = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
```

新风格：

```python
N = 1024

@pto.jit(target="a5")
def add(A: pto.ptr(pto.f32, "gm"), O: pto.ptr(pto.f32, "gm"), batch: pto.i32):
    a = pto.make_tensor_view(A, shape=[batch, N], strides=[N, 1])
    o = pto.make_tensor_view(O, shape=[batch, N], strides=[N, 1])
```

launch 侧：

```python
add.compile()[grid, stream](A, O, int(A.shape[0]))
```

## 实施计划

### Phase 1: 最小功能闭环

目标：一个 ptr + int dynamic shape kernel 可以 compile，并生成预期 IR。

改动：

1. 修改 `parse_jit_kernel_signature`，支持 `pto.ptr(..., "gm")` entry；
2. 新增 frontend compile test；
3. 修改原有 ptr entry diagnostics test；
4. 验证 `make_tensor_view` shape 混合 runtime scalar 和 static int。

验收：

```bash
python -m pytest test/python/ptodsl_jit_compile.py test/python/ptodsl_jit_diagnostics.py
```

如本地缺 MLIR/PTOAS Python 环境，则至少跑目标单测或记录无法运行原因。

### Phase 2: launch 侧闭环

目标：torch tensor / pointer + Python int 可以调用 compiled kernel wrapper。

改动：

1. 添加 `_marshal_launch_args` 单测，覆盖 `DeviceParameterSpec + RuntimeScalarParameterSpec`；
2. 验证 `tensor.data_ptr()` 被用于 ptr 参数；
3. 验证 Python int 按 `pto.i32` marshaling；
4. 如有真实 NPU 环境，再跑一个小型 JIT launch demo。

验收：

- `compiled[grid, stream](tensor, int(tensor.shape[0]))` 不再要求 `tensor_spec`；
- launch wrapper 生成参数顺序为 `grid, stream, ptr, batch`。

### Phase 3: target 兼容策略

目标：明确 A5 与 A2/A3 的边界。

改动：

1. public entry ptr ABI 对所有 target 可用；
2. A5 继续允许 body 内使用 vreg/tile；
3. A2/A3 不支持的 vreg/tile op 由 target legality 或 lowering 阶段诊断；
4. 文档说明 ptr + int 是跨芯片推荐 entry ABI。

验收：

- `@pto.jit(target="a5")` ptr entry 测试通过；
- 新增 `target="a3"` 或 `target="a2"` 的 compile-only entry test，如果当前 toolchain 支持；
- 若 toolchain 暂不支持 A2/A3 Python DSL compile，则文档标注 pending，不阻塞 Phase 1。

### Phase 4: 文档和示例迁移

目标：把 public-facing 示例从 `tensor_spec` 主路径迁移到 ptr + int。

改动：

1. 更新 quick start / kernel entry 文档；
2. 更新至少一个 JIT launch 示例；
3. docs-as-test fixture 对齐；
4. 保留 `tensor_spec` 兼容说明，或者标为 legacy/compat。

验收：

- 文档中的主推荐写法是 `pto.ptr(..., "gm") + pto.i32`；
- `tensor_spec` 不再被描述为唯一 public entry ABI。

## 风险与待确认问题

1. `tensor_spec` 是否最终删除  
   建议不在本 issue 中删除。先保留兼容，避免大规模 docs/tests 迁移阻塞 ptr ABI 落地。

2. `pto.ptr(dtype=pto.f32)` 是否必须第一阶段支持  
   issue 示例用了这个写法，但当前实现已有 `pto.ptr(pto.f32, "gm")`。建议第一阶段先支持现有 API，第二阶段补 keyword 兼容。

3. public entry 是否只允许 GM pointer  
   建议只允许 GM pointer。host tensor 的 `.data_ptr()` 是全局内存地址，标成 UB/MAT pointer 不合理。

4. shape scalar 用 `i32` 还是 `index/i64`  
   issue 示例用 `pto.i32`，但当前 `tensor_spec` metadata 使用 `int64_t`。建议遵循 issue：动态轴参数由用户 annotation 决定，示例使用 `pto.i32`；如需要大 shape，可用 `pto.i64` 或 `pto.index`。

5. strides 是否也要作为 runtime int 参数  
   issue 核心是动态 shape 轴，但 `make_tensor_view` 也需要 strides。第一阶段示例可用 static/closure strides；非 contiguous tensor 的通用支持需要用户额外传 stride 参数，或者在更高层 wrapper 自动传入。

## 推荐开发顺序

1. 先实现 `@pto.jit` ptr entry parsing；
2. 加最小 compile test，确认 IR entry 与 `make_tensor_view` 正确；
3. 加 launch marshaling test；
4. 改一个 JIT 示例；
5. 更新文档；
6. 再考虑 `pto.ptr(dtype=...)` keyword 兼容和 `tensor_spec` legacy 策略。

