# ATen - 核心算子库（二）：算子声明系统

> **前序**: [Part 1 - 架构概览](./01-architecture.md)  
> **源码位置**: `aten/src/ATen/native/native_functions.yaml`

---

## 1. 什么是 native_functions.yaml?

`native_functions.yaml` 是 ATen 的**算子声明中心**，所有原生算子都在此声明。

**核心作用**:
1. 定义算子签名（参数、返回值）
2. 指定后端实现（CPU/CUDA/MPS 等）
3. 生成 C++ 头文件/绑定代码
4. 生成 Python 绑定

**文件大小**: 约 15000+ 行，声明 2000+ 算子

---

## 2. YAML 声明格式详解

### 2.1 基本结构

每个算子声明包含以下部分：

```yaml
- func: <函数名>(<参数列表>) -> <返回值>
  <可选属性 1>: <值>
  <可选属性 2>: <值>
  dispatch:
    <后端>: <实现函数名>
```

### 2.2 完整示例：abs 算子

```yaml
# aten/src/ATen/native/native_functions.yaml: L340-L365

# 函数版本和方法版本
- func: abs(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: abs
    SparseCPU, SparseCUDA, SparseMPS: abs_sparse
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: abs_sparse_csr
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_abs
  tags: [core, pointwise]

# Inplace 版本
- func: abs_(Tensor(a!) self) -> Tensor(a!)
  device_check: NoCheck
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: abs_
    SparseCPU, SparseCUDA, SparseMPS: abs_sparse_
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: abs_sparse_csr_
    NestedTensorCPU, NestedTensorHPU, NestedTensorCUDA: NestedTensor_abs_

# out 版本 (预分配输出)
- func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  device_check: NoCheck
  dispatch:
    CPU, CUDA, MPS, MTIA: abs_out
    SparseCPU, SparseCUDA, SparseMPS: abs_sparse_out
    SparseCsrCPU, SparseCsrCUDA, SparseCsrMPS, SparseCsrMeta: abs_sparse_csr_out
  tags: pointwise
```

**解读**:
- 一个算子有 3 个变体：函数版、inplace 版、out 版
- `dispatch` 指定不同后端的实现函数
- `tags` 标记算子类型（pointwise = 逐元素运算）

---

## 3. func 字段：函数签名

### 3.1 参数类型

支持的参数类型：

| 类型 | C++ 对应 | 说明 |
|------|---------|------|
| `Tensor` | `const Tensor&` | 张量（默认 undefined 不允许） |
| `Tensor?` | `std::optional<Tensor>` | 可选张量（可为 None） |
| `Tensor[]` | `ArrayRef<Tensor>` | 张量列表 |
| `int` | `int64_t` | 整数 |
| `int[]` | `ArrayRef<int64_t>` | 整数列表 |
| `float` | `double` | 浮点数 |
| `bool` | `bool` | 布尔值 |
| `str` | `std::string_view` | 字符串 |
| `Scalar` | `Scalar` | 通用数值（int/float/tensor） |
| `Generator?` | `Generator?` | 随机数生成器 |
| `*` | - | 分隔符，后续参数必须 keyword-only |

**示例** (L41-43):
```yaml
- func: _backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, 
                   bool? retain_graph=None, bool create_graph=False) -> ()
```

### 3.2 默认值

```yaml
# 数值默认值
- func: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor

# 布尔默认值
- func: dropout(Tensor input, float p=0.5, bool train=True) -> Tensor

# 列表默认值
- func: ones(int[] size, *, int[2] stride=[1, 1]) -> Tensor

# None 默认值 (可选参数)
- func: linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
```

### 3.3 返回值类型

**单一返回值**:
```yaml
- func: abs(Tensor self) -> Tensor
- func: sum(Tensor self) -> Tensor
```

**Tuple 返回值**:
```yaml
# L273: _fused_dropout 返回 tensor 和 mask
- func: _fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)

# L286: native_dropout 返回 tensor 和 mask
- func: native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
```

**带命名的返回值** (用于 autograd):
```yaml
- func: max(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
```

### 3.4 Overload 机制

同一函数名可以有多个 overload：

```yaml
# 基础版本
- func: native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)

# Tensor 版本（不同的参数类型）
- func: native_dropout.Tensor(Tensor input, Tensor p, bool? train) -> (Tensor, Tensor)
```

**Overload 命名规则**:
- 第一个 overload 可以没有后缀（默认 overload）
- 后续 overload 必须唯一命名
- 常见命名：按第一个不同参数类型/名称命名

---

## 4. variants 字段：生成变体

### 4.1 可选值

```yaml
variants: function, method
```

| 变体 | 生成内容 | 示例 |
|------|---------|------|
| `function` | 命名空间函数 | `at::abs(tensor)` |
| `method` | Tensor 方法 | `tensor.abs()` |

### 4.2 默认行为

**不指定 variants** → 只生成 function:
```yaml
- func: empty(int[] size, ...) -> Tensor
# 生成：at::empty(...)
# 不生成：tensor.empty()
```

**指定 variants: function, method**:
```yaml
- func: abs(Tensor self) -> Tensor
  variants: function, method
# 生成：at::abs(tensor) 和 tensor.abs()
```

### 4.3 method 的要求

如果声明 `method`，必须有 `Tensor self` 参数：

```yaml
# 正确：self 是第一个参数
- func: abs_(Tensor(a!) self) -> Tensor(a!)
  variants: method
# 生成：tensor.abs_()

# 错误：没有 self 参数
- func: add(Tensor a, Tensor b) -> Tensor
  variants: method  # 这样会生成 a.add(b)，但语义不对
```

---

## 5. dispatch 字段：后端实现

### 5.1 DispatchKey 列表

常用 DispatchKey：

| Key | 说明 |
|-----|------|
| `CPU` | CPU 后端 |
| `CUDA` | NVIDIA GPU |
| `MPS` | Apple Silicon GPU |
| `MTIA` | Meta TPU |
| `CompositeExplicitAutograd` | 所有后端，需要手动定义 backward |
| `CompositeImplicitAutograd` | 所有后端，自动支持 autograd |
| `SparseCPU`, `SparseCUDA` | 稀疏张量后端 |
| `NestedTensorCPU`, `NestedTensorCUDA` | NestedTensor 后端 |

### 5.2 注册格式

```yaml
dispatch:
    CPU: func_cpu
    CUDA: func_cuda
    MPS: func_mps
```

**多后端共享实现**:
```yaml
dispatch:
    CPU, CUDA, MPS, MTIA: abs_out  # 多个后端用逗号分隔
```

**省略 dispatch** → 默认 `CompositeImplicitAutograd`:
```yaml
# 以下两种写法等价
- func: my_op(Tensor self) -> Tensor
  dispatch:
    CompositeImplicitAutograd: my_op

- func: my_op(Tensor self) -> Tensor
  # 默认就是 CompositeImplicitAutograd
```

### 5.3 选择 DispatchKey 的决策树

```
1. 你的 Kernel 是否对所有后端都适用？
   ├── No → 使用具体后端 (CPU/CUDA/MPS)
   │        并在每个后端实现不同版本
   │
   └── Yes → 进入下一步

2. 是否自动支持 autograd？
   ├── Yes → CompositeImplicitAutograd (或不写 dispatch)
   └── No → CompositeExplicitAutograd
            (需要在 derivatives.yaml 定义 backward)
```

### 5.4 实战示例

**示例 1**: 逐元素运算 (所有后端相同)
```yaml
- func: abs(Tensor self) -> Tensor
  dispatch:
    CompositeExplicitAutograd: abs
```

**示例 2**: 卷积（不同后端不同实现）
```yaml
- func: cudnn_convolution(...) -> Tensor
  dispatch:
    CUDA: cudnn_convolution_cuda
    # CPU 没有 cuDNN 实现
```

**示例 3**: 稀疏张量
```yaml
- func: abs(Tensor self) -> Tensor
  dispatch:
    CompositeExplicitAutograd: abs
    SparseCPU, SparseCUDA, SparseMPS: abs_sparse
```

---

## 6. 标注系统 (Annotations)

### 6.1 为什么需要标注？

标注用于描述 Tensor 参数的**别名**和**可变性**关系。

### 6.2 标注语法

```
Tensor(<annotation>)
```

| 标注 | 含义 |
|------|------|
| `a` | 属于别名集合 a |
| `a!` | 属于集合 a 且可被写入（inplace） |
| `a! -> a\|b` | 写入后从集合 a 变为集合 a 和 b |
| `a -> *` | 从集合 a 进入通配符集合（列表中） |

### 6.3 使用场景

**场景 1: Inplace 运算**
```yaml
# abs_ 修改 self 并返回
- func: abs_(Tensor(a!) self) -> Tensor(a!)
```

**场景 2: out 参数**
```yaml
# out 参数被写入并返回
- func: abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
```

**场景 3: View 运算**
```yaml
# transpose 返回 self 的视图（共享内存）
- func: transpose(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
```

**场景 4: 列表输出**
```yaml
# chunk 返回的列表元素都 alias self
- func: chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]
```

---

## 7. 其他重要字段

### 7.1 device_check

```yaml
device_check: NoCheck
```

**默认行为**: 检查所有 Tensor 参数是否在同一设备上。

**使用 `NoCheck`** 的情况:
- 函数允许不同设备参数（如 CTC loss）
- 函数不访问设备（如 metadata 查询）

**示例** (L222-225):
```yaml
- func: _use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, ...) -> bool
  device_check: NoCheck  # log_probs 在 CUDA，targets 在 CPU
  dispatch:
    CUDA: _use_cudnn_ctc_loss
```

### 7.2 tags

```yaml
tags: [core, pointwise]
```

常见标签：
- `core`: 核心算子
- `pointwise`: 逐元素运算
- `nondeterministic_seeded`: 随机性算子
- `view`: 视图运算
- `inplace_view`: inplace 视图

### 7.3 autogen

自动生成其他变体：

```yaml
# 从 inplace 生成 functional 和 out 版本
- func: my_op_(Tensor(a!) self) -> Tensor(a!)
  autogen: my_op, my_op.out
```

---

## 8. 别名 (Alias) 声明

### 8.1 什么是别名？

别名是同一算子的不同名称（如 `abs` 和 `absolute`）。

### 8.2 添加别名的步骤

参考 Note [Adding an alias] (L367-L387):

```yaml
# 原始算子
- func: abs(Tensor self) -> Tensor
  dispatch:
    CompositeExplicitAutograd: abs

# 别名 - 不指定 dispatch
- func: absolute(Tensor self) -> Tensor
  variants: function, method
  # 注意：没有 dispatch 字段！
```

**关键点**:
- 别名**不能有 dispatch 字段**
- 否则 autograd 无法继承原始行为

### 8.3 完整清单

添加别名需要修改 6 个文件：

1. `native_functions.yaml` - 声明别名
2. C++ 实现 - 重定向到原算子
3. `torch/_torch_docs.py` - 文档
4. `torch/overrides.py` - 重载支持
5. `torch/csrc/jit/passes/normalize_ops.cpp` - JIT 规范化
6. `torch/testing/_internal/common_methods_invocations.py` - 测试

---

## 9. 特殊算子类型

### 9.1 Factory 函数

不依赖输入 Tensor 创建新 Tensor：

```yaml
# 注意：没有 Tensor 输入参数
- func: empty(int[] size, *, ScalarType? dtype=None, Layout? layout=None, 
               Device? device=None) -> Tensor
```

### 9.2 结构化算子

委托给另一个算子实现：

```yaml
- func: sgn(Tensor self) -> Tensor
  structured_delegate: sgn.out  # 委托给 sgn.out 实现
```

### 9.3 手动绑定

```yaml
- func: _backward(Tensor self, ...) -> ()
  manual_cpp_binding: True  # 不自动生成绑定
```

---

## 10. 源码阅读实战

### 10.1 定位算子

```bash
# 查找算子声明
grep "^- func: relu" aten/src/ATen/native/native_functions.yaml

# 查找所有 overload
grep "^- func: native_dropout" aten/src/ATen/native/native_functions.yaml
```

### 10.2 理解声明

以 `add` 为例：

```yaml
# L??? (grep 查找)
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CompositeExplicitAutograd: add
```

**解读**:
1. 函数名：`add.Tensor` (Tensor 版本)
2. 参数：self, other, alpha=1
3. 生成：`at::add(self, other, alpha=1)` 和 `self.add(other, alpha=1)`
4. 实现：`add` 函数（CompositeExplicitAutograd）

### 10.3 追踪实现

```bash
# 1. 查找 YAML 中 dispatch 的函数名
grep "add.Tensor" aten/src/ATen/native/native_functions.yaml

# 2. 查找 C++ 实现
rg "add_out\(" aten/src/ATen/native/

# 3. 查找 REGISTER_DISPATCH
rg "REGISTER_DISPATCH.*add" aten/
```

---

## 11. 代码生成预览

从 YAML 生成的代码：

### 11.1 C++ 函数声明

```cpp
// aten/src/ATen/Operators.h
Tensor abs(const Tensor& self);
Tensor abs_(Tensor& self);
Tensor abs_out(const Tensor& self, Tensor& out);
```

### 11.2 Tensor 方法

```cpp
// aten/src/ATen/TensorMethods.h
Tensor Tensor::abs() const;
Tensor& Tensor::abs_();
```

### 11.3 注册代码

```cpp
// 生成的注册代码
m.def("abs(Tensor self) -> Tensor", &abs_cpu);
m.def("abs_(Tensor(a!) self) -> Tensor(a!)", &abs__cpu);
```

### 11.4 Python 绑定

```python
# torch/_C/__init__.py 生成
torch.abs(tensor)      # 函数
tensor.abs()           # 方法
```

---

## 12. 常见错误与排查

### 12.1 编译错误

**错误**: "missing dispatch for backend X"

**原因**: 添加了新算子但没有为所有必要后端实现

**解决**: 在 `dispatch` 中添加对应后端

### 12.2 Python 绑定错误

**错误**: "argument X must be Tensor, not int"

**原因**: YAML 类型与实际类型不匹配

**解决**: 检查 YAML 参数类型定义

### 12.3 Autograd 错误

**错误**: "element 0 of tensors does not require grad"

**原因**: `CompositeExplicitAutograd` 但没有 derivatives.yaml 公式

**解决**: 
1. 添加 derivatives.yaml 公式，或
2. 改用 `CompositeImplicitAutograd`

---

## 13. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `native_functions.yaml` | L1-L40 | 废弃的 cast 算子 |
| `native_functions.yaml` | L274-L385 | dispatch 关键字说明 |
| `native_functions.yaml` | L340-L400 | abs 算子完整声明 |
| `native_functions.yaml` | L486-L498 | autogen 字段说明 |
| `native_functions.yaml` | L501-L665 | C++ 实现指南 |
| `native/README.md` | L1-L50 | YAML 格式详解 |
| `native/README.md` | L274-L385 | dispatch 选择指南 |

---

## 14. 下一步

| Part | 主题 |
|------|------|
| [Part 3](./03-dispatch-mechanism.md) | Dispatch 机制详解 |
| [Part 4](./04-kernel-implementation.md) | C++ Kernel 实现 |

---

**参考资料**:
- [ATen Native README](../../../aten/src/ATen/native/README.md)
- `torchgen/` - 代码生成器源码
