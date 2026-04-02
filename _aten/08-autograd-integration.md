# ATen - 核心算子库（八）：自动微分集成

> **前序**: [Part 7 - Tensor 核心数据结构](./07-tensor-structure.md)  
> **核心源码**: `tools/autograd/derivatives.yaml`, `aten/src/ATen/native/AutogradComposite.cpp`

---

## 1. Autograd 与 ATen 的边界

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  Python 层 (torch.autograd)              │
│                   autograd.Function                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  C++ 层 (torch/csrc/autograd)           │
│                   Engine.cpp (反向传播引擎)              │
│                   function.cpp (Autograd Function)       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  ATen 层 (aten/src/ATen)                │
│                   derivatives.yaml (导数公式)            │
│                   AutogradComposite.cpp (复合实现)       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                  底层 Kernel (CPU/CUDA)                  │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Autograd 集成方式

PyTorch 提供两种 Autograd 集成方式：

| 方式 | 机制 | 适用场景 |
|------|------|---------|
| **CompositeImplicitAutograd** | 自动微分 | 算子由其他可微算子组成 |
| **derivatives.yaml** | 手动定义导数公式 | 需要数值稳定或高效实现 |

---

## 2. derivatives.yaml: 导数公式配置

### 2.1 文件结构

**源码**: `tools/autograd/derivatives.yaml` (L1-L200+)

```yaml
# 每个条目包含:
# - name: 算子名称和参数
# - dispatch (可选): 指定特定 Autograd 后端的导数
# - gradients: 输入梯度计算公式
# - output_differentiability (可选): 输出是否可微
# - non_differentiable: 标记不可微参数
```

### 2.2 基本格式

```yaml
# 示例：abs 算子
- name: abs(Tensor self) -> Tensor
  self: grad * self.sgn()
```

**解读**:
- `name`: 指定算子签名（必须与 native_functions.yaml 一致）
- `self`: 输入参数名，对应梯度公式
- `grad`: 输出梯度（grad_output）
- `self.sgn()`: sign 函数，abs 的导数是 sign(x)

### 2.3 梯度计算变量

在导数公式中，以下变量可用：

| 变量 | 含义 |
|------|------|
| `grad` | 第一个输出的梯度 |
| `grads[i]` | 第 i 个输出的梯度（多输出时） |
| `grad_{name}` | 命名输出的梯度 |
| `result` | 前向计算结果 |
| `resultX` | 第 X 个输出结果 |
| `grad_input_mask` | 布尔数组，标识哪些输入需要梯度 |
| `self`, `other`, ... | 输入参数 |

### 2.4 完整示例

#### 示例 1: 二元运算

```yaml
# add.Tensor
- name: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  self: handle_r_to_c(self.scalar_type(), grad)
  other: handle_r_to_c(other.scalar_type(), maybe_multiply(grad, alpha.conj()))
```

**解读**:
- `d(self)/d(self) = 1`, 所以 `grad_self = grad * 1`
- `d(self)/d(other) = alpha`, 所以 `grad_other = grad * alpha`
- `handle_r_to_c`: 处理实数/复数类型转换
- `maybe_multiply`: 条件乘法（alpha=1 时优化）

#### 示例 2: 矩阵乘法

```yaml
# addmm: out = self + beta * mat1 @ mat2 * alpha
- name: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  self: maybe_multiply(grad, beta.conj())
  mat1: mm_mat1_backward(grad, mat2, mat1.sym_sizes(), mat1.sym_strides(), mat1.layout(), alpha)
  mat2: mm_mat2_backward(grad, mat1, mat2.sym_sizes(), mat2.sym_strides(), mat2.layout(), alpha)
```

**解读**:
- `grad_self = grad * beta`
- `grad_mat1 = grad @ mat2.T * alpha`
- `grad_mat2 = mat1.T @ grad * alpha`

#### 示例 3: 多输出算子

```yaml
# max: 返回最大值和索引
- name: max(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
  output_differentiability: [True, False]
  self: max_backward(grad, self, dim, keepdim)
```

**解读**:
- `output_differentiability`: values 可微，indices 不可微
- 只需要一个梯度公式（因为只有一个可微输出）

#### 示例 4: 不可微算子

```yaml
# ceil: 向上取整
- name: ceil(Tensor self) -> Tensor
  self: zeros_like(grad)
```

**解读**:
- ceil 是阶梯函数，导数为 0
- `zeros_like(grad)`: 返回与 grad 同形状的零张量

### 2.5 特殊语法

#### 使用 `not_implemented`

```yaml
# 尚未实现反向传播
- name: acosh_(Tensor(a!) self) -> Tensor(a!)
  self: not_implemented("inplace version of acosh")
```

#### 使用 `non_differentiable`

```yaml
# 参数不可微
- name: _is_all_true(Tensor self) -> Tensor
  self: non_differentiable
```

#### 使用 `auto_element_wise`

```yaml
# 逐元素算子，自动推导前向 JVP
- name: acos(Tensor self) -> Tensor
  self: grad * -((-self * self + 1).rsqrt()).conj()
  result: auto_element_wise
```

#### 使用 `auto_linear`

```yaml
# 线性算子，自动推导
- name: alias(Tensor(a) self) -> Tensor(a)
  self: grad
  result: auto_linear
```

---

## 3. CompositeImplicitAutograd

### 3.1 什么是 CompositeImplicitAutograd？

**CompositeImplicitAutograd** 是一种"元后端"，适用于：
- 算子实现只调用其他 `at::` 算子
- 自动支持所有后端（CPU/CUDA/MPS 等）
- 自动支持 autograd（无需 derivatives.yaml）

### 3.2 实现模式

```cpp
// 示例：简单的复合算子
Tensor my_op(const Tensor& self) {
    // 调用其他 at:: 算子
    return at::relu(at::conv2d(self, weight, bias));
}
```

**自动微分原理**:
```
前向：my_op(x) = relu(conv2d(x, w, b))

反向传播时:
1. PyTorch 记录计算图：x -> conv2d -> relu -> output
2. 反向时自动应用链式法则
3. 无需手动定义导数公式
```

### 3.3 默认行为

**YAML 不指定 dispatch**:
```yaml
# 默认注册到 CompositeImplicitAutograd
- func: my_op(Tensor self) -> Tensor
  # 等价于：
  # dispatch:
  #   CompositeImplicitAutograd: my_op
```

---

## 4. CompositeExplicitAutograd

### 4.1 与 CompositeImplicitAutograd 的区别

| 特性 | CompositeImplicitAutograd | CompositeExplicitAutograd |
|------|--------------------------|---------------------------|
| 实现 | 调用其他 `at::` 算子 | 直接操作数据 |
| Autograd | 自动支持 | 需要 derivatives.yaml |
| 性能 | 可能有额外开销 | 可优化实现 |
| 适用场景 | 简单复合算子 | 需要数值稳定/高效 |

### 4.2 实现模式

```cpp
// 示例：直接实现 abs，不调用其他 at:: 算子
Tensor abs(const Tensor& self) {
    // 直接操作数据
    return self.data_ptr() > 0 ? self : -self;
}
```

### 4.3 需要 derivatives.yaml

```yaml
# 必须在 derivatives.yaml 中定义
- name: abs(Tensor self) -> Tensor
  self: grad * self.sgn()
```

---

## 5. AutogradComposite.cpp

### 5.1 文件作用

**源码**: `aten/src/ATen/native/AutogradComposite.cpp`

该文件包含 Autograd 相关的复合算子实现。

### 5.2 核心函数

#### `_make_dual`

```cpp
// L22-L30: 创建对偶张量（前向模式 AD）
Tensor _make_dual(const Tensor& primal, const Tensor& tangent, int64_t level) {
  TORCH_INTERNAL_ASSERT(
      InferenceMode::is_enabled() && primal.is_inference() && tangent.is_inference(),
      "Expected this function to only be reached in inference mode...");
  return at::alias(primal);  // 返回 pr imal 的视图
}
```

**用途**: 前向模式自动微分中创建对偶张量。

#### `_unpack_dual`

```cpp
// L35-L37: 解包对偶张量
std::tuple<at::Tensor, at::Tensor> _unpack_dual(const at::Tensor& tensor, int64_t level) {
  return std::tuple<at::Tensor, at::Tensor>(
      tensor._fw_primal(level),   // 原始张量
      tensor._fw_grad(level));    // 梯度张量
}
```

#### `_new_zeros_with_same_feature_meta`

```cpp
// L42-L88: 创建与输入具有相同元数据的新张量
Tensor _new_zeros_with_same_feature_meta(
    const at::Tensor& self,
    const at::Tensor& other,
    int64_t self_num_batch_dims) {
  
  // 获取 other 的 shapes/strides
  auto other_sizes = other.sym_sizes();
  auto other_strides = other.sym_strides();
  auto other_storage_offset = other.storage_offset();
  
  // 创建零张量并设置 strides
  auto new_tensor = at::zeros_symint({other_storage_numel}, other.options());
  return new_tensor.as_strided_symint(other_sizes, other_strides, other_storage_offset);
}
```

**用途**: 在反向传播中创建梯度张量。

#### `_lazy_clone`

```cpp
// L94-L107: 惰性克隆（延迟内存分配）
Tensor _lazy_clone(Tensor const& self) {
  // 使用 COW (Copy-On-Write) 语义
  c10::StorageImpl* self_storage = self.storage().unsafeGetStorageImpl();
  c10::intrusive_ptr<c10::StorageImpl> storage =
    c10::impl::cow::lazy_clone_storage(*self_storage);
  
  // 创建新 TensorImpl，共享 Storage
  auto tensor = c10::make_intrusive<c10::TensorImpl>(
      c10::Storage(std::move(storage)),
      self.key_set(),
      self.dtype());
  tensor->set_sizes_and_strides(self.sym_sizes(),
                                self.sym_strides(),
                                self.sym_storage_offset());
  return Tensor(std::move(tensor));
}
```

**用途**: 在反向传播中高效创建张量副本。

---

## 6. 导数公式实战

### 6.1 逐元素运算

```yaml
# sigmoid: σ(x) = 1 / (1 + exp(-x))
# d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
- name: sigmoid(Tensor self) -> Tensor
  self: grad * result * (1 - result)
```

**使用 `result`**:
- `result` 是前向计算的输出
- 避免重复计算 sigmoid(x)

### 6.2 归约运算

```yaml
# sum: 沿维度求和
- name: sum.dim_IntList(Tensor self, int[]? dim=None, bool keepdim=False, ScalarType? dtype=None) -> Tensor
  self: sum_to(self, grad, self.sizes())
```

**`sum_to`**: 将梯度广播回原始形状

### 6.3 视图运算

```yaml
# transpose: 转置
- name: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
  self: transpose_backward(grad, self.sizes(), dim0, dim1)
```

**视图运算特点**:
- 梯度需要反向转置
- 共享内存，无需额外分配

### 6.4 复杂运算

```yaml
# softmax: softmax(x)_i = exp(x_i) / sum(exp(x_j))
# d(softmax)/dx = softmax(x) * (grad - sum(grad * softmax(x)))
- name: _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
  self: _softmax_backward_data(grad, result, dim, self)
```

**使用 `_softmax_backward_data`**:
- 分离的后向函数
- 实现数值稳定的 softmax 反向传播

---

## 7. 自定义算子的 Autograd

### 7.1 方法选择

```
你的算子是否调用其他 at:: 算子？
├── Yes → 使用 CompositeImplicitAutograd（自动支持 autograd）
│
└── No → 需要定义导数公式
    ├── 简单公式 → derivatives.yaml
    └── 复杂逻辑 → 手动实现 Autograd Function
```

### 7.2 derivatives.yaml 方式

**Step 1: 实现前向**
```cpp
// MyOp.cpp
Tensor my_op(const Tensor& self) {
    return self * 2 + 1;  // 简单线性变换
}
```

**Step 2: 添加导数公式**
```yaml
# derivatives.yaml
- name: my_op(Tensor self) -> Tensor
  self: grad * 2
```

### 7.3 手动实现 Autograd Function

当导数公式无法表达复杂逻辑时：

```cpp
// 在 FunctionsManual.cpp 中
Tensor my_op_backward(const Tensor& grad, const Tensor& self) {
    // 复杂逻辑
    auto mask = self > 0;
    return grad * mask.to(grad.scalar_type());
}

// derivatives.yaml
- name: my_op(Tensor self) -> Tensor
  self: my_op_backward(grad, self)
```

---

## 8. 常见错误与排查

### 8.1 错误：Missing derivatives.yaml entry

```
RuntimeError: element 0 of tensors does not require grad

原因: CompositeExplicitAutograd 没有在 derivatives.yaml 定义

解决:
1. 添加 derivatives.yaml 条目，或
2. 改用 CompositeImplicitAutograd
```

### 8.2 错误：Gradient shape mismatch

```
RuntimeError: Mismatch in shape of grad_output

原因: 梯度公式返回的形状与输入不匹配

解决:
# 使用 sum_to 或 expand_as 确保形状一致
self: sum_to(grad, self.sizes())
```

### 8.3 错误：Undefined gradient

```
RuntimeError: Expected to have a gradient for all outputs

原因: 某些输出应该可微但没有定义梯度

解决:
# 添加 output_differentiability
output_differentiability: [True, True]
```

---

## 9. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `derivatives.yaml` | L1-L200 | 导数公式语法说明 |
| `derivatives.yaml` | L221-L223 | abs 导数公式 |
| `derivatives.yaml` | L229-L232 | add.Tensor 导数公式 |
| `derivatives.yaml` | L279-L281 | 多输出示例 |
| `derivatives.yaml` | L394-L396 | ceil (导数为 0) |
| `AutogradComposite.cpp` | L22-L30 | _make_dual 实现 |
| `AutogradComposite.cpp` | L42-L88 | _new_zeros_with_same_feature_meta |
| `AutogradComposite.cpp` | L94-L107 | _lazy_clone 实现 |

---

## 10. 下一步

| Part | 主题 |
|------|------|
| [Part 9](./09-backend-extension.md) | 后端扩展指南 |
| [Part 10](./10-debugging-testing.md) | 调试与测试 |

---

**参考资料**:
- `tools/autograd/derivatives.yaml` - 导数公式完整定义
- `aten/src/ATen/native/AutogradComposite.cpp` - 复合 Autograd 算子
- `torch/csrc/autograd/` - C++ Autograd 引擎
- [PyTorch Autograd 文档](https://pytorch.org/docs/stable/autograd.html)
