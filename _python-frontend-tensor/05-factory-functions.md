# Python 层 Tensor API（五）：工厂函数详解

> **前序**: [Part 4 - 存储与内存管理](./04-storage-memory.md)
> **核心源码**: `torch/csrc/utils/tensor_new.cpp`, `aten/src/ATen/native/TensorFactories.cpp`

---

## 1. 工厂函数分类

PyTorch 的工厂函数可以分为以下几类：

```
┌─────────────────────────────────────────────────────────┐
│                  工厂函数分类                            │
├─────────────────────────────────────────────────────────┤
│  1. 数据创建类                                           │
│     - torch.tensor(), torch.as_tensor()                 │
│     - torch.from_numpy()                                │
│                                                         │
│  2. 内存分配类                                           │
│     - torch.empty(), torch.empty_like()                 │
│     - torch.zeros(), torch.zeros_like()                 │
│     - torch.ones(), torch.ones_like()                   │
│     - torch.full(), torch.full_like()                   │
│                                                         │
│  3. 序列生成类                                           │
│     - torch.arange(), torch.linspace()                  │
│     - torch.logspace(), torch.geomspace()               │
│                                                         │
│  4. 随机数类                                             │
│     - torch.rand(), torch.randn()                       │
│     - torch.randint(), torch.bernoulli()                │
│                                                         │
│  5. 张量变形类                                           │
│     - torch.tensor.new_*() 系列                          │
│                                                         │
│  6. 特殊格式类                                           │
│     - torch.eye(), torch.diag()                         │
│     - torch.tril(), torch.triu()                        │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 数据创建类工厂函数

### 2.1 torch.tensor()

**源码**: `torch/csrc/utils/tensor_new.cpp`

```python
torch.tensor(
    data,               # 输入数据 (列表、元组、numpy 数组等)
    *,
    dtype=None,         # 数据类型 (可选，默认推断)
    device=None,        # 设备 (可选)
    requires_grad=False,# 是否需要梯度
    pin_memory=False,   # 是否使用页锁定内存
) -> Tensor
```

**实现流程**:

```
Python: torch.tensor([1, 2, 3])
        ↓
C++ : new_from_data_copy()
        ↓
internal_new_from_data(
    copy_variables=True,   # 复制变量
    copy_numpy=True,       # 复制 numpy 数据
    type_inference=False   # 不使用类型推断
)
        ↓
┌─────────────────────────────────────────┐
│ 数据类型检查与转换                       │
│ - Tensor → detach() + to()              │
│ - numpy → tensor_from_numpy()           │
│ - __cuda_array_interface__ → 转换       │
│ - __dlpack__ → from_dlpack()            │
│ - Storage → set_()                      │
│ - Sequence → recursive_store()          │
└─────────────────────────────────────────┘
        ↓
at::empty() 分配内存
        ↓
recursive_store() 递归填充数据
        ↓
tensor.to(device, dtype) 移动到目标设备
        ↓
at::lift_fresh() 提升到当前上下文
```

**数据类型推断规则**:

```python
# Python 标量类型映射
0.0      → torch.float32  (默认浮点类型)
0       → torch.int64
True    → torch.bool
1+2j    → torch.complex64 (默认复数类型)

# 序列类型提升规则
[1, 2.0]     → torch.float32  (int + float → float)
[1, 2+3j]    → torch.complex64 (int + complex → complex)
[1.0, 2+3j]  → torch.complex64 (float + complex → complex)

# NumPy 数组
np.array([1, 2], dtype=np.int32) → torch.int32
```

### 2.2 torch.as_tensor()

```python
torch.as_tensor(
    data,
    dtype=None,
    device=None
) -> Tensor
```

**与 torch.tensor() 的区别**:

| 特性 | `torch.tensor()` | `torch.as_tensor()` |
|------|-----------------|--------------------|
| 数据拷贝 | 总是拷贝 | 共享内存 (如果可能) |
| NumPy 互操作 | 拷贝数据 | 零拷贝 (共享内存) |
| 梯度追踪 | 可以设置 requires_grad | 不支持 requires_grad |

**示例**:

```python
import numpy as np

# numpy 数组
np_arr = np.array([1, 2, 3])

# torch.tensor() - 拷贝数据
t1 = torch.tensor(np_arr)
np_arr[0] = 100
print(t1)  # tensor([1, 2, 3]) - 不受影响

# torch.as_tensor() - 共享内存
t2 = torch.as_tensor(np_arr)
np_arr[0] = 100
print(t2)  # tensor([100, 2, 3]) - 受影响
```

### 2.3 torch.from_numpy()

```python
torch.from_numpy(ndarray) -> Tensor
```

**特点**:
- 与 NumPy 数组共享内存
- 只能用于 CPU 张量
- 修改一个会影响另一个

**源码**: `torch/csrc/utils/tensor_numpy.cpp`

```cpp
// 从 NumPy 创建 Tensor
Tensor tensor_from_numpy(PyObject* data, bool warn_if_not_writeable) {
  // 获取 NumPy 数组信息
  auto array = reinterpret_cast<PyArrayObject*>(data);
  auto dtype = numpy_dtype_to_aten(PyArray_TYPE(array));
  auto sizes = get_sizes(array);

  // 获取数据指针
  void* data_ptr = PyArray_DATA(array);

  // 创建 Deleter，确保 NumPy 数组在 Tensor 销毁时被正确释放
  py::object obj = py::reinterpret_borrow<py::object>((PyObject*)array);
  auto deleter = [obj = std::move(obj)](void*) {};

  // 创建 Storage
  auto storage = at::Storage(
      at::Storage::use_byte_size_t(),
      PyArray_NBYTES(array),
      at::DataPtr(
          data_ptr,
          deleter,
          at::Device(kCPU),
          nullptr),
      nullptr,  // allocator
      false);   // resizable

  // 创建 Tensor
  return at::from_storage(storage, sizes, dtype);
}
```

---

## 3. 内存分配类工厂函数

### 3.1 torch.empty()

```python
torch.empty(
    size,               # 形状 (整数或整数元组)
    *,
    dtype=None,
    device=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format
) -> Tensor
```

**C++ 实现**:

```cpp
// aten/src/ATen/native/TensorFactories.cpp
Tensor empty_symint(
    SymIntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> memory_format = c10::nullopt) {

  return at::native::empty_symint(
      size,
      options.layout(),
      options.device(),
      options.dtype(),
      memory_format.value_or(MemoryFormat::Contiguous));
}

// CPU 实现
Tensor empty_strided_cpu(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {

  // 1. 分配内存
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      c10::compute_numel(size) * elementSize(options.dtype()),
      allocator,  // CPU 分配器
      options.device());

  // 2. 创建 Tensor
  return at::native::empty_strided(
      size, stride, options.dtype(), options.device(), storage);
}
```

**特点**:
- 不初始化内存内容 (包含随机数据)
- 最快的分配方式
- 适用于稍后填充数据的场景

### 3.2 torch.zeros() / torch.ones()

```python
torch.zeros(size, *, dtype=None, device=None, ...) -> Tensor
torch.ones(size, *, dtype=None, device=None, ...) -> Tensor
```

**实现**:

```cpp
// zeros 实现
Tensor zeros_symint(SymIntArrayRef size, const TensorOptions& options) {
  // 先分配未初始化内存
  Tensor tensor = at::empty_symint(size, options);
  // 原地填充 0
  tensor.zero_();
  return tensor;
}

// ones 实现
Tensor ones_symint(SymIntArrayRef size, const TensorOptions& options) {
  Tensor tensor = at::empty_symint(size, options);
  tensor.fill_(1);  // 原地填充 1
  return tensor;
}
```

**_like 变体**:

```python
torch.zeros_like(input, *, dtype=None, device=None, ...)
torch.ones_like(input, *, dtype=None, device=None, ...)
```

从输入 Tensor 派生属性:

```cpp
Tensor zeros_like(const Tensor& self, const TensorOptions& options) {
  return at::zeros(
      self.sizes(),
      options.dtype(optional_dtype(self.dtype(), options.dtype()))
             .device(or_device(self.device(), options.device()))
             .layout(self.layout())
             .pinned_memory(options.pinned_memory()));
}
```

### 3.3 torch.full()

```python
torch.full(
    size,
    fill_value,         # 填充值
    *,
    dtype=None,
    device=None,
    ...
) -> Tensor
```

**实现**:

```cpp
Tensor full_symint(
    SymIntArrayRef size,
    const Scalar& fill_value,
    const TensorOptions& options) {

  Tensor tensor = at::empty_symint(size, options);
  tensor.fill_(fill_value);
  return tensor;
}
```

---

## 4. 序列生成类工厂函数

### 4.1 torch.arange()

```python
torch.arange(
    start=0,
    end=None,
    step=1,
    *,
    dtype=None,
    device=None,
    requires_grad=False
) -> Tensor
```

**C++ 实现**:

```cpp
// aten/src/ATen/native/TensorFactories.cpp
Tensor arange_start_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {

  // 1. 计算输出大小
  int64_t size = ceil((end - start) / step);

  // 2. 分配内存
  out.resize_({size});

  // 3. 填充等差数列
  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      out.scalar_type(), "arange", [&] {
        auto* out_data = out.data_ptr<scalar_t>();
        scalar_t val = start.to<scalar_t>();
        scalar_t step_val = step.to<scalar_t>();

        for (int64_t i = 0; i < size; i++) {
          out_data[i] = val;
          val += step_val;
        }
      });

  return out;
}
```

**示例**:

```python
torch.arange(5)           # tensor([0, 1, 2, 3, 4])
torch.arange(2, 8)        # tensor([2, 3, 4, 5, 6, 7])
torch.arange(0, 10, 2)    # tensor([0, 2, 4, 6, 8])
torch.arange(0.5, 2, 0.5) # tensor([0.5, 1.0, 1.5])
```

### 4.2 torch.linspace()

```python
torch.linspace(
    start,
    end,
    steps=100,
    *,
    dtype=None,
    device=None
) -> Tensor
```

**实现**:

```cpp
Tensor linspace(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    const TensorOptions& options) {

  Tensor result = at::empty({steps}, options);

  if (steps > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Double, ScalarType::BFloat16,
        result.scalar_type(), "linspace", [&] {
          scalar_t start_val = start.to<scalar_t>();
          scalar_t end_val = end.to<scalar_t>();
          scalar_t step = (end_val - start_val) / (steps - 1);

          for (int64_t i = 0; i < steps; i++) {
            result.data_ptr<scalar_t>()[i] = start_val + i * step;
          }
        });
  }

  return result;
}
```

**示例**:

```python
torch.linspace(0, 1, steps=5)
# tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

---

## 5. Tensor.new_*() 方法族

### 5.1 概述

`new_*` 方法基于调用 Tensor 的 dtype 和 device 创建新 Tensor：

```python
x = torch.randn(3, 4, dtype=torch.float64, device='cuda:0')

# 基于 x 的属性创建新 Tensor
y = x.new_ones(2, 3)       # dtype=torch.float64, device='cuda:0'
z = x.new_zeros(4, 5)      # dtype=torch.float64, device='cuda:0'
w = x.new_empty((6, 7))    # dtype=torch.float64, device='cuda:0'

# 可以覆盖默认属性
a = x.new_ones(2, 3, dtype=torch.float32)  # dtype=torch.float32, device='cuda:0'
```

### 5.2 C++ 绑定

**源码**: `torch/csrc/Module.cpp`

```cpp
// Tensor 类的方法绑定
m.def(TensorType, "new_empty", &new_empty);
m.def(TensorType, "new_ones", &new_ones);
m.def(TensorType, "new_zeros", &new_zeros);
m.def(TensorType, "new_full", &new_full);
m.def(TensorType, "new_tensor", &new_tensor);
```

### 5.3 实现

```cpp
// Tensor.new_*() 的通用实现模式
Tensor Tensor_new_ones(
    const Tensor& self,
    IntArrayRef size,
    const TensorOptions& options) {

  // 从 self 派生 options
  auto derived_options = self.options()
      .dtype(options.dtype().value_or(self.dtype()))
      .device(options.device().value_or(self.device()))
      .layout(self.layout())
      .pinned_memory(options.pinned_memory());

  return at::ones(size, derived_options);
}
```

---

## 6. 特殊格式工厂函数

### 6.1 torch.eye()

```python
torch.eye(
    n,              # 行数
    m=None,         # 列数 (默认等于 n)
    *,
    dtype=None,
    device=None
) -> Tensor
```

**实现**:

```cpp
Tensor eye(
    int64_t n,
    c10::optional<int64_t> m,
    const TensorOptions& options) {

  int64_t rows = n;
  int64_t cols = m.value_or(n);

  Tensor result = at::zeros({rows, cols}, options);

  // 填充对角线
  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      result.scalar_type(), "eye", [&] {
        int64_t diagonal_size = std::min(rows, cols);
        scalar_t* result_data = result.data_ptr<scalar_t>();
        int64_t stride = result.stride(0) + result.stride(1) + 1;

        for (int64_t i = 0; i < diagonal_size; i++) {
          result_data[i * stride] = scalar_t(1);
        }
      });

  return result;
}
```

### 6.2 torch.diag()

```python
# 从对角线创建矩阵
torch.diag(diagonal, diagonal_index=0) -> Tensor

# 从矩阵提取对角线
torch.diag(input, diagonal_index=0) -> Tensor
```

**实现**:

```cpp
Tensor diagflat(const Tensor& diagonal, int64_t diagonal_index) {
  // 计算矩阵大小
  int64_t n = diagonal.size(0) + std::abs(diagonal_index);

  Tensor result = at::zeros({n, n}, diagonal.options());

  // 填充对角线
  result.diagonal(diagonal_index).copy_(diagonal);

  return result;
}
```

---

## 7. 工厂函数的 DispatchKey 处理

### 7.1 DispatchKey 排除

工厂函数需要排除某些 DispatchKey 以确保正确的行为：

```cpp
Tensor empty_with_dispatch(
    IntArrayRef size,
    const TensorOptions& options) {

  // 排除以下 DispatchKey
  at::AutoDispatchBelowADInplaceOrView guard;  // 排除 Autograd
  c10::impl::ExcludeDispatchKeyGuard python_guard(
      c10::DispatchKey::Python);  // 排除 Python
  c10::impl::ExcludeDispatchKeyGuard fake_guard(
      c10::DispatchKey::Fake);    // 排除 Fake

  // 实际分配
  return at::native::empty_strided(
      size,
      compute_strides(size, MemoryFormat::Contiguous),
      options.dtype(),
      options.device());
}
```

### 7.2 lift_fresh()

```cpp
// 将 Tensor 提升到当前上下文
Tensor lift_fresh(const Tensor& tensor) {
  // 处理 functorch 包装
  // 处理 functionalization
  // 处理 FakeTensor
  return at::lift_fresh_copy(tensor);
}
```

---

## 8. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `torch/csrc/utils/tensor_new.cpp` | L265-L489 | internal_new_from_data() |
| `torch/csrc/utils/tensor_new.cpp` | L203-L263 | recursive_store() |
| `torch/csrc/utils/tensor_new.cpp` | L123-L200 | infer_scalar_type() |
| `aten/src/ATen/native/TensorFactories.cpp` | - | empty/zeros/ones 实现 |
| `aten/src/ATen/native/TensorFactories.cpp` | - | arange/linspace 实现 |

---

## 9. 下一步

| 章节 | 主题 |
|------|------|
| [Part 6](./06-dispatcher.md) | 分发机制 |

---

**参考资料**:
- `torch/csrc/utils/tensor_new.cpp` - Tensor 创建核心实现
- `aten/src/ATen/native/TensorFactories.cpp` - ATen 工厂函数
