# Python 层 Tensor API（二）：Tensor 构造与工厂函数

> **前序**: [Part 1 - Tensor 类结构](./01-tensor-class-structure.md)
> **核心源码**: `torch/csrc/utils/tensor_new.cpp`, `torch/_VF.py`, `torch/functional.py`

---

## 1. Tensor 构造机制概览

### 1.1 构造方式分类

PyTorch 提供多种 Tensor 构造方式：

```
┌─────────────────────────────────────────────────────────┐
│              Tensor 构造方式                              │
├─────────────────────────────────────────────────────────┤
│  1. torch.tensor()      - 从数据创建                     │
│  2. torch.empty()       - 未初始化内存                   │
│  3. torch.zeros()/ones() - 填充特定值                   │
│  4. torch.arange()      - 等差数列                       │
│  5. Tensor.new_*()      - 基于现有 Tensor 的属性创建      │
│  6. torch.Storage()     - 从 Storage 创建                │
└─────────────────────────────────────────────────────────┘
```

### 1.2 构造流程

```
Python: torch.tensor(data, dtype=None, device=None)
        ↓
C++ : internal_new_from_data()
        ↓
┌─────────────────────────────────────────────────────────┐
│  数据类型判断：                                          │
│  - Tensor → to() 转换                                   │
│  - numpy.ndarray → tensor_from_numpy()                  │
│  - __cuda_array_interface__ → tensor_from_cuda_array()  │
│  - __dlpack__ → from_dlpack()                           │
│  - Storage → set_()                                     │
│  - Sequence → recursive_store()                         │
└─────────────────────────────────────────────────────────┘
        ↓
DispatchKey 排除（防止追踪和包装）
        ↓
at::empty() 分配内存
        ↓
recursive_store() 递归填充数据
        ↓
tensor.to(device, dtype) 移动到目标设备
        ↓
at::lift_fresh() 提升到当前变换上下文
```

---

## 2. C++ 层 Tensor 创建

### 2.1 internal_new_from_data()

**源码**: `torch/csrc/utils/tensor_new.cpp` (L265-L489)

这是最核心的数据创建函数，处理所有类型的数据输入：

```cpp
Tensor internal_new_from_data(
    c10::TensorOptions options,
    at::ScalarType scalar_type,
    std::optional<Device> device_opt,
    PyObject* data,
    bool copy_variables,
    bool copy_numpy,
    bool type_inference,
    bool pin_memory = false) {

  // 1. 检查字符串类型（不支持）
  TORCH_CHECK_TYPE(
      !THPUtils_checkString(data),
      "new(): invalid data type '",
      Py_TYPE(data)->tp_name,
      "'");

  // 2. 处理 Tensor 输入
  if (THPVariable_Check(data)) {
    auto var = THPVariable_Unpack(data);
    if (copy_variables) {
      var = var.detach();
    }
    const auto& inferred_scalar_type =
        type_inference ? var.scalar_type() : scalar_type;
    auto device = device_opt.has_value() ? *device_opt : var.device();
    return var.to(device, inferred_scalar_type, /*non_blocking=*/false,
                   /*copy=*/copy_variables);
  }

  // 3. 处理 CUDA Array Interface
  if (PyObject_HasAttrString(data, "__cuda_array_interface__")) {
    auto tensor = tensor_from_cuda_array_interface(data, device_opt);
    // ... 转换逻辑
  }

  // 4. 处理 NumPy Array
  if (is_numpy_available() && PyArray_Check(data)) {
    auto tensor = tensor_from_numpy(data, /*warn_if_not_writeable=*/!copy_numpy);
    // ... 转换逻辑
  }

  // 5. 处理 DLPack
  if (PyObject_HasAttrString(data, "__dlpack__")) {
    py::object tensor_o =
        py::module::import("torch").attr("utils").attr("dlpack")
          .attr("from_dlpack")(py::handle(data));
    // ... 转换逻辑
  }

  // 6. 处理 Storage
  if (isStorage(data)) {
    auto [storage, storage_scalar_type, is_typed_storage] =
        createStorageGetType(data);
    tensor = at::empty({0}, opts.device(storage.device()));
    tensor.set_(storage);
  }

  // 7. 处理序列数据
  auto sizes = compute_sizes(data, scalar_type);
  ScalarType inferred_scalar_type =
      type_inference ? infer_scalar_type(data) : scalar_type;

  // 8. 分配内存（排除各种 DispatchKey）
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    c10::impl::ExcludeDispatchKeyGuard torchdispatchmode_guard(
        c10::DispatchKey::Python);
    // ... 排除更多 DispatchKey
    tensor = at::empty(sizes, opts.pinned_memory(pin_memory));

    // 9. 递归填充数据
    if (c10::multiply_integers(tensor.sizes()) != 0) {
      recursive_store(
          (char*)tensor.data_ptr(),
          tensor.sizes(),
          tensor.strides(),
          0,
          inferred_scalar_type,
          tensor.dtype().itemsize(),
          data);
    }
  }

  // 10. 移动到目标设备
  tensor = tensor.to(device, inferred_scalar_type, /*non_blocking=*/false,
                     /*copy=*/false);

  // 11. Lift 到当前上下文
  tensor = at::lift_fresh(tensor);

  return tensor;
}
```

### 2.2 recursive_store() - 递归数据填充

**源码**: `torch/csrc/utils/tensor_new.cpp` (L203-L263)

```cpp
void recursive_store(
    char* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    int64_t dim,
    ScalarType scalarType,
    size_t elementSize,
    PyObject* obj) {

  int64_t ndim = static_cast<int64_t>(sizes.size());

  // 递归终止条件：到达最内层维度
  if (dim == ndim) {
    torch::utils::store_scalar(data, scalarType, obj);
    return;
  }

  auto n = sizes[dim];
  auto seq = THPObjectPtr(PySequence_Fast(obj, "not a sequence"));
  auto seq_size = PySequence_Fast_GET_SIZE(seq.get());

  TORCH_CHECK_VALUE(
      seq_size == n,
      "expected sequence of length ", n, " at dim ", dim,
      " (got ", seq_size, ")");

  PyObject** items = PySequence_Fast_ITEMS(seq.get());
  for (const auto i : c10::irange(n)) {
    // 递归填充下一维度
    recursive_store(
        data, sizes, strides, dim + 1,
        scalarType, elementSize, items[i]);

    // 移动到下一个元素
    data += strides[dim] * elementSize;
  }
}
```

**示意图**:

```
输入：[[1, 2], [3, 4]]
sizes = [2, 2], strides = [2, 1]

dim=0: 处理外层列表
  ├─ i=0: items[0] = [1, 2]
  │   dim=1: 处理内层列表
  │     ├─ i=0: store_scalar(data_ptr, 1)  // data += 2*4 = 8
  │     └─ i=1: store_scalar(data_ptr, 2)
  │
  └─ i=1: items[1] = [3, 4]
      dim=1: 处理内层列表
        ├─ i=0: store_scalar(data_ptr, 3)  // data += 2*4 = 8
        └─ i=1: store_scalar(data_ptr, 4)
```

### 2.3 infer_scalar_type() - 数据类型推断

**源码**: `torch/csrc/utils/tensor_new.cpp` (L123-L200)

```cpp
ScalarType infer_scalar_type(PyObject* obj) {
  // SymInt/SymFloat
  if (torch::is_symint(obj)) return ScalarType::Long;
  if (torch::is_symfloat(obj)) return get_default_scalar_type();

  // NumPy
#ifdef USE_NUMPY
  if (is_numpy_available()) {
    if (PyArray_Check(obj))
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)obj));
    if (PyArray_CheckScalar(obj)) {
      THPObjectPtr arr(PyArray_FromScalar(obj, nullptr));
      return numpy_dtype_to_aten(PyArray_TYPE((PyArrayObject*)arr.get()));
    }
  }
#endif

  // Python 标量
  if (PyFloat_Check(obj))
    return get_default_scalar_type();  // 默认 float32
  if (THPUtils_checkLong(obj))
    return ScalarType::Long;
  if (PyBool_Check(obj))
    return ScalarType::Bool;
  if (PyComplex_Check(obj)) {
    switch (get_default_scalar_type()) {
      case ScalarType::Float: return ScalarType::ComplexFloat;
      case ScalarType::Double: return ScalarType::ComplexDouble;
      case ScalarType::Half: return ScalarType::ComplexHalf;
    }
  }

  // Tensor
  if (THPVariable_Check(obj))
    return THPVariable_Unpack(obj).scalar_type();

  // 序列 - 递归推断并提升类型
  if (PySequence_Check(obj)) {
    auto length = PySequence_Length(obj);
    if (length == 0)
      return get_default_scalar_type();

    ScalarType scalarType{};
    for (const auto i : c10::irange(length)) {
      THPObjectPtr handle(PySequence_GetItem(obj, i));
      ScalarType item_scalarType = infer_scalar_type(handle);
      scalarType = (i > 0) ? at::promoteTypes(scalarType, item_scalarType)
                           : item_scalarType;
      if (scalarType == ScalarType::ComplexDouble)
        return scalarType;  // 提前终止
    }
    return scalarType;
  }

  TORCH_CHECK(false, "Could not infer dtype of ", Py_TYPE(obj)->tp_name);
}
```

### 2.4 DispatchKey 排除机制

在创建 Tensor 时，需要排除某些 DispatchKey 以防止不必要的包装：

```cpp
{
  at::AutoDispatchBelowADInplaceOrView guard;

  // 排除 Python 分发
  c10::impl::ExcludeDispatchKeyGuard torchdispatchmode_guard(
      c10::DispatchKey::Python);
  c10::impl::ExcludeDispatchKeyGuard torchdispatchmode_snapshot_guard(
      c10::DispatchKey::PythonTLSSnapshot);

  // 排除 functorch
  c10::impl::ExcludeDispatchKeyGuard functorch_front_guard(
      c10::DispatchKey::FuncTorchDynamicLayerFrontMode);
  c10::impl::ExcludeDispatchKeyGuard functorch_back_guard(
      c10::DispatchKey::FuncTorchDynamicLayerBackMode);

  // 排除 Fake Tensor 和延迟初始化
  c10::impl::ExcludeDispatchKeyGuard fake_and_deferred_init_guard(
      c10::DispatchKeySet{
          c10::DispatchKey::Fake,
          c10::DispatchKey::DeferredInit
      });

  // 排除函数化
  c10::impl::ExcludeDispatchKeyGuard functionalize_guard(
      c10::DispatchKey::Functionalize);

  // 禁用追踪器
  at::tracer::impl::NoTracerDispatchMode tracer_guard;

  // 分配内存
  tensor = at::empty(sizes, opts.pinned_memory(pin_memory));
}
```

**排除原因**:

| DispatchKey | 排除原因 |
|-------------|----------|
| `Python` | 防止递归调用 `__torch_dispatch__` |
| `FuncTorchDynamicLayer*` | functorch 包装在 lift_fresh 时处理 |
| `Fake` | 防止创建 FakeTensor |
| `Functionalize` | 函数化在 lift_fresh 时处理 |
| `Autograd*` | 构造阶段不需要 autograd |

---

## 3. Python 层工厂函数

### 3.1 torch.tensor()

**源码**: `torch/_VF.py` (由 C++ 绑定)

```python
# Python 调用示例
t = torch.tensor(
    data,           # 输入数据
    dtype=None,     # 数据类型（推断）
    device=None,    # 设备
    requires_grad=False,  # 是否需要梯度
    pin_memory=False,     # 是否使用页锁定内存
)
```

**C++ 绑定**:

```cpp
// torch/csrc/Module.cpp 或相关绑定文件
m.def("tensor", &new_from_data_copy,
      "Create a tensor from data");
```

### 3.2 torch.empty() 系列

**源码**: 通过 ATen 原生函数绑定

```python
# 基础用法
torch.empty(size, *, dtype=None, device=None, layout=None, requires_grad=False)
torch.empty_like(input, *, dtype=None, device=None, layout=None, requires_grad=False)
torch.empty_strided(size, strides, *, dtype=None, device=None, requires_grad=False)
torch.empty_quantized(size, qtensor, *, dtype=None, device=None, requires_grad=False)
```

**C++ 实现**:

```cpp
// aten/src/ATen/native/TensorFactories.cpp
Tensor empty_symint(
    SymIntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> memory_format = c10::nullopt) {

  return at::native::empty_symint(
      size, options.layout(), options.device(),
      options.dtype(), memory_format.value_or(MemoryFormat::Contiguous));
}
```

### 3.3 torch.zeros() / torch.ones()

```python
# 创建全零 Tensor
torch.zeros(size, *, dtype=None, device=None, layout=None, requires_grad=False)
torch.zeros_like(input, *, dtype=None, device=None, layout=None, requires_grad=False)

# 创建全一 Tensor
torch.ones(size, *, dtype=None, device=None, layout=None, requires_grad=False)
torch.ones_like(input, *, dtype=None, device=None, layout=None, requires_grad=False)
```

**实现机制**:

```cpp
// aten/src/ATen/native/TensorFactories.cpp
Tensor zeros_symint(SymIntArrayRef size, const TensorOptions& options) {
  Tensor tensor = at::empty_symint(size, options);
  tensor.zero_();  // 原地填充 0
  return tensor;
}

Tensor ones_symint(SymIntArrayRef size, const TensorOptions& options) {
  Tensor tensor = at::empty_symint(size, options);
  tensor.fill_(1);  // 原地填充 1
  return tensor;
}
```

### 3.4 torch.arange()

```python
torch.arange(
    start=0,      # 起始值
    end=None,     # 结束值（不包含）
    step=1,       # 步长
    *,
    dtype=None,   # 数据类型
    device=None,  # 设备
    requires_grad=False
)
```

**实现**:

```cpp
// aten/src/ATen/native/TensorFactories.cpp
Tensor arange_start_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {

  // 计算输出大小
  int64_t size = ceil((end - start) / step);

  // 分配内存
  out.resize_({size});

  // 填充等差数列
  AT_DISPATCH_ALL_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16,
      out.scalar_type(), "arange", [&] {
        auto* out_data = out.data_ptr<scalar_t>();
        scalar_t val = start.to<scalar_t>();
        for (int64_t i = 0; i < size; i++) {
          out_data[i] = val;
          val += step.to<scalar_t>();
        }
      });

  return out;
}
```

---

## 4. Tensor.new_*() 方法

### 4.1 new() 方法族

`Tensor.new_*()` 方法基于调用 Tensor 的 dtype 和 device 创建新 Tensor：

```python
x = torch.randn(3, 4, dtype=torch.float64, device='cuda:0')

# 基于 x 的属性创建新 Tensor
y = x.new_ones(2, 3)       # dtype=torch.float64, device='cuda:0'
z = x.new_zeros(4, 5)      # dtype=torch.float64, device='cuda:0'
w = x.new_empty((6, 7))    # dtype=torch.float64, device='cuda:0'

# 可以覆盖默认属性
a = x.new_ones(2, 3, dtype=torch.float32)  # dtype=torch.float32, device='cuda:0'
```

### 4.2 C++ 实现

**源码**: `torch/csrc/utils/tensor_new.cpp` (L526-L600)

```cpp
// "base" here refers to the Tensor type on which the function was invoked,
// e.g.: in x.new(y), 'x' is the base.

void check_base_legacy_new(
    c10::DispatchKey dispatch_key,
    at::Layout expected_layout) {

  if (expected_layout == c10::kStrided) {
    constexpr c10::DispatchKeySet expected_key_set({
        c10::DispatchKey::CPU,
        c10::DispatchKey::CUDA,
        c10::DispatchKey::HIP,
        c10::DispatchKey::XLA,
        c10::DispatchKey::Lazy,
        c10::DispatchKey::IPU,
        c10::DispatchKey::XPU,
        c10::DispatchKey::HPU,
        c10::DispatchKey::MPS,
        c10::DispatchKey::Meta,
        c10::DispatchKey::PrivateUse1,
    });
    TORCH_CHECK(
        expected_key_set.has(dispatch_key),
        "new(): expected key in ", expected_key_set,
        " but got: ", dispatch_key);
  }
  // ... sparse layout 检查
}

std::optional<Device> device_or_from_dispatch_key(
    std::optional<Device> device,
    c10::DispatchKey dispatch_key) {
  if (device.has_value()) {
    return device;
  } else {
    return Device(dispatchKeyToDeviceType(dispatch_key));
  }
}
```

### 4.3 Python 绑定

`new_*` 方法在 `torch/_tensor.py` 中没有显式定义，它们通过 C++ 直接绑定到 Tensor 类：

```python
# 以下方法由 C++ 直接绑定:
# - Tensor.new()
# - Tensor.new_empty()
# - Tensor.new_ones()
# - Tensor.new_zeros()
# - Tensor.new_full()
# - Tensor.new_tensor()
```

---

## 5. Storage 创建

### 5.1 Storage 构造

```python
# 创建 Storage
storage = torch.FloatStorage(100)  # CPU
storage = torch.cuda.FloatStorage(100)  # CUDA

# 从数据创建
storage = torch.FloatStorage.from_buffer(buffer)

# 共享内存
storage.share_memory_()
```

### 5.2 Tensor 从 Storage 创建

```python
# 从 Storage 创建 Tensor
storage = torch.FloatStorage(100)
tensor = torch.FloatTensor(storage)

# 指定偏移量和形状
tensor = torch.FloatTensor(storage, offset=10, size=(3, 4))

# set_ 方法
tensor = torch.empty(1)
tensor.set_(storage, offset=0, size=(10,), stride=(1,))
```

### 5.3 Storage 类型

PyTorch 为每种数据类型提供对应的 Storage 类型：

```python
# CPU Storage
torch.FloatStorage
torch.DoubleStorage
torch.IntStorage
torch.LongStorage
torch.ShortStorage
torch.ByteStorage
torch.BoolStorage
torch.HalfStorage
torch.BFloat16Storage
torch.ComplexFloatStorage
torch.ComplexDoubleStorage

# CUDA Storage
torch.cuda.FloatStorage
torch.cuda.DoubleStorage
# ... 其他 CUDA 类型
```

---

## 6. 特殊 Tensor 创建

### 6.1 量化 Tensor

```python
# 创建量化 Tensor
torch.qint8 = torch.quantize_per_tensor(
    tensor,      # 浮点 Tensor
    scale,       # 缩放因子
    zero_point,  # 零点
    dtype=torch.qint8
)

# 从量化参数创建
torch._utils._rebuild_qtensor(
    storage,
    storage_offset,
    size,
    stride,
    quantizer_params,  # (qscheme, scale, zero_point) 或 (qscheme, scales, zero_points, axis)
    requires_grad,
    backward_hooks
)
```

### 6.2 稀疏 Tensor

```python
# COO 格式稀疏 Tensor
torch.sparse_coo_tensor(
    indices,     # 索引 (2, nnz)
    values,      # 值 (nnz,)
    size,        # 形状
    device=None,
    requires_grad=False
)

# CSR 格式
torch.sparse_csr_tensor(
    crow_indices,  # 行指针
    col_indices,   # 列索引
    values,        # 值
    size
)
```

### 6.3 嵌套 Tensor

```python
# 嵌套 Tensor (用于变长序列)
torch.nested.nested_tensor(
    tensors,     # Tensor 列表
    dtype=None,
    device=None
)
```

### 6.4 Meta Tensor

```python
# Meta Tensor (只有 metadata，没有实际数据)
tensor = torch.empty((3, 4), device='meta')

# 用途：
# - 延迟分配 (lazy allocation)
# - 形状推断 (shape inference)
# - 内存规划 (memory planning)
```

---

## 7. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `torch/csrc/utils/tensor_new.cpp` | L203-L263 | recursive_store() |
| `torch/csrc/utils/tensor_new.cpp` | L123-L200 | infer_scalar_type() |
| `torch/csrc/utils/tensor_new.cpp` | L265-L489 | internal_new_from_data() |
| `torch/csrc/utils/tensor_new.cpp` | L491-L504 | new_from_data_copy() |
| `torch/csrc/utils/tensor_new.cpp` | L506-L524 | legacy_new_from_sequence() |
| `aten/src/ATen/native/TensorFactories.cpp` | - | empty/zeros/ones/arange 实现 |

---

## 8. 下一步

| 章节 | 主题 |
|------|------|
| [Part 3](./03-autograd.md) | 自动微分集成 |
| [Part 4](./04-storage-memory.md) | 存储与内存管理 |
| [Part 5](./05-factory-functions.md) | 工厂函数详解 |

---

**参考资料**:
- `torch/csrc/utils/tensor_new.cpp` - Tensor 创建核心实现
- `aten/src/ATen/native/TensorFactories.cpp` - ATen 工厂函数
- `torch/_VF.py` - Python 绑定函数
