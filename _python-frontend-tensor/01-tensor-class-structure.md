# Python 层 Tensor API（一）：Tensor 类结构

> **前序**: [ATen Part 7 - Tensor 核心数据结构](../_aten/07-tensor-structure.md)
> **核心源码**: `torch/_tensor.py`, `torch/csrc/tensor/python_tensor.cpp`, `torch/csrc/autograd/python_variable.cpp`

---

## 1. 整体架构

### 1.1 Python Tensor 继承层次

```
┌─────────────────────────────────────────────────────────┐
│              Python 层 (torch/_tensor.py)                │
│                    class Tensor                          │
│                   inherits from                          │
│              torch._C.TensorBase                         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              C++ 扩展层 (torch._C)                       │
│                  TensorBase 类                           │
│              (THPVariable 结构)                          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              C++ 核心层 (ATen)                           │
│                 at::Tensor                               │
│              (TensorImpl 指针)                           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              底层实现 (c10)                              │
│              TensorImpl + Storage                        │
└─────────────────────────────────────────────────────────┘
```

### 1.2 关键文件职责

| 文件 | 职责 |
|------|------|
| `torch/_tensor.py` | Python 层 Tensor 类定义，包含高级方法 |
| `torch/csrc/tensor/python_tensor.h` | Python Tensor 类型绑定声明 |
| `torch/csrc/tensor/python_tensor.cpp` | Python Tensor 类型注册和初始化 |
| `torch/csrc/autograd/python_variable.h` | THPVariable 结构定义 |
| `torch/csrc/autograd/python_variable.cpp` | Variable 与 Python 的绑定 |
| `torch/_C/__init__.pyi.in` | TensorBase 类型注解 |

---

## 2. Python Tensor 类定义

### 2.1 类继承关系

**源码**: `torch/_tensor.py` (L110)

```python
class Tensor(torch._C.TensorBase):
    _is_param: bool
    __dlpack_c_exchange_api__: object = torch._C._dlpack_exchange_api()
```

**关键点**:
- `Tensor` 继承自 `torch._C.TensorBase`
- `TensorBase` 是 C++ 扩展类，包含核心属性和方法
- Python 层 `Tensor` 添加了高级方法和 `__torch_function__` 支持

### 2.2 TensorBase 属性

**源码**: `torch/_C/__init__.pyi.in` (L1953-L1980)

```python
class TensorBase(metaclass=_TensorMeta):
    # ===== 自动微分相关 =====
    requires_grad: _bool          # 是否需要梯度
    retains_grad: _bool           # 是否保留梯度
    grad: Tensor | None           # 梯度
    _backward_hooks: OrderedDict  # 反向传播钩子
    _post_accumulate_grad_hooks: OrderedDict  # 梯度累积后钩子

    # ===== 元数据 =====
    shape: Size                   # 形状
    names: list[str]              # 命名维度
    device: _device               # 设备
    dtype: _dtype                 # 数据类型
    layout: _layout               # 内存布局
    ndim: _int                    # 维度数

    # ===== 存储相关 =====
    data: Tensor                  # 底层数据 (已废弃)
    _version: _int                # 版本计数器
    _base: Tensor | None          # 基础 Tensor (视图操作)
    _cdata: _int                  # C++ 指针地址

    # ===== 特殊属性 =====
    output_nr: _int               # 输出序号
    grad_fn: Node | None          # 梯度函数节点
    is_leaf: _bool                # 是否为叶节点

    # ===== 转置属性 =====
    T: Tensor                     # 2D 转置
    H: Tensor                     # 共轭转置
    real: Tensor                  # 实部
    imag: Tensor                  # 虚部
    mT: Tensor                    # 批量矩阵转置
    mH: Tensor                    # 批量共轭转置
```

---

## 3. Python 层核心方法

### 3.1 自动微分方法

#### backward()

**源码**: `torch/_tensor.py` (L576-L633)

```python
def backward(
    self,
    gradient=None,
    retain_graph=None,
    create_graph=False,
    inputs=None
):
    """Computes the gradient of current tensor wrt graph leaves."""
    if has_torch_function_unary(self):
        return handle_torch_function(
            Tensor.backward, (self,), self,
            gradient=gradient, retain_graph=retain_graph,
            create_graph=create_graph, inputs=inputs
        )
    torch.autograd.backward(
        self, gradient, retain_graph, create_graph, inputs=inputs
    )
```

**功能**:
- 计算当前张量相对于图叶节点的梯度
- 委托给 `torch.autograd.backward` 实现

#### register_hook()

**源码**: `torch/_tensor.py` (L663-L711)

```python
def register_hook(self, hook):
    """Registers a backward hook.

    hook(grad) -> Tensor or None
    """
    if not self.requires_grad:
        raise RuntimeError(
            "cannot register a hook on a tensor that doesn't require gradient"
        )
    if self._backward_hooks is None:
        self._backward_hooks = OrderedDict()
        if self.grad_fn is not None:
            self.grad_fn._register_hook_dict(self)

    from torch.utils.hooks import RemovableHandle
    handle = RemovableHandle(self._backward_hooks)
    self._backward_hooks[handle.id] = hook
    return handle
```

#### register_post_accumulate_grad_hook()

**源码**: `torch/_tensor.py` (L713-L773)

```python
def register_post_accumulate_grad_hook(self, hook):
    """Registers a backward hook that runs after grad accumulation."""
    if self.grad_fn is not None:
        raise RuntimeError(
            "post accumulate grad hooks cannot be registered on non-leaf tensors"
        )
    # ... 类似 register_hook 的实现
```

### 3.2 序列化方法

#### __reduce_ex__()

**源码**: `torch/_tensor.py` (L274-L296)

```python
def __reduce_ex__(self, proto):
    """Pickle 序列化支持"""
    state = torch._utils._get_obj_state(self)
    if type(self) is Tensor and not state:
        # Fast path for regular tensor without Python state
        return self._reduce_ex_internal(proto)

    func, args = self._reduce_ex_internal(proto)
    return (_rebuild_from_type_v2, (func, type(self), args, state))
```

#### _reduce_ex_internal()

处理不同类型 Tensor 的序列化:
- 稀疏 Tensor
- 量化 Tensor
- 嵌套 Tensor
- Meta Tensor
- 包装子类 Tensor

### 3.3 命名张量方法

#### refine_names()

**源码**: `torch/_tensor.py` (L1351-L1393)

```python
def refine_names(self, *names):
    """Refines the dimension names of self according to names."""
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.refine_names, (self,), self, *names)
    names = resolve_ellipsis(names, self.names, "refine_names")
    return super().refine_names(names)
```

#### align_to()

**源码**: `torch/_tensor.py` (L1395-L1438)

```python
def align_to(self, *names):
    """Permutes the dimensions to match the order specified in names."""
    ellipsis_idx = single_ellipsis_index(names, "align_to")
    if ellipsis_idx is None:
        return super().align_to(names)
    return super().align_to(
        [name for name in names if not is_ellipsis(name)],
        ellipsis_idx
    )
```

#### rename() / rename_()

**源码**: `torch/_tensor.py` (L1477-L1517)

```python
def rename(self, *names, **rename_map):
    """Renames dimension names of self."""
    return update_names(self, names, rename_map, inplace=False)

def rename_(self, *names, **rename_map):
    """In-place version of rename()."""
    return update_names(self, names, rename_map, inplace=True)
```

### 3.4 特殊方法

#### __torch_function__()

**源码**: `torch/_tensor.py` (L1681-L1707)

```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    """
    This __torch_function__ implementation wraps subclasses such that
    methods called on subclasses return a subclass instance.
    """
    if kwargs is None:
        kwargs = {}

    if not all(issubclass(cls, t) for t in types):
        return NotImplemented

    with _C.DisableTorchFunctionSubclass():
        ret = func(*args, **kwargs)
        if func in get_default_nowrap_functions():
            return ret
        else:
            return _convert(ret, cls)
```

**功能**: 实现 `__torch_function__` 协议，支持 Tensor 子类重载

#### __dlpack__()

**源码**: `torch/_tensor.py` (L1711-L1840)

```python
def __dlpack__(
    self,
    *,
    stream: Any | None = -1,
    max_version: tuple[int, int] | None = None,
    dl_device: tuple[enum.IntEnum, int] | None = None,
    copy: bool | None = None,
):
    """Creates a DLPack capsule for cross-framework tensor exchange."""
    if self.requires_grad:
        raise BufferError(
            "Can't export tensors that require gradient, use tensor.detach()"
        )
    if self.is_conj():
        raise BufferError("Can't export tensors with the conjugate bit set")
    # ... DLPack 导出逻辑
```

### 3.5 运算符重载

```python
# 算术运算符
__rsub__ = lambda self, other: _C._VariableFunctions.rsub(self, other)
__rdiv__ = lambda self, other: self.reciprocal() * other
__rpow__ = lambda self, other: torch.pow(other, self)
__rmatmul__ = lambda self, other: torch.matmul(other, self)

# 一元运算符
__pos__ = _C.TensorBase.positive
__neg__ = _C.TensorBase.neg
__abs__ = _C.TensorBase.abs

# 迭代和长度
def __len__(self):
    if self.dim() == 0:
        raise TypeError("len() of a 0-d tensor")
    return self.shape[0]

def __iter__(self):
    if self.dim() == 0:
        raise TypeError("iteration over a 0-d tensor")
    return iter(self.unbind(0))
```

---

## 4. C++ 层绑定机制

### 4.1 THPVariable 结构

**源码**: `torch/csrc/autograd/python_variable.h` (L17-L28)

```cpp
struct THPVariable {
  PyObject_HEAD
  // Payload - C++ Tensor
  at::Tensor cdata;

  // Hooks to be run on backwards pass
  PyObject* backward_hooks = nullptr;

  // Hooks to be run after accumulate grad
  PyObject* post_accumulate_grad_hooks = nullptr;
};
```

### 4.2 Tensor 包装和解包

**包装 (C++ -> Python)**:

```cpp
// python_variable.h (L40-L44)
TORCH_PYTHON_API PyObject* THPVariable_Wrap(at::TensorBase&& var);
TORCH_PYTHON_API PyObject* THPVariable_Wrap(const at::TensorBase& var);
TORCH_PYTHON_API PyObject* THPVariable_Wrap(
    const at::TensorBase& var,
    PyTypeObject* type);
```

**解包 (Python -> C++)**:

```cpp
// python_variable.h (L75-L81)
inline const at::Tensor& THPVariable_Unpack(THPVariable* var) {
  return var->cdata;
}

inline const at::Tensor& THPVariable_Unpack(PyObject* obj) {
  return THPVariable_Unpack(reinterpret_cast<THPVariable*>(obj));
}
```

### 4.3 Python Tensor 类型初始化

**源码**: `torch/csrc/tensor/python_tensor.cpp` (L365-L391)

```cpp
void initialize_python_bindings() {
  // 1. 初始化 aten 类型 (PyTensorType 向量)
  initialize_aten_types(tensor_types);

  // 2. 初始化 Python metaclass (torch.FloatTensor 等的元类)
  py_initialize_metaclass(metaclass);

  // 3. 获取 Variable 类的 tp_dict
  auto tensor_dict = get_tensor_dict();

  // 4. 初始化每个 Python 类型对象
  for (auto& tensor_type : tensor_types) {
    py_initialize_tensor_type(
        tensor_type->py_type,
        tensor_type->name,
        tensor_dict.get()
    );
  }

  // 5. 添加到对应模块
  py_bind_tensor_types(tensor_types);
}
```

### 4.4 PyTensorType 结构

**源码**: `torch/csrc/tensor/python_tensor.cpp` (L33-L55)

```cpp
struct PyTensorType {
  PyTypeObject py_type;     // Python 类型对象
  THPDtype* dtype;          // dtype 对象
  THPLayout* layout;        // layout 对象
  bool is_cuda;             // 是否 CUDA 张量
  bool is_xpu;              // 是否 XPU 张量
  char name[64];            // 类型名称 (如 "torch.FloatTensor")
  int backend;              // 后端类型
  int scalar_type;          // 标量类型

  Backend get_backend() const {
    return static_cast<Backend>(backend);
  }

  DispatchKey get_dispatch_key() const {
    return backendToDispatchKey(static_cast<Backend>(backend));
  }
};
```

---

## 5. Tensor 类型层次

### 5.1 所有 Tensor 类型

PyTorch 为每个 `(backend, dtype)` 组合生成一个 Tensor 类型:

```python
# CPU 浮点类型
torch.FloatTensor    # torch.float32
torch.DoubleTensor   # torch.float64
torch.HalfTensor     # torch.float16
torch.BFloat16Tensor # torch.bfloat16

# CPU 整型类型
torch.IntTensor      # torch.int32
torch.LongTensor     # torch.int64
torch.ShortTensor    # torch.int16
torch.ByteTensor     # torch.uint8
torch.BoolTensor     # torch.bool

# CUDA 类型 (如果启用 CUDA)
torch.cuda.FloatTensor
torch.cuda.DoubleTensor
# ... 其他 CUDA 类型
```

### 5.2 类型注册

**源码**: `torch/csrc/tensor/python_tensor.cpp` (L348-L363)

```cpp
static void initialize_aten_types(std::vector<PyTensorType*>& tensor_types) {
  // 包括 CUDA 类型 (即使 PyTorch 没有编译 CUDA 支持)
  auto declared_types = torch::utils::all_declared_types();
  tensor_types.resize(declared_types.size());

  for (size_t i = 0, end = declared_types.size(); i != end; i++) {
    tensor_types[i] = new PyTensorType();
    auto& tensor_type = *tensor_types[i];
    Backend backend = declared_types[i].first;
    ScalarType scalar_type = declared_types[i].second;
    set_type(tensor_type, backend, scalar_type);
    set_name(tensor_type, get_name(backend, scalar_type));
  }

  set_default_tensor_type(Backend::CPU, ScalarType::Float);
}
```

---

## 6. _torch_function 协议

### 6.1 协议概述

`__torch_function__` 允许 Tensor 子类拦截和自定义 PyTorch 函数的行为。

### 6.2 执行流程

```
用户调用 torch.add(a, b)
        ↓
检查是否有 __torch_function__
        ↓
如果有，调用: Tensor.__torch_function__(torch.add, types, (a, b), {})
        ↓
子类可以:
1. 调用 super().__torch_function__() 使用默认行为
2. 返回自定义结果
3. 返回 NotImplemented 传递给下一个子类
```

### 6.3 实现示例

```python
class MyTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # 检查是否所有类型都是当前类型的子类
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        # 禁用子类分发，防止递归
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)

        # 将结果转换为当前类型
        return cls._convert(ret)
```

---

## 7. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `torch/_tensor.py` | L110 | Tensor 类定义 |
| `torch/_tensor.py` | L576-L633 | backward() 方法 |
| `torch/_tensor.py` | L663-L711 | register_hook() |
| `torch/_tensor.py` | L1681-L1707 | __torch_function__() |
| `torch/_C/__init__.pyi.in` | L1953-L1980 | TensorBase 类型注解 |
| `torch/csrc/autograd/python_variable.h` | L17-L28 | THPVariable 结构 |
| `torch/csrc/tensor/python_tensor.cpp` | L365-L391 | initialize_python_bindings() |
| `torch/csrc/tensor/python_tensor.cpp` | L33-L55 | PyTensorType 结构 |

---

## 8. 下一步

| 章节 | 主题 |
|------|------|
| [Part 2](./02-tensor-operations.md) | Tensor 操作与工厂函数 |
| [Part 3](./03-autograd.md) | 自动微分集成 |

---

**参考资料**:
- `torch/_tensor.py` - Python Tensor 完整定义
- `torch/csrc/tensor/python_tensor.cpp` - C++ Tensor 类型绑定
- `torch/csrc/autograd/python_variable.cpp` - Variable 绑定实现
- `torch/_C/__init__.pyi.in` - TensorBase 类型注解
