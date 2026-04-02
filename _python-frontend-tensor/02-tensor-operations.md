# 02. Tensor 操作与方法

> 本文档深入解析 Tensor 的运算符重载、索引切片、类型转换等操作方法的实现

---

## 01. 运算符重载

### 1.1 算术运算符

Tensor 类通过重载 Python 算术运算符，支持直观的数学表达式：

**源码位置**: `torch/_tensor.py:1114-1186`

```python
# 减法 (右侧操作数)
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rsub__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
    return _C._VariableFunctions.rsub(self, other)

# 除法 (右侧操作数)
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rdiv__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
    return self.reciprocal() * other

__rtruediv__ = __rdiv__
__itruediv__ = _C.TensorBase.__idiv__

# 幂运算
__pow__ = cast(
    Callable[
        ["torch._C.TensorBase", Union["Tensor", int, float, bool, complex]],
        "Tensor",
    ],
    _handle_torch_function_and_wrap_type_error_to_not_implemented(
        _C.TensorBase.pow
    ),
)

__ipow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(
    _C.TensorBase.pow_
)

# 取模
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rmod__(self, other: Union["Tensor", int, float, bool, complex]) -> "Tensor":
    return torch.remainder(other, self)

# 整除
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __floordiv__(self, other: Union["Tensor", int, float, bool]) -> "Tensor":
    return torch.floor_divide(self, other)

@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rfloordiv__(self, other: Union["Tensor", int, float, bool]) -> "Tensor":
    return torch.floor_divide(other, self)

# 矩阵乘法
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rmatmul__(self, other: "Tensor") -> "Tensor":
    return torch.matmul(other, self)

# 一元运算符 (直接绑定 C++ 实现)
__pos__ = _C.TensorBase.positive
__neg__ = _C.TensorBase.neg
__abs__ = _C.TensorBase.abs
```

### 1.2 位运算符

**源码位置**: `torch/_tensor.py:1167-1177`

```python
# 右移 (左侧操作数)
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rlshift__(
    self, other: Union["Tensor", int, float, bool, complex]
) -> "Tensor":
    return torch.bitwise_left_shift(other, self)

# 左移 (左侧操作数)
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rrshift__(
    self, other: Union["Tensor", int, float, bool, complex]
) -> "Tensor":
    return torch.bitwise_right_shift(other, self)
```

### 1.3 装饰器说明

```python
def _handle_torch_function_and_wrap_type_error_to_not_implemented(
    f: Callable[Concatenate[_TensorLike, _P], "Tensor"],
) -> Callable[Concatenate[_TensorLike, _P], "Tensor"]:
    """
    装饰器：为方法添加 __torch_function__ 支持
    
    1. 检查参数是否有自定义 __torch_function__ 实现
    2. 捕获 TypeError 并返回 NotImplemented
    """
    @functools.wraps(f)
    def wrapped(self: _TensorLike, *args: _P.args, **kwargs: _P.kwargs) -> "Tensor":
        try:
            sargs = self, *args
            if has_torch_function(sargs):
                return handle_torch_function(wrapped, sargs, *sargs, **kwargs)
            return f(self, *args, **kwargs)
        except TypeError:
            return NotImplemented
    
    return wrapped
```

### 1.4 运算符优先级示例

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 运算符重载使得表达式直观
z = (x + y) * 2 - x @ y  # 加法、乘法、减法、矩阵乘法

# 实际调用链:
# z = (x.__add__(y)).__mul__(2).__sub__(x.__matmul__(y))
```

---

## 02. 索引与切片

### 2.1 __getitem__ 实现

**源码位置**: C++ 实现 (`torch/csrc/autograd/python_variable_indexing.cpp`)

```python
# Python 层调用
x = torch.randn(3, 4, 5)

# 基本索引
x[0]         # 第一行
x[0, 1]      # 第一行第二列
x[0:2, 1:3]  # 切片

# 高级索引
x[[0, 2]]    # 花式索引
x[x > 0]     # 布尔索引

# 底层都调用 __getitem__
x.__getitem__(0)
x.__getitem__((0, 1))
x.__getitem__(slice(0, 2), slice(1, 3))
```

### 2.2 __setitem__ 实现

```python
# 设置元素值
x[0] = 1.0
x[0:2, 1:3] = 0.0

# 底层调用
x.__setitem__(0, 1.0)
x.__setitem__((slice(0, 2), slice(1, 3)), 0.0)
```

### 2.3 索引类型转换

**源码位置**: `torch/_tensor.py:1187-1200`

```python
def __len__(self):
    """
    返回第一个维度的大小
    
    注意：0-d Tensor 不支持 len()
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.__len__, (self,), self)
    if self.dim() == 0:
        raise TypeError("len() of a 0-d tensor")
    if torch._C._get_tracing_state():
        warnings.warn(
            "Using len to get tensor shape might cause the trace to be incorrect. "
            "Recommended usage would be tensor.shape[0].",
            category=torch.jit.TracerWarning,
            stacklevel=2,
        )
    return self.size(0)
```

---

## 03. 类型转换方法

### 3.1 Python 类型转换

**源码位置**: C++ 实现

```python
x = torch.tensor(3.14)

# 转换为 Python 标量
float(x)    # 3.14
int(x)      # 3
bool(x)     # True
complex(x)  # (3.14+0j)

# 底层调用
x.__float__()
x.__int__()
x.__bool__()
x.__complex__()
```

### 3.2 __bool__ 的特殊处理

```python
def __bool__(self):
    """
    Tensor 的布尔转换
    
    只对单元素 Tensor 有效
    """
    if self.numel() == 1:
        return bool(self.item())
    else:
        raise RuntimeError(
            "Boolean value of Tensor with more than one value is ambiguous"
        )
```

### 3.3 item() 方法

```python
def item(self):
    """
    将单元素 Tensor 转换为 Python 标量
    
    Returns:
        int, float, or bool depending on tensor dtype
    """
    if self.numel() != 1:
        raise ValueError("only one element tensors can be converted to Python scalars")
    return self._item()  # C++ 实现
```

---

## 04. 字符串表示

### 4.1 __repr__ 实现

**源码位置**: `torch/_tensor_str.py`

```python
def __repr__(self) -> str:
    """
    Tensor 的字符串表示
    
    包含:
    - 数据内容
    - dtype
    - device
    - requires_grad
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.__repr__, (self,), self)
    return torch._tensor_str._str(self)
```

### 4.2 __str__ 实现

```python
def __str__(self) -> str:
    """
    用户友好的字符串表示
    
    与 __repr__ 类似，但格式更简洁
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.__str__, (self,), self)
    return self._str()
```

### 4.3 格式化输出示例

```python
x = torch.randn(3, 4, dtype=torch.float64, device='cuda:0', requires_grad=True)

print(x)
# 输出:
# tensor([[-0.3688,  0.5497, -1.6769, -1.2073],
#         [-1.2491, -1.7167,  0.0859,  0.1681],
#         [-0.1235,  1.4537, -0.3495, -0.2929]],
#        dtype=torch.float64, device='cuda:0', requires_grad=True)

print(repr(x))  # 更详细的表示
```

### 4.4 __format__ 方法

**源码位置**: `torch/_tensor.py:1144-1151`

```python
def __format__(self, format_spec):
    """
    支持 f-string 格式化
    
    只对标量 Tensor 有效
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
    if self.dim() == 0 and not self.is_meta and type(self) is Tensor:
        # 使用 detach() 避免警告
        return self.detach().item().__format__(format_spec)
    return object.__format__(self, format_spec)

# 使用示例
x = torch.tensor(3.14159)
print(f"{x:.2f}")  # 3.14
```

---

## 05. 设备转移与类型转换

### 5.1 cuda() 方法

**源码位置**: `torch/_tensor.py` (通过 C++ 绑定)

```python
def cuda(self, device=None, non_blocking=False):
    """
    将 Tensor 转移到 CUDA 设备
    
    Args:
        device (int, optional): CUDA 设备 ID
        non_blocking (bool): 是否异步传输
    
    Returns:
        CUDA 上的 Tensor 副本
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.cuda, (self,), self, device, non_blocking)
    return self.to(device=torch.device('cuda', device), non_blocking=non_blocking)
```

### 5.2 cpu() 方法

```python
def cpu(self):
    """
    将 Tensor 转移到 CPU
    
    Returns:
        CPU 上的 Tensor 副本
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.cpu, (self,), self)
    return self.to(device='cpu')
```

### 5.3 to() 方法 (通用设备/类型转换)

```python
def to(self, *args, **kwargs):
    """
    通用设备/类型转换方法
    
    支持的调用方式:
        to(dtype)                  # 只转换类型
        to(device)                 # 只转换设备
        to(dtype, device)          # 同时转换
        to(tensor)                 # 匹配另一个 tensor 的 dtype/device
    """
    # 实现细节复杂，处理多种参数组合
    # 最终调用 C++ 实现
```

### 5.4 类型转换方法

```python
# 浮点类型转换
x.float()    # torch.float32
x.double()   # torch.float64
x.half()     # torch.float16
x.bfloat16() # torch.bfloat16

# 整数类型转换
x.byte()     # torch.uint8
x.char()     # torch.int8
x.short()    # torch.int16
x.int()      # torch.int32
x.long()     # torch.int64
```

---

## 06. Autograd 相关方法

### 6.1 backward() 方法

**源码位置**: `torch/_tensor.py:595-633`

```python
def backward(
    self,
    gradient=None,
    retain_graph=None,
    create_graph=False,
    inputs=None,
):
    r"""
    计算梯度并累积到 .grad 属性中
    
    Args:
        gradient (Tensor, optional): 当前梯度 (对于非标量 Tensor)
        retain_graph (bool, optional): 是否保留计算图
        create_graph (bool, optional): 是否构建导数图 (支持高阶导数)
        inputs (Sequence[Tensor], optional): 指定哪些 Tensor 需要累积梯度
    
    Example::
    
        >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x.sum()
        >>> y.backward()
        >>> x.grad
        tensor([1., 1., 1.])
    """
    if has_torch_function_unary(self):
        return handle_torch_function(
            Tensor.backward,
            (self,),
            self,
            gradient=gradient,
            retain_graph=retain_graph,
            create_graph=create_graph,
            inputs=inputs,
        )
    torch.autograd.backward(
        self, gradient, retain_graph, create_graph, inputs=inputs
    )
```

### 6.2 detach() 方法

**源码位置**: `torch/_tensor.py:806-833`

```python
detach = _C._add_docstr(
    _C.TensorBase.detach,
    r"""
返回一个新的 Tensor，从当前计算图中分离。

结果永远不需要梯度。

注意:
  返回的 Tensor 与原 Tensor 共享存储。

""",
)

detach_ = _C._add_docstr(
    _C.TensorBase.detach_,
    r"""
原地版本：从图中分离 Tensor，使其成为叶子节点。
""",
)
```

### 6.3 register_hook() 方法

**源码位置**: `torch/_tensor.py:663-711`

```python
def register_hook(self, hook):
    r"""
    注册反向传播钩子
    
    每次计算相对于该 Tensor 的梯度时都会调用钩子。
    
    Hook 签名::
    
        hook(grad) -> Tensor or None
    
    Args:
        hook (Callable): 接收梯度并返回修改后梯度的函数
    
    Returns:
        RemovableHandle: 用于移除钩子的句柄
    
    Example::
    
        >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
        >>> h = v.register_hook(lambda grad: grad * 2)  # 梯度翻倍
        >>> v.backward(torch.tensor([1., 2., 3.]))
        >>> v.grad
        tensor([2., 4., 6.])
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.register_hook, (self,), self, hook)
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

### 6.4 register_post_accumulate_grad_hook()

**源码位置**: `torch/_tensor.py:713-773`

```python
def register_post_accumulate_grad_hook(self, hook):
    r"""
    注册梯度累积后执行的钩子
    
    在所有梯度累积完成后调用（即 .grad 已更新）。
    只对叶子 Tensor 有效。
    
    Hook 签名::
    
        hook(param: Tensor) -> None
    
    Example::
    
        >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
        >>> lr = 0.01
        >>> h = v.register_post_accumulate_grad_hook(
        ...     lambda p: p.add_(p.grad, alpha=-lr)
        ... )
        >>> v.backward(torch.tensor([1., 2., 3.]))
        >>> v  # 已执行 SGD 更新
        tensor([-0.0100, -0.0200, -0.0300], requires_grad=True)
    """
    # ... 实现细节
```

---

## 07. 其他实用方法

### 7.1 is_shared() 与 share_memory_()

**源码位置**: `torch/_tensor.py:835-855`

```python
def is_shared(self):
    r"""
    检查 Tensor 是否在共享内存中
    
    CUDA Tensor 总是返回 True
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.is_shared, (self,), self)
    return self._typed_storage()._is_shared()


def share_memory_(self):
    r"""
    将底层存储移到共享内存
    
    如果已经在共享内存中或是 CUDA Tensor，则为无操作。
    共享内存中的 Tensor 不能调整大小。
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.share_memory_, (self,), self)
    self._typed_storage()._share_memory_()
    return self
```

### 7.2 __reversed__()

**源码位置**: `torch/_tensor.py:887-894`

```python
def __reversed__(self):
    """
    沿维度 0 反转 Tensor
    
    支持 reversed() 内置函数
    """
    if has_torch_function_unary(self):
        return handle_torch_function(Tensor.__reversed__, (self,), self)
    if self.dim() == 0:
        return self
    else:
        return self.flip(0)
```

---

## 08. 方法重载模式总结

### 8.1 三种实现模式

PyTorch Tensor 方法的实现遵循三种模式：

| 模式 | 说明 | 示例 |
|------|------|------|
| Python 纯实现 | 完全在 Python 层实现 | `__deepcopy__`, `__reduce_ex__` |
| Python 包装 | Python 层包装，调用 C++ 实现 | `backward()`, `register_hook()` |
| 直接绑定 | 直接引用 C++ 方法 | `detach`, `__abs__`, `__neg__` |

### 8.2 has_torch_function 检查模式

大多数方法遵循相同的模式：

```python
def method(self, arg1, arg2, ...):
    # 1. 检查是否需要调用自定义 __torch_function__
    if has_torch_function_unary(self):
        return handle_torch_function(
            Tensor.method, (self,), self, arg1, arg2, ...
        )
    
    # 2. 正常实现
    return actual_implementation(...)
```

### 8.3 装饰器模式

对于运算符重载，使用统一装饰器：

```python
@_handle_torch_function_and_wrap_type_error_to_not_implemented
def __rsub__(self, other):
    return _C._VariableFunctions.rsub(self, other)
```

---

## 附录：完整方法清单

### 算术运算符 (torch/_tensor.py:1114-1182)

```
__rsub__, __rdiv__, __rtruediv__, __itruediv__
__pow__, __ipow__, __rmod__, __floordiv__, __rfloordiv__
__rlshift__, __rrshift__, __rmatmul__
__pos__, __neg__, __abs__
```

### 索引与长度 (torch/_tensor.py:1187+)

```
__len__, __getitem__, __setitem__
```

### 类型转换

```
__float__, __int__, __bool__, __complex__
item()
```

### 字符串

```
__repr__, __str__, __format__
```

### Autograd (torch/_tensor.py:595-773)

```
backward(), detach(), detach_()
register_hook(), register_post_accumulate_grad_hook()
```

### 设备转移

```
cuda(), cpu(), to()
float(), double(), half(), bfloat16()
byte(), char(), short(), int(), long()
```

### 内存管理

```
is_shared(), share_memory_()
data_ptr(), storage(), storage_offset()
```

---

## 后续章节

- [03. Autograd 自动微分](./03-autograd.md) - 计算图与反向传播引擎
- [04. 内存管理与设备抽象](./04-memory-device.md) - 分配器与设备管理
