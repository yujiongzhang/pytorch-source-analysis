# Python 层 Tensor API（六）：分发机制

> **前序**: [Part 5 - 工厂函数详解](./05-factory-functions.md)
> **核心源码**: `torch/utils/_python_dispatch.py`, `torch/overrides.py`, `torch/_python_dispatcher.py`

---

## 1. 分发机制概览

PyTorch 提供两种 Python 层分发机制：

```
┌─────────────────────────────────────────────────────────┐
│              PyTorch 分发机制                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  __torch_function__                             │   │
│  │  - 重载 torch 函数 (如 torch.add)                 │   │
│  │  - 受 NumPy __array_function__ 启发              │   │
│  │  - 用于 Tensor 子类                              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  __torch_dispatch__                             │   │
│  │  - 重载底层 ATen 操作                             │   │
│  │  - 用于更细粒度的控制                            │   │
│  │  - 支持 TorchDispatchMode                        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 1.1 两种机制的对比

| 特性 | `__torch_function__` | `__torch_dispatch__` |
|------|---------------------|---------------------|
| 重载对象 | torch 函数 | ATen 操作 |
| 粒度 | 粗 (函数级) | 细 (操作级) |
| 性能 | 较慢 | 较快 |
| 用途 | Tensor 子类 | 模式/代理 Tensor |
| 工厂函数 | 支持 | 支持 |
| 组合性 | 有限 | 良好 (模式栈) |

---

## 2. __torch_function__ 机制

### 2.1 基本概念

`__torch_function__` 允许 Tensor 子类重载 PyTorch 函数的行为。

**灵感来源**: NumPy 的 `__array_function__` 协议

### 2.2 基本用法

```python
import torch

class MyTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        拦截并自定义 torch 函数的行为。

        Args:
            func: 被调用的函数 (如 torch.add)
            types: 参与调用的所有 Tensor 子类类型
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            函数结果
        """
        if kwargs is None:
            kwargs = {}

        # 检查是否所有类型都是当前类型的子类
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        # 调用原始函数
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)

        # 将结果转换为当前类型
        return cls._convert(ret)

    @staticmethod
    def _convert(ret, cls):
        """将结果转换为指定类型"""
        if cls is torch.Tensor:
            return ret

        if isinstance(ret, torch.Tensor) and not isinstance(ret, cls):
            ret = ret.as_subclass(cls)

        if isinstance(ret, (tuple, list)):
            ret = type(ret)(cls._convert(r, cls) for r in ret)

        return ret
```

### 2.3 执行流程

```
用户调用：torch.add(my_tensor, other_tensor)
        ↓
检查参数是否有 __torch_function__ 方法
        ↓
调用：MyTensor.__torch_function__(torch.add, (MyTensor,), (my_tensor, other_tensor), {})
        ↓
在 __torch_function__ 中:
  1. 检查类型兼容性
  2. 禁用 TorchFunction 防止递归
  3. 调用原始函数
  4. 转换返回类型
        ↓
返回结果
```

### 2.4 handle_torch_function

**源码**: `torch/overrides.py`

```python
def handle_torch_function(
    public_api: Callable,
    relevant_types: Iterable[Type],
    *args,
    **kwargs
) -> Any:
    """
    处理 __torch_function__ 分发。

    当函数检测到有自定义 __torch_function__ 的参数时调用此函数。
    """
    # 获取实现了 __torch_function__ 的类型
    types = tuple(type(arg) for arg in args if hasattr(type(arg), '__torch_function__'))

    if not types:
        # 没有自定义类型，调用默认实现
        return public_api(*args, **kwargs)

    # 获取第一个有 __torch_function__ 的类型
    tensor_type = types[0]

    # 调用该类型的 __torch_function__
    result = tensor_type.__torch_function__(public_api, types, args, kwargs)

    if result is NotImplemented:
        # 如果返回 NotImplemented，尝试下一个类型
        if len(types) > 1:
            return handle_torch_function(public_api, types[1:], *args, **kwargs)
        else:
            return public_api(*args, **kwargs)

    return result
```

### 2.5 has_torch_function 检查

```python
# 检查单个参数
def has_torch_function_unary(tensor):
    """检查单个 Tensor 是否有 __torch_function__"""
    return type(tensor) is not Tensor and has_torch_function(tensor)

# 检查多个参数
def has_torch_function(tensors):
    """检查多个 Tensor 中是否有 __torch_function__"""
    for tensor in tensors:
        if type(tensor) is not Tensor and hasattr(type(tensor), '__torch_function__'):
            return True
    return False
```

### 2.6 在 Python Tensor 方法中的使用

**源码**: `torch/_tensor.py`

```python
class Tensor(torch._C.TensorBase):

    def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.backward, (self,), self,
                gradient=gradient, retain_graph=retain_graph,
                create_graph=create_graph, inputs=inputs
            )
        torch.autograd.backward(
            self, gradient, retain_graph, create_graph, inputs=inputs
        )

    def __len__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__len__, (self,), self)
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        # 注意：__iter__ 不进行 __torch_function__ 分发
        # 参见 https://github.com/pytorch/pytorch/issues/54457
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        return iter(self.unbind(0))
```

---

## 3. __torch_dispatch__ 机制

### 3.1 基本概念

`__torch_dispatch__` 提供了更细粒度的操作重载机制，用于：
- 创建代理 Tensor（如 FakeTensor）
- 实现自定义分发模式
- 追踪操作执行

### 3.2 Tensor 子类实现

**源码**: `torch/_tensor.py` (L1709)

```python
class Tensor(torch._C.TensorBase):
    __torch_dispatch__ = _C._disabled_torch_dispatch_impl
```

默认的 `__torch_dispatch__` 被禁用，需要子类重写。

### 3.3 TorchDispatchMode

**源码**: `torch/utils/_python_dispatch.py`

```python
class TorchDispatchMode:
    """
    允许在动态作用域内重载所有 __torch_dispatch__ 可重载函数，
    无需创建 Tensor 子类或手动修改 PyTorch API 中的函数。

    使用场景:
    1. 重载工厂函数 (不接受 Tensor 参数的函数)
    2. 记录中间计算
    3. 控制执行顺序
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        重载此方法以自定义操作行为。

        Args:
            func: 被调用的操作 (OpOverload)
            types: 参与调用的所有类型
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            操作结果
        """
        raise NotImplementedError

    def __enter__(self):
        global _is_in_torch_dispatch_mode
        self.old_dispatch_mode_flags.append(_is_in_torch_dispatch_mode)
        _is_in_torch_dispatch_mode = True
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _is_in_torch_dispatch_mode
        _is_in_torch_dispatch_mode = self.old_dispatch_mode_flags.pop()
        _pop_mode()
```

### 3.4 模式栈管理

```python
# 模式栈操作
def _push_mode(mode: TorchDispatchMode) -> None:
    """将模式推入栈中"""
    _push_on_torch_dispatch_stack(mode)

def _pop_mode() -> TorchDispatchMode:
    """从栈中弹出模式"""
    return _pop_torch_dispatch_stack()

def _get_current_dispatch_mode() -> TorchDispatchMode | None:
    """获取当前模式 (栈顶)"""
    stack_len = _len_torch_dispatch_stack()
    if stack_len > 0:
        return _get_dispatch_stack_at(stack_len - 1)
    return None

def _get_current_dispatch_mode_stack() -> list[TorchDispatchMode]:
    """获取整个模式栈"""
    stack_len = _len_torch_dispatch_stack()
    return [_get_dispatch_stack_at(i) for i in range(stack_len)]
```

### 3.5 禁用当前模式

```python
@contextlib.contextmanager
def _disable_current_modes():
    """临时禁用所有模式"""
    mode_len = _len_torch_dispatch_stack()
    old_modes = [_pop_mode() for _ in range(mode_len)]

    try:
        yield old_modes
    finally:
        # 恢复模式
        for mode in reversed(old_modes):
            _push_mode(mode)
```

---

## 4. FakeTensor 与 __torch_dispatch__

### 4.1 FakeTensor 概述

FakeTensor 是 `__torch_dispatch__` 的典型应用，用于：
- 形状推断（无需实际分配内存）
- 计算图分析
- 编译优化

**源码**: `torch/_subclasses/fake_tensor.py`

### 4.2 FakeTensorMode

```python
class FakeTensorMode(TorchDispatchMode):
    """
    将操作转换为使用 FakeTensor 的模式。
    FakeTensor 只有 metadata（形状、dtype、device），没有实际数据。
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 禁用缓存（可选）
        if not self.cache_enabled:
            return self._dispatch(func, types, args, kwargs)

        # 检查缓存
        cache_key = self._make_cache_key(func, args, kwargs)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 执行分发
        result = self._dispatch(func, types, args, kwargs)

        # 缓存结果
        self.cache[cache_key] = result
        return result

    def _dispatch(self, func, types, args=(), kwargs=None):
        # 将 FakeTensor 转换为 Meta Tensor
        meta_args = tuple(
            self._to_meta(a) if isinstance(a, FakeTensor) else a
            for a in args
        )
        meta_kwargs = {
            k: self._to_meta(v) if isinstance(v, FakeTensor) else v
            for k, v in kwargs.items()
        }

        # 在 Meta 设备上执行操作
        with torch._C.DisableTorchDispatchSubclass():
            meta_result = func(*meta_args, **meta_kwargs)

        # 将结果转换回 FakeTensor
        return self._from_meta(meta_result)
```

### 4.3 FakeTensor 实现

```python
class FakeTensor(torch.Tensor):
    """
    只有 metadata 的 Tensor，用于形状推断。
    """

    def __new__(cls, elem, *, fake_device=None):
        # 从 Meta Tensor 创建
        if isinstance(elem, torch.Tensor):
            # 创建 wrapper
            r = torch.Tensor._make_wrapper_subclass(
                cls,
                elem.size(),
                strides=elem.stride(),
                storage_offset=elem.storage_offset(),
                dtype=elem.dtype,
                layout=elem.layout,
                device=fake_device or elem.device,
                requires_grad=elem.requires_grad
            )
            r.elem = elem  # 保存底层 Meta Tensor
            return r

    def __tensor_flatten__(self):
        """返回内部 Tensor 名称和上下文"""
        return ["elem"], {}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride):
        """从内部 Tensor 重建"""
        elem = inner_tensors["elem"]
        return FakeTensor(elem)

    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 通过 FakeTensorMode 处理
        mode = maybe_get_fake_mode(args[0])
        if mode:
            return mode.__torch_dispatch__(func, types, args, kwargs)
        raise NotImplementedError
```

---

## 5. return_and_correct_aliasing

### 5.1 问题背景

在 `__torch_dispatch__` 子类中，需要正确处理：
- View 操作的存储共享
- Inplace 操作的返回
- Out 参数的处理

### 5.2 API 使用

**源码**: `torch/utils/_python_dispatch.py` (L869)

```python
def return_and_correct_aliasing(func, args, kwargs, out):
    """
    确保 __torch_dispatch__ 子类正确处理操作的别名行为。

    处理:
    1. View 操作：共享输入和输出 Tensor 的存储
    2. Inplace/out 操作：直接返回输入 Tensor

    Args:
        func: OpOverload
        args: 位置参数
        kwargs: 关键字参数
        out: 操作输出

    Returns:
        修正后的输出
    """
    # 获取 Schema 信息（缓存）
    schema_info = get_alias_info(func)

    # 1. 修正 View 操作的存储别名
    _correct_storage_aliasing(func, schema_info, args, out)

    # 2. 处理 inplace_view 操作
    if schema_info.is_inplace_view_op:
        mutated_args = [...]
        with torch.utils._mode_utils.no_dispatch():
            func(*args, **kwargs)  # 更新 metadata

    # 3. 处理 inplace/out 操作
    if schema_info.outs_write_aliases is None:
        return out  # 无需修正

    # 返回输入 Tensor 而不是新输出
    if len(schema_info.outs_write_aliases) == 1:
        return get_arg_from_alias(
            schema_info.outs_write_aliases[0], schema_info, args, kwargs
        )

    # 多返回情况
    return type(out)(
        get_arg_from_alias(write_alias, schema_info, args, kwargs)
        for write_alias in schema_info.outs_write_aliases
    )
```

### 5.3 使用示例

```python
class MyTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 执行操作
        out = func(*args, **kwargs)

        # 修正别名关系
        return return_and_correct_aliasing(func, args, kwargs, out)
```

---

## 6. PythonDispatcher - 分发器演示

**源码**: `torch/_python_dispatcher.py`

```python
class PythonDispatcher:
    """
    演示 C++ 分发器预计算工作原理的简化版本。

    展示对于某个操作，在注册到不同 DispatchKey 后，
    计算出的分发表是什么样子的。
    """

    supported_keys = [
        # 运行时 Keys
        "CPU", "AutogradCPU",
        "FPGA", "AutogradOther",
        "XLA", "AutogradXLA",
        "Lazy", "AutogradLazy",
        # 别名 Keys
        "CompositeExplicitAutograd",
        "Autograd",
        "CompositeImplicitAutograd",
    ]

    def __init__(self):
        # 创建测试操作
        self.ref = C._dispatch_library("FRAGMENT", "__test__", "")
        self.ref.def_("foo(Tensor x) -> Tensor")

    def register(self, dispatchKeys):
        """注册内核到指定 DispatchKey"""
        for key in dispatchKeys:
            self.ref.impl_t_t("foo", dispatch=key, debug="fn_" + key)

    def dispatchTable(self):
        """返回计算后的分发表"""
        output = "Computed Dispatch Table\n"
        output += "key             kernel\n"
        output += "---------------------------\n"

        table = self.rawDispatchTable()
        for line in table.split("\n"):
            k = line.split(":")[0]
            if k in self.runtime_keys:
                output += f"{k:<15} {line.split(':')[1]}\n"
        return output

# 使用示例
dispatcher = PythonDispatcher()
dispatcher.register(["CPU", "XLA", "CompositeImplicitAutograd"])
print(dispatcher.dispatchTable())
```

---

## 7. 实际应用示例

### 7.1 日志记录模式

```python
class LoggingMode(TorchDispatchMode):
    """记录所有操作的模式"""

    def __init__(self, file=None):
        self.file = file or sys.stdout
        self.indent = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 打印操作
        indent_str = "  " * self.indent
        print(f"{indent_str}{func}", file=self.file)

        self.indent += 1
        try:
            # 调用下一个模式/实现
            result = func(*args, **kwargs)
            return result
        finally:
            self.indent -= 1

# 使用
x = torch.randn(3, 4)
with LoggingMode():
    y = x.sin().cos().sum()
```

### 7.2 形状追踪模式

```python
class ShapeTracingMode(TorchDispatchMode):
    """只追踪形状，不执行实际计算"""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 转换为 Meta Tensor
        meta_args = tuple(
            torch.empty_like(a) if isinstance(a, torch.Tensor) else a
            for a in args
        )

        # 在 Meta 设备上执行
        with torch._C.DisableTorchDispatchSubclass():
            return func(*meta_args, **kwargs)
```

---

## 8. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `torch/utils/_python_dispatch.py` | L70-L266 | TorchDispatchMode 类 |
| `torch/utils/_python_dispatch.py` | L869-L965 | return_and_correct_aliasing |
| `torch/overrides.py` | - | handle_torch_function |
| `torch/_tensor.py` | L1681-L1707 | __torch_function__ 实现 |
| `torch/_subclasses/fake_tensor.py` | - | FakeTensor 实现 |
| `torch/_python_dispatcher.py` | - | PythonDispatcher 演示 |

---

## 总结

| 机制 | 用途 | 实现位置 |
|------|------|----------|
| `__torch_function__` | Tensor 子类重载 | `torch/overrides.py` |
| `__torch_dispatch__` | 细粒度操作重载 | `torch/utils/_python_dispatch.py` |
| TorchDispatchMode | 上下文式分发 | `torch/utils/_python_dispatch.py` |
| FakeTensor | 形状推断 | `torch/_subclasses/fake_tensor.py` |

---

**参考资料**:
- `torch/utils/_python_dispatch.py` - TorchDispatchMode 实现
- `torch/overrides.py` - __torch_function__ 实现
- `torch/_subclasses/fake_tensor.py` - FakeTensor 实现
- `torch/_python_dispatcher.py` - 分发器演示
