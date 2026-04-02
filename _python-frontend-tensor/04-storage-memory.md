# 04. Storage 与内存管理

> 本文档深入解析 PyTorch 的底层存储系统与内存管理机制

---

## 01. Storage 存储系统详解

### 1.1 Storage 层次结构

PyTorch 的存储系统分为两层：

```
Tensor (逻辑视图)
  │
  ├─ storage_offset: 在 Storage 中的起始位置
  ├─ size: 逻辑形状 [3, 4]
  └─ stride: 逻辑步幅 (4, 1)
        │
        ↓
TypedStorage (类型化存储，已弃用)
  │
  └─ _untyped_storage
        │
        ↓
UntypedStorage (无类型存储，底层字节数组)
```

### 1.2 _StorageBase 基类

**源码位置**: `torch/storage.py:41-200`

```python
class _StorageBase:
    """所有 Storage 类的抽象基类"""
    
    _cdata: Any  # C++ 存储对象的指针
    is_sparse: bool = False
    is_sparse_csr: bool = False
    device: torch.device
    _fake_device: torch.device | None = None  # FakeTensor 用
    _checkpoint_offset: int | None = None  # 序列化用
    
    def __len__(self) -> int:
        """返回存储的元素数量"""
        raise NotImplementedError
    
    def __getitem__(self, idx):
        """通过索引访问元素"""
        raise NotImplementedError
    
    def __setitem__(self, *args, **kwargs):
        """通过索引设置元素"""
        raise NotImplementedError
    
    def copy_(self, source, non_blocking=False):
        """从另一个存储复制数据"""
        raise NotImplementedError
    
    def nbytes(self) -> int:
        """返回存储占用的字节数"""
        raise NotImplementedError
    
    def size(self) -> int:
        """返回存储大小（字节数）"""
        return self.nbytes()
    
    def element_size(self) -> int:
        """返回每个元素的字节数"""
        raise NotImplementedError
    
    def data_ptr(self) -> int:
        """返回底层数据指针（内存地址）"""
        raise NotImplementedError
    
    def resizable(self) -> bool:
        """返回存储是否可调整大小"""
        raise NotImplementedError
    
    def resize_(self, size: int):
        """原地调整存储大小"""
        raise NotImplementedError
```

### 1.3 TypedStorage 类

**源码位置**: `torch/storage.py`

```python
class TypedStorage:
    """
    带类型信息的 Storage（已弃用，但为了向后兼容仍在使用）
    
    .. warning::
        TypedStorage is deprecated. It will be removed in the future, and
        UntypedStorage will be the only storage class.
    """
    
    def __init__(self, wrap_storage, dtype, _internal=False):
        """
        Args:
            wrap_storage: 底层 UntypedStorage
            dtype: 数据类型
            _internal: 内部使用标志（避免弃用警告）
        """
        self._untyped_storage = wrap_storage
        self.dtype = dtype
    
    def __repr__(self):
        info_str = f"[{torch.typename(self)}(device={self.device}) of size {len(self)}]"
        if self.device.type == "meta":
            return "...\n" + info_str
        data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
        return data_str + "\n" + info_str
    
    def __copy__(self):
        return self.clone()
    
    def __deepcopy__(self, memo):
        memo = memo.setdefault("torch", {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage
    
    def clone(self):
        """返回此存储的副本"""
        return type(self)(self.nbytes(), device=self.device).copy_(self)
    
    def _share_memory_(self):
        """将存储移到共享内存"""
        with _share_memory_lock:
            # ... 共享内存实现
            pass
```

### 1.4 UntypedStorage 类

**源码位置**: `torch/storage.py`

```python
class UntypedStorage:
    """
    无类型底层存储 - 原始字节数组
    
    这是未来唯一的 Storage 类型
    """
    
    def __new__(cls, size_or_sequence=None, device=None, dtype=None):
        """
        创建 UntypedStorage
        
        Args:
            size_or_sequence: 大小或序列
            device: 设备类型
            dtype: (可选) 用于推断元素大小的数据类型
        """
        # 实际实现在 C++ 层
        pass
    
    @classmethod
    def from_file(cls, filename, shared=False, nbytes=None):
        """从文件创建存储"""
        pass
    
    @classmethod
    def from_buffer(cls, buffer):
        """从缓冲区（如 numpy 数组）创建存储"""
        pass
```

### 1.5 Storage 类型转换

**源码位置**: `torch/storage.py:273-353`

```python
class _StorageBase:
    def _to(self, dtype):
        """
        转换为指定 dtype 的 TypedStorage
        
        注意：如果 dtype 与当前类型相同，仍会创建副本
        """
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype")
        
        # 创建 Tensor 视图，转换类型，再提取 Storage
        storage = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .to(dtype)
            ._typed_storage()
        )
        
        # 如果数据指针相同，需要克隆（避免共享）
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage
    
    # 便捷类型转换方法
    def double(self): return self._to(torch.double)
    def float(self): return self._to(torch.float)
    def half(self): return self._to(torch.half)
    def long(self): return self._to(torch.long)
    def int(self): return self._to(torch.int)
    def short(self): return self._to(torch.short)
    def char(self): return self._to(torch.int8)
    def byte(self): return self._to(torch.uint8)
    def bool(self): return self._to(torch.bool)
    def bfloat16(self): return self._to(torch.bfloat16)
    def complex_double(self): return self._to(torch.cdouble)
    def complex_float(self): return self._to(torch.cfloat)
```

---

## 02. Tensor 与 Storage 的关系

### 2.1 Tensor 使用 Storage

**源码位置**: `torch/_tensor.py:298-321`

```python
class Tensor:
    def storage(self):
        r"""
        返回底层 TypedStorage（已弃用）
        
        .. warning::
            TypedStorage is deprecated.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage, (self,), self)
        torch.storage._warn_typed_storage_removal(stacklevel=2)
        return self._typed_storage()
    
    def _typed_storage(self):
        """内部方法，获取 TypedStorage"""
        untyped_storage = self.untyped_storage()
        return torch.TypedStorage(
            wrap_storage=untyped_storage, 
            dtype=self.dtype, 
            _internal=True
        )
    
    def untyped_storage(self):
        """
        返回底层 UntypedStorage（推荐方式）
        
        Example::
            >>> x = torch.randn(3, 4)
            >>> storage = x.untyped_storage()
            >>> len(storage)  # 元素数量
            12
            >>> storage.nbytes()  # 字节数
            48  # 12 * 4 (float32)
        """
        # C++ 实现
        pass
```

### 2.2 共享 Storage 的 Tensor 视图

```python
import torch

# 创建 Tensor
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

# 创建视图（共享存储）
y = x[1:3]

# 验证共享存储
print(x.storage().data_ptr() == y.storage().data_ptr())  # True

# 修改视图会影响原 Tensor
y[0] = 100.0
print(x)  # tensor([1., 100., 3., 4.])

# storage_offset 表示视图在存储中的起始位置
print(y.storage_offset())  # 1
```

### 2.3 set_ 方法 - 直接设置 Storage

**源码位置**: C++ 实现

```python
x = torch.Tensor()

# 直接使用 Storage 设置 Tensor
storage = torch.FloatStorage([1.0, 2.0, 3.0, 4.0])
x.set_(storage, 0, (2, 2), (2, 1))

print(x)
# tensor([[1., 2.],
#         [3., 4.]])

# set_ 参数说明:
# set_(storage, storage_offset, size, stride)
```

---

## 03. 跨进程共享内存

### 3.1 share_memory_() 方法

**源码位置**: `torch/storage.py:391-400` 和 `torch/_tensor.py:844-855`

```python
class _StorageBase:
    def share_memory_(self):
        """
        将存储移到共享内存
        
        对于 CUDA 或 PrivateUse1 设备，这是无操作（因为它们使用 IPC）
        对于 CPU，使用文件系统或 POSIX 共享内存
        """
        from torch.multiprocessing import get_sharing_strategy
        
        if self.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
            pass  # CUDA 使用 IPC，不需要特殊处理
        elif get_sharing_strategy() == "file_system":
            self._share_filename_cpu_()  # 使用文件描述符共享
        else:
            self._share_fd_cpu_()  # 使用文件名共享
        
        return self


class Tensor:
    def share_memory_(self):
        """
        将 Tensor 的底层存储移到共享内存
        
        Returns:
            self (用于链式调用)
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.share_memory_, (self,), self)
        self._typed_storage()._share_memory_()
        return self
```

### 3.2 is_shared() 检查

```python
class Tensor:
    def is_shared(self):
        r"""
        检查 Tensor 是否在共享内存中
        
        CUDA Tensor 总是返回 True
        
        Returns:
            bool
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.is_shared, (self,), self)
        return self._typed_storage()._is_shared()
```

### 3.3 多进程使用示例

```python
import torch
import torch.multiprocessing as mp

def worker(tensor):
    # 修改共享 Tensor
    tensor.add_(1)
    print(f"Worker: {tensor}")

if __name__ == "__main__":
    # 创建 Tensor 并移到共享内存
    x = torch.tensor([1.0, 2.0, 3.0])
    x.share_memory_()
    
    # 启动进程
    p = mp.Process(target=worker, args=(x,))
    p.start()
    p.join()
    
    # 主进程能看到修改
    print(f"Main: {x}")  # tensor([2., 3., 4.])
```

---

## 04. 内存分配器

### 4.1 CPU 内存分配

CPU 内存使用标准分配器：

```python
# 创建 Tensor 时分配内存
x = torch.randn(1000, 1000)  # 分配 ~8MB

# 查看内存使用
import sys
print(f"Tensor size: {x.element_size() * x.numel()} bytes")  # 8000000 bytes

# 手动释放内存
del x
import gc
gc.collect()
```

### 4.2 CUDA 内存分配

**源码位置**: `torch/cuda/memory.py`

```python
def summary(memory_type=None):
    """
    返回 CUDA 内存使用摘要
    
    Returns:
        dict: 包含已分配/缓存内存的统计信息
    """
    pass

def memory_allocated(device=None):
    """
    返回当前为张量分配的 GPU 内存（字节）
    """
    pass

def max_memory_allocated(device=None):
    """
    返回历史上为张量分配的最大 GPU 内存（字节）
    """
    pass

def memory_reserved(device=None):
    """
    返回当前缓存中保留的 GPU 内存（字节）
    """
    pass

def memory_stats(device=None):
    """
    返回详细的内存统计信息字典
    """
    pass
```

### 4.3 CUDA 内存管理函数

```python
def empty_cache():
    """
    释放所有未使用的缓存内存
    
    注意：这不会释放正在使用的内存，只会释放缓存池中的空闲内存
    """
    torch._C._cuda_emptyCache()

def reset_peak_memory_stats(device=None):
    """
    重置峰值内存统计
    """
    pass

def _record_memory_history(
    enabled=True,
    device=None,
    record_shapes=True,
    record_memory_allocations=True,
    stack_traces_mode=None,
):
    """
    记录内存历史用于分析
    
    可与 torch.cuda._convert_memory_history 配合使用生成报告
    """
    pass
```

### 4.4 Pinned Memory（锁定内存）

```python
class _StorageBase:
    def pin_memory(self, device="cuda"):
        r"""
        将 CPU 存储复制到锁定（pinned）内存
        
        锁定内存可以与 GPU 进行异步数据传输
        
        Returns:
            锁定内存中的存储副本
        """
        if self.device.type != "cpu":
            raise TypeError(f"cannot pin '{self.type()}' only CPU memory can be pinned")
        
        pinned_tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .pin_memory(device)
        )
        return pinned_tensor.untyped_storage()
    
    def is_pinned(self, device="cuda"):
        r"""
        检查存储是否已在锁定内存中
        """
        pass
```

### 4.5 异步数据传输示例

```python
# 创建 CPU Tensor 并移到 pinned memory
x = torch.randn(1000, 1000, pin_memory=True)  # 等价于 pin_memory()

# 异步传输到 GPU
y = x.cuda(non_blocking=True)

# 可以做其他操作，传输在后台进行
# ...

# 等待传输完成（如果需要）
torch.cuda.synchronize()
```

---

## 05. 内存布局与格式

### 5.1 连续与非连续 Tensor

```python
x = torch.randn(3, 4)
print(x.is_contiguous())  # True

# 转置后变为非连续
y = x.t()
print(y.is_contiguous())  # False

# 创建连续副本
z = y.contiguous()
print(z.is_contiguous())  # True
```

### 5.2 内存格式

```python
# 获取内存格式
print(x.memory_format)  # torch.contiguous_format

# 创建指定格式的 Tensor
x = torch.randn(3, 4, memory_format=torch.channels_last)
x = torch.randn(3, 4, memory_format=torch.preserve_format)
```

### 5.3 stride 与 storage_offset

```python
x = torch.randn(3, 4)
print(x.stride())       # (4, 1) - 行优先
print(x.storage_offset())  # 0

# 切片后的 Tensor
y = x[1:, 1:]
print(y.stride())       # (4, 1) - stride 不变
print(y.storage_offset())  # 5 (1*4 + 1) - 偏移量增加
print(y.size())         # (2, 3)
```

---

## 06. 序列化的存储处理

### 6.1 Storage 序列化

**源码位置**: `torch/storage.py:245-248`

```python
class _StorageBase:
    def __reduce__(self):
        """pickle 序列化支持"""
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))


def _load_from_bytes(b):
    """从字节加载 Storage"""
    return torch.load(io.BytesIO(b))
```

### 6.2 _rebuild_tensor

**源码位置**: `torch/_utils.py:193-196`

```python
def _rebuild_tensor(storage, storage_offset, size, stride):
    """
    从 Storage 重建 Tensor
    
    Args:
        storage: TypedStorage 或 UntypedStorage
        storage_offset: 在 Storage 中的偏移
        size: Tensor 形状
        stride: Tensor 步幅
    
    Returns:
        重建的 Tensor
    """
    # 首先创建正确 dtype/device 的 Tensor
    t = torch.empty((0,), dtype=storage.dtype, device=storage._untyped_storage.device)
    return t.set_(storage._untyped_storage, storage_offset, size, stride)
```

### 6.3 序列化中的注意事项

**Note [Don't serialize hooks]** (`torch/_utils.py:154-188`):

```python
# 向后钩子不再序列化，原因:
# 1. 脆弱：函数重命名会导致无法加载
# 2. 不常用：推荐保存 state_dict 而非完整模型
# 3. DDP 等框架会自行重新添加钩子

# 序列化时会发出警告（如果 Tensor 有钩子）
def warn_if_has_hooks(tensor):
    if tensor._backward_hooks:
        warnings.warn(
            "Backward hooks are not serialized and will be lost"
        )
```

---

## 附录：Storage API 快速参考

### 创建 Storage

```python
# CPU Storage
storage = torch.UntypedStorage(100)  # 100 个元素
storage = torch.FloatStorage([1.0, 2.0, 3.0])  # 从列表

# CUDA Storage
storage = torch.cuda.UntypedStorage(100)
storage = torch.cuda.FloatStorage(100)

# 从 Tensor 获取
storage = tensor.untyped_storage()
storage = tensor.storage()  # TypedStorage (已弃用)
```

### Storage 操作

```python
len(storage)              # 元素数量
storage.nbytes()          # 字节数
storage.element_size()    # 每元素字节数
storage.data_ptr()        # 数据指针
storage.device            # 设备
storage.clone()           # 副本
storage.copy_(src)        # 复制数据
storage.resize_(size)     # 调整大小
```

### 类型转换

```python
storage.float()           # float 类型
storage.double()          # double 类型
storage.int()             # int 类型
# ... 其他类型
```

### 内存管理

```python
storage.share_memory_()   # 共享内存
storage.is_shared()       # 检查是否共享
storage.pin_memory()      # 锁定内存 (CPU only)
storage.is_pinned()       # 检查是否锁定
```

### CUDA 内存统计

```python
torch.cuda.memory_allocated()           # 已分配内存
torch.cuda.memory_reserved()            # 缓存内存
torch.cuda.max_memory_allocated()       # 峰值分配
torch.cuda.empty_cache()                # 清空缓存
torch.cuda.reset_peak_memory_stats()    # 重置峰值统计
```

---

## 后续章节

- [05. 工厂函数实现](./05-factory-functions.md) - Tensor 创建机制
- [06. Dispatcher 调度系统](./06-dispatcher.md) - Dispatch Key 机制
