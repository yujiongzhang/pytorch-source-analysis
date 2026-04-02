# Python 层 Tensor API（四）：存储与内存管理

> **前序**: [Part 3 - 自动微分集成](./03-autograd.md)
> **核心源码**: `torch/csrc/Storage.cpp`, `c10/core/Storage.h`, `c10/core/StorageImpl.h`

---

## 1. Storage 架构概览

### 1.1 Storage 层次结构

```
┌─────────────────────────────────────────────────────────┐
│              Python Storage (THPStorage)                 │
│           (torch.FloatStorage, etc.)                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              C++ Storage (c10::Storage)                  │
│              持有 StorageImpl 指针                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              StorageImpl (c10::StorageImpl)              │
│  - DataPtr (数据指针)                                    │
│  - Allocator (分配器)                                    │
│  - nbytes (内存大小)                                     │
│  - resizable (是否可调整大小)                            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              DataPtr + 实际内存                          │
│              (CPU/CUDA/MPS Memory)                       │
└─────────────────────────────────────────────────────────┘
```

### 1.2 设计与 Tensor 的关系

```
Tensor (at::Tensor)
    ↓ (指向)
TensorImpl
    ↓ (指向)
Storage
    ↓ (指向)
StorageImpl
    ↓ (指向)
DataPtr → 实际数据内存
```

**关键区别**:
- **Tensor**: 多维数组视图，包含 sizes/strides
- **Storage**: 一维连续内存块，不关心形状

---

## 2. Python Storage 类

### 2.1 Storage 类型

PyTorch 为每种数据类型提供对应的 Storage 类型：

```python
# CPU Storage
torch.Storage              # 默认 (FloatStorage)
torch.FloatStorage         # float32
torch.DoubleStorage        # float64
torch.IntStorage           # int32
torch.LongStorage          # int64
torch.ShortStorage         # int16
torch.ByteStorage          # uint8
torch.BoolTensor           # bool
torch.HalfStorage          # float16
torch.BFloat16Storage      # bfloat16
torch.ComplexFloatStorage  # complex64
torch.ComplexDoubleStorage # complex128

# CUDA Storage
torch.cuda.Storage         # 默认
torch.cuda.FloatStorage
torch.cuda.DoubleStorage
# ... 其他 CUDA 类型
```

### 2.2 Storage 构造

**源码**: `torch/csrc/Storage.cpp` (L105-L250)

```cpp
static PyObject* THPStorage_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {

  // 参数解析
  static torch::PythonArgParser parser({
      "Storage(*, allocator=None, device=None)",
      "Storage(int64_t size, *, allocator=None, device=None)",
      "Storage(PyObject* sequence, *, allocator=None, device=None)",
  });

  auto r = parser.parse(args, kwargs, parsed_args);

  // 获取分配器
  c10::Allocator* allocator = nullptr;
  if (device_opt.has_value()) {
    switch (device.type()) {
      case at::kCPU:
        allocator = c10::GetDefaultCPUAllocator();
        break;
#ifdef USE_CUDA
      case at::kCUDA:
        allocator = c10::cuda::CUDACachingAllocator::get();
        break;
#endif
#ifdef USE_MPS
      case at::kMPS:
        allocator = at::mps::GetMPSAllocator();
        break;
#endif
      // ... 其他设备
    }
  }

  // 创建 Storage
  if (r.idx == 0) {
    // 空 Storage
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            0,
            at::DataPtr(),
            allocator,
            /*resizable=*/true,
            device_opt));

  } else if (r.idx == 1) {
    // 指定大小的 Storage
    auto size = r.toInt64(0);
    self = THPStorage_NewWithStorage(
        type,
        make_storage_impl(
            c10::StorageImpl::use_byte_size_t(),
            size * element_size(dtype),
            allocator,
            device_opt));

  } else if (r.idx == 2) {
    // 从序列创建
    auto data = r.pyobject(0);
    // ... 数据转换逻辑
  }

  return self;
}
```

### 2.3 Python 使用示例

```python
import torch

# 创建空 Storage
storage = torch.FloatStorage(100)  # 100 个 float32 元素

# 从序列创建
storage = torch.FloatStorage([1.0, 2.0, 3.0, 4.0])

# 指定设备
storage_cpu = torch.FloatStorage(100, device='cpu')
storage_cuda = torch.cuda.FloatStorage(100)  # 需要 CUDA

# 从 Buffer 创建
import io
buffer = io.BytesIO()
storage.save(buffer)
buffer.seek(0)
restored = torch.FloatStorage.from_buffer(buffer)
```

---

## 3. StorageImpl 核心结构

### 3.1 StorageImpl 定义

**源码**: `c10/core/StorageImpl.h`

```cpp
struct C10_API StorageImpl : public intrusive_ptr_target {
  // 构造函数
  StorageImpl(
      caffe2::TypeMeta dtype,
      const SymInt& size_bytes,
      at::DataPtr data_ptr,
      Allocator* allocator,
      bool resizable = false);

  // 数据访问
  void* mutable_data() const;
  const void* data() const;
  at::DataPtr& mutable_data_ptr() const;
  const at::DataPtr& data_ptr() const;

  // 设置数据指针
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) const;

  // 大小
  size_t nbytes() const { return nbytes_; }
  SymInt sym_nbytes() const { return sym_nbytes_; }

  // 分配器
  Allocator* allocator() const { return allocator_; }

  // 是否可调整大小
  bool resizable() const { return resizable_; }

  // 设备
  Device device() const { return data_ptr_.device(); }

  // 删除器
  void set_deleter(std::function<void(void*)> deleter) {
    deleter_ = std::move(deleter);
  }

 protected:
  at::DataPtr data_ptr_;           // 数据指针
  size_t nbytes_;                  // 内存大小 (字节)
  SymInt sym_nbytes_;              // 符号内存大小
  Allocator* allocator_;           // 内存分配器
  bool resizable_;                 // 是否可调整大小
  std::function<void(void*)> deleter_;  // 自定义删除器
};
```

### 3.2 DataPtr 智能指针

**源码**: `c10/core/MemoryFormat.h` / `c10/core/impl/DataPtrImpl.h`

```cpp
struct DataPtr {
  // 构造
  DataPtr() : ptr_(nullptr), deleter_(nullptr), device_(kCPU), allocator_(nullptr) {}

  DataPtr(
      void* ptr,
      int64_t device_index,
      DeviceType device_type,
      std::function<void(void*)> deleter,
      Allocator* allocator = nullptr)
      : ptr_(ptr),
        deleter_(std::move(deleter)),
        device_(device_type, device_index),
        allocator_(allocator) {}

  // 析构 - 自动释放内存
  ~DataPtr() {
    if (ptr_ && deleter_) {
      deleter_(ptr_);
    }
  }

  // 访问器
  void* get() const { return ptr_; }
  Device device() const { return device_; }
  Allocator* allocator() const { return allocator_; }

 private:
  void* ptr_;                              // 原始指针
  std::function<void(void*)> deleter_;     // 删除函数
  Device device_;                          // 设备信息
  Allocator* allocator_;                   // 分配器
};
```

---

## 4. 内存分配器

### 4.1 CPU 分配器

**源码**: `c10/core/CPUAllocator.cpp`

```cpp
// 默认 CPU 分配器
class DefaultCPUAllocator : public Allocator {
 public:
  DefaultCPUAllocator() = default;

  DataPtr allocate(size_t n) override {
    if (n == 0) return DataPtr(nullptr, 0, kCPU, nullptr, this);

    void* ptr = aligned_alloc(kAlignment, n);
    if (!ptr) {
      throw std::bad_alloc();
    }

    return DataPtr(
        ptr,
        0,  // device_index
        kCPU,
        [](void* ptr) { ::free(ptr); },  // deleter
        this);
  }

  void* raw_alloc(size_t nbytes) override {
    return aligned_alloc(kAlignment, nbytes);
  }

  void raw_dealloc(void* ptr) override {
    ::free(ptr);
  }

 private:
  static constexpr size_t kAlignment = 64;  // 缓存行对齐
};
```

### 4.2 CUDA 分配器

**源码**: `c10/cuda/CUDACachingAllocator.h`

```cpp
// CUDA 缓存分配器
class CUDACachingAllocator : public Allocator {
 public:
  // 获取单例
  static Allocator* get();

  DataPtr allocate(size_t n) override {
    if (n == 0) return DataPtr(nullptr, 0, kCUDA, nullptr, this);

    void* ptr = allocateMemory(n, nullptr);

    return DataPtr(
        ptr,
        CUDADeviceGuard::current_device(),
        kCUDA,
        [](void* ptr) { delete ptr; },  // 实际由缓存池管理
        this);
  }

  // 缓存统计
  struct Stats {
    size_t allocated_bytes;
    size_t freed_bytes;
    size_t reserved_bytes;
    size_t allocated_bytes_peak;
    size_t freed_bytes_peak;
    size_t reserved_bytes_peak;
  };

  static Stats getDeviceStats(int device);
  static void resetAccumulatedMemoryStats(int device);
};
```

**CUDA 缓存分配器特点**:
- 缓存已释放的 GPU 内存，避免频繁 cudaMalloc/cudaFree
- 支持内存池，减少碎片
- 提供统计信息用于调试和优化

### 4.3 MPS 分配器 (Metal)

**源码**: `aten/src/ATen/mps/MPSAllocator.mm`

```cpp
// MPS (Metal Performance Shaders) 分配器
class MPSAllocator : public Allocator {
 public:
  DataPtr allocate(size_t n) override {
    // 使用 Metal 的 shared 内存
    auto device = at::mps::getMPSDevice();
    void* ptr = device->allocate(n);

    return DataPtr(
        ptr,
        0,  // MPS 只有一个设备
        kMPS,
        [device](void* ptr) { device->dealloc(ptr); },
        this);
  }
};
```

---

## 5. 内存共享

### 5.1 share_memory_()

**源码**: `torch/csrc/StorageSharing.cpp`

```python
# Python API
storage = torch.FloatStorage(100)
storage.share_memory_()  # 移动到共享内存

# 多进程使用
import torch.multiprocessing as mp

def worker(storage):
    storage[0] = 42.0

if __name__ == '__main__':
    storage = torch.FloatStorage(100)
    storage.share_memory_()

    p = mp.Process(target=worker, args=(storage,))
    p.start()
    p.join()

    print(storage[0])  # 42.0
```

**C++ 实现**:

```cpp
static PyObject* THPStorage_shareMemory_(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS

  auto storage = THPStorage_Unpack(self);

  // 创建共享内存
  #if defined(_WIN32)
    // Windows: 使用 CreateFileMapping
    HANDLE handle = CreateFileMapping(...);
  #else
    // Unix: 使用 shm_open
    int fd = shm_open(...);
  #endif

  // 设置共享内存
  storage.share_memory_();

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```

### 5.2 多进程 Storage

**源码**: `torch/multiprocessing/reductions.py`

```python
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import StorageWeakRef

# Tensor 序列化与反序列化
def reduce_tensor(tensor):
    storage = tensor.storage()
    return (
        storage._reduce_ex_internal(None),
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
    )

def rebuild_tensor(storage_data, offset, size, stride):
    storage = torch.storage.TypedStorage._rebuild(storage_data)
    return storage.as_tensor(offset, size, stride)

# 注册 pickle 序列化器
import copyreg
copyreg.pickle(torch.Tensor, reduce_tensor, rebuild_tensor)
```

---

## 6. 内存管理最佳实践

### 6.1 避免内存泄漏

```python
# 不好：累积梯度
for _ in range(100):
    loss = compute_loss()
    loss.backward()  # 梯度累积
    # 忘记清零梯度

# 好：及时清零
for _ in range(100):
    optimizer.zero_grad()  # 清零梯度
    loss = compute_loss()
    loss.backward()
    optimizer.step()
```

### 6.2 使用 no_grad 减少内存

```python
# 训练时需要梯度
model.train()
for batch in dataloader:
    output = model(batch)
    loss = criterion(output, target)
    loss.backward()

# 推理时禁用梯度
model.eval()
with torch.no_grad():
    for batch in dataloader:
        output = model(batch)
        # 不追踪梯度，节省内存
```

### 6.3 CUDA 内存管理

```python
# 查看 CUDA 内存统计
print(torch.cuda.memory_stats())

# 查看已分配内存
print(torch.cuda.memory_allocated())

# 查看缓存内存
print(torch.cuda.memory_reserved())

# 清空缓存
torch.cuda.empty_cache()

# 使用缓存分配器统计
print(torch.cuda.memory_summary())
```

### 6.4 使用 pin_memory 加速 CPU-GPU 传输

```python
# 创建 pinned memory DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # 使用页锁定内存
    num_workers=4
)

# 或者手动创建 pinned tensor
tensor = torch.empty(1000, pin_memory=True)
tensor_cuda = tensor.cuda(non_blocking=True)  # 异步传输
```

---

## 7. TypedStorage 与 UntypedStorage

### 7.1 类型化 Storage

```python
# TypedStorage (推荐)
storage = torch.storage.TypedStorage(
    wrap_storage=untyped_storage,
    dtype=torch.float32
)

# 或者从 Tensor 获取
tensor = torch.randn(10)
typed_storage = tensor.storage()  # TypedStorage
```

### 7.2 UntypedStorage

```python
# UntypedStorage (底层，无类型信息)
tensor = torch.randn(10)
untyped_storage = tensor.untyped_storage()

# 从 UntypedStorage 创建 TypedStorage
typed = torch.storage.TypedStorage(
    wrap_storage=untyped_storage,
    dtype=torch.float32,
    _internal=True
)
```

### 7.3 迁移指南

```python
# 旧代码 (已废弃)
storage = tensor.storage()  # 返回 TypedStorage，但有警告

# 新代码
untyped_storage = tensor.untyped_storage()  # 获取 UntypedStorage
typed_storage = tensor.storage()  # 仍然可用，但未来会移除
```

---

## 8. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `torch/csrc/Storage.cpp` | L35-L103 | THPStorage 定义与 dealloc |
| `torch/csrc/Storage.cpp` | L105-L250 | THPStorage_pynew 构造 |
| `torch/csrc/StorageSharing.cpp` | - | 共享内存实现 |
| `c10/core/StorageImpl.h` | - | StorageImpl 完整定义 |
| `c10/core/CPUAllocator.cpp` | - | CPU 分配器 |
| `c10/cuda/CUDACachingAllocator.h` | - | CUDA 缓存分配器 |

---

## 9. 下一步

| 章节 | 主题 |
|------|------|
| [Part 5](./05-factory-functions.md) | 工厂函数详解 |
| [Part 6](./06-dispatcher.md) | 分发机制 |

---

**参考资料**:
- `torch/csrc/Storage.cpp` - Python Storage 绑定
- `c10/core/Storage.h` - C++ Storage 定义
- `c10/core/StorageImpl.h` - StorageImpl 实现
- `c10/core/CPUAllocator.cpp` - CPU 内存分配
- `c10/cuda/CUDACachingAllocator.h` - CUDA 缓存分配
