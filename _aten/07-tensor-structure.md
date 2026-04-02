# ATen - 核心算子库（七）：Tensor 核心数据结构

> **前序**: [Part 6 - AT_DISPATCH 宏](./06-dispatch-macro.md)  
> **核心源码**: `c10/core/TensorImpl.h`, `c10/core/Storage.h`, `c10/core/TensorOptions.h`

---

## 1. Tensor 数据结构概览

### 1.1 核心设计：Tensor 与 TensorImpl 分离

PyTorch 的 Tensor 采用**高层包装 + 底层实现**的分离设计：

```
┌─────────────────────────────────────────────────────────┐
│                    Python Tensor                         │
│                   (torch.Tensor)                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                     C++ Tensor                           │
│                   (at::Tensor)                          │
│                  高级 API 包装层                           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    TensorImpl                            │
│             (c10::TensorImpl)                           │
│          元数据 + Storage 指针                            │
│  - sizes/strides  - dtype    - device                  │
│  - storage_       - options  - version_counter         │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                     Storage                              │
│                  (c10::Storage)                         │
│              实际内存管理                                │
│  - data_ptr  - allocator  - nbytes                     │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    实际数据内存                           │
│              (CPU/CUDA/MPS Memory)                      │
└─────────────────────────────────────────────────────────┘
```

### 1.2 为什么分离？

**Tensor 轻量**：
- Tensor 本身只包含指向 TensorImpl 的指针
- 复制 Tensor 只是复制指针（浅拷贝）
- 多个 Tensor 可共享同一 TensorImpl

**TensorImpl 共享**：
- View 操作（transpose, slice）共享底层数据
- 只修改 TensorImpl 的 sizes/strides
- 无需重新分配内存

---

## 2. TensorImpl 核心结构

### 2.1 类定义

**源码**: `c10/core/TensorImpl.h` (L440-L500+)

```cpp
/**
 * TensorImpl 是 Tensor 的低层表示，包含：
 * - 指向 Storage 的指针（实际数据）
 * - 描述 Tensor 视图的元数据（sizes, strides, offset）
 */
struct C10_API TensorImpl : public intrusive_ptr_target {
  // 构造函数
  TensorImpl(
      StorageImpl&& storage,
      DispatchKeySet dispatch_keyset,
      const c10::optional<caffe2::TypeMeta>& dtype_opt = c10::nullopt,
      const c10::optional<Device>& device_opt = c10::nullopt);
  
  // 禁止拷贝，只能移动
  TensorImpl(const TensorImpl&) = delete;
  TensorImpl& operator=(const TensorImpl&) = delete;
  TensorImpl(TensorImpl&&) = delete;
  TensorImpl& operator=(TensorImpl&&) = delete;
  
  // 核心方法
  virtual ~TensorImpl();
  
  // 数据访问
  void* mutable_data() const;
  const void* data() const;
  template <typename T> T* data_ptr();
  template <typename T> const T* data_ptr() const;
  
  // 元数据
  IntArrayRef sizes() const;
  IntArrayRef strides() const;
  int64_t dim() const;
  int64_t numel() const;
  ScalarType scalar_type() const;
  Device device() const;
  
  // Storage 访问
  StorageImpl& storage() const;
  c10::StorageImpl* storage_impl() const;
  void set_storage(Storage storage);
  
  // 视图操作
  void set_sizes_and_strides(IntArrayRef sizes, IntArrayRef strides);
  void set_storage_offset(c10::SymInt storage_offset);
  
  // Autograd 版本计数
  void bump_version() const;
  uint32_t current_version() const;
  
  // ... 更多方法
};
```

### 2.2 核心成员变量

```cpp
// 简化的 TensorImpl 成员结构
struct TensorImpl {
  // ===== Storage 相关 =====
  StorageImpl* storage_;        // 指向底层存储
  size_t storage_offset_ = 0;   // Storage 中的偏移量
  
  // ===== 形状相关 =====
  SizesAndStrides sizes_and_strides_;  // sizes 和 strides
  
  // ===== 类型和设备 =====
  DispatchKeySet dispatch_keyset_;     // DispatchKey 集合
  caffe2::TypeMeta dtype_;             // 数据类型
  Device device_;                      // 设备
  
  // ===== Autograd 相关 =====
  mutable VariableVersion version_counter_;  // 版本计数器
  AutogradMeta* autograd_meta_ = nullptr;    // 自动微分元数据
  
  // ===== 引用计数 =====
  std::atomic<size_t> ref_count_;      // 引用计数
};
```

### 2.3 关键特性

** intrusive_ptr 引用计数**:
```cpp
// TensorImpl 使用 intrusive_ptr 进行引用计数
// 优点：控制块与对象在一起，减少一次内存分配
intrusive_ptr<TensorImpl> impl = make_intrusive<TensorImpl>(...);
```

**VariableVersion 版本计数器**:
```cpp
// 源码：TensorImpl.h (L329-L419)
struct VariableVersion {
  struct VersionCounter : intrusive_ptr_target {
    std::atomic<uint32_t> version_;
  };
  c10::intrusive_ptr<VersionCounter> version_counter_;
  
  void bump() {
    if (version_counter_) {
      ++version_counter_->version_;  // inplace 操作时递增
    }
  }
  
  uint32_t current_version() const {
    TORCH_CHECK(version_counter_, "Inference tensors do not track version counter.");
    return version_counter_->version_;
  }
};
```

**作用**: 检测 inplace 操作导致的张量失效

---

## 3. Storage: 存储管理

### 3.1 Storage 结构

**源码**: `c10/core/Storage.h` (L25-L150+)

```cpp
struct C10_API Storage {
  // 默认构造
  Storage() = default;
  
  // 从 StorageImpl 构造
  Storage(c10::intrusive_ptr<StorageImpl> ptr)
      : storage_impl_(std::move(ptr)) {}
  
  // 分配新 Storage
  Storage(
      use_byte_size_t /*use_byte_size*/,
      const SymInt& size_bytes,
      Allocator* allocator = nullptr,
      bool resizable = false);
  
  // 访问数据
  const void* data() const;
  void* mutable_data() const;
  at::DataPtr& mutable_data_ptr() const;
  const at::DataPtr& data_ptr() const;
  
  // 设置数据指针
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) const;
  
  // StorageImpl 访问
  StorageImpl* storage_impl() const { return storage_impl_.get(); }
  
 private:
  c10::intrusive_ptr<StorageImpl> storage_impl_;
};
```

### 3.2 StorageImpl

**核心成员**:
```cpp
struct StorageImpl {
  // 数据指针
  at::DataPtr data_ptr_;
  
  // 内存大小
  size_t nbytes_;
  SymInt sym_nbytes_;
  
  // 分配器
  Allocator* allocator_;
  
  // 是否可调整大小
  bool resizable_;
  
  // 删除器 (用于释放自定义内存)
  std::function<void(void*)> deleter_;
};
```

### 3.3 DataPtr: 智能指针

**DataPtr** 是 PyTorch 的内存管理智能指针：

```cpp
struct DataPtr {
  void* ptr_;                    // 原始指针
  at::DeleterFnPtr deleter_;     // 删除函数
  at::Device device_;            // 设备信息
  at::Allocator* allocator_;     // 分配器
  
  ~DataPtr() {
    if (ptr_ && deleter_) {
      deleter_(ptr_);  // 自定义删除逻辑
    }
  }
};
```

---

## 4. SizesAndStrides: 形状和步长

### 4.1 什么是 Stride？

**Stride (步长)**: 访问相邻维度元素需要跳过的元素数量。

**示例**: 2D Tensor (3x4)
```cpp
// Tensor: [[0, 1, 2, 3],
//          [4, 5, 6, 7],
//          [8, 9, 10, 11]]

// 内存布局 (连续):
// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

// sizes  = [3, 4]
// strides = [4, 1]
// 含义：
// - 第 0 维 (行): 移动到下一行需要跳过 4 个元素
// - 第 1 维 (列): 移动到下一列需要跳过 1 个元素

// 计算元素 [i, j] 的偏移量:
offset = i * strides[0] + j * strides[1]
       = i * 4 + j * 1
```

### 4.2 视图操作的 Stride 变化

**Transpose (转置)**:
```cpp
// 原始 Tensor: sizes=[3, 4], strides=[4, 1]
auto t = tensor.t();
// 转置后：sizes=[4, 3], strides=[1, 4]
// 只修改 metadata，不复制数据
```

**Slice (切片)**:
```cpp
// 原始 Tensor: sizes=[10], strides=[1]
auto s = tensor.narrow(0, 2, 5);  // [2:7]
// 切片后：sizes=[5], strides=[1], storage_offset=2
// storage_offset 指向 Storage 中的起始位置
```

**View (重塑)**:
```cpp
// 原始 Tensor: sizes=[2, 6], strides=[6, 1], numel=12
auto v = tensor.view(3, 4);
// 重塑后：sizes=[3, 4], strides=[4, 1]
// numel 不变，stride 重新计算
```

### 4.3 连续与非连续

**连续 Tensor**:
```cpp
bool is_contiguous() const {
  // 检查 stride 是否匹配连续布局
  int64_t expected_stride = 1;
  for (int i = dim() - 1; i >= 0; i--) {
    if (strides()[i] != expected_stride) return false;
    expected_stride *= sizes()[i];
  }
  return true;
}
```

**非连续 Tensor**:
```cpp
// transpose 后的 Tensor 通常是非连续的
auto t = tensor.t();  // 非连续
auto c = t.contiguous();  // 复制为连续
```

---

## 5. TensorOptions: 配置选项

### 5.1 TensorOptions 结构

**源码**: `c10/core/TensorOptions.h`

```cpp
class TensorOptions {
 public:
  // 链式 API
  TensorOptions dtype(ScalarType dtype) const;
  TensorOptions device(Device device) const;
  TensorOptions layout(Layout layout) const;
  TensorOptions requires_grad(bool requires_grad) const;
  TensorOptions pin_memory(bool pin_memory) const;
  
  // 组合使用
  Tensor tensor = torch::ones(
      {3, 4},
      torch::TensorOptions()
          .dtype(torch::kFloat32)
          .device(torch::kCUDA, 0)
          .requires_grad(true));
  
 private:
  caffe2::TypeMeta dtype_;
  Device device_;
  Layout layout_ = kStrided;
  bool requires_grad_ = false;
  bool pin_memory_ = false;
};
```

### 5.2 从 Tensor 派生 Options

```cpp
// 从现有 Tensor 获取 options
auto opts = tensor.options();

// 创建相同 dtype/device 的新 Tensor
auto new_tensor = torch::empty({3, 4}, opts);

// 修改部分选项
auto float_tensor = tensor.to(torch::kFloat32);
// 或
auto new_opts = opts.dtype(torch::kFloat32);
auto float_tensor = torch::empty({3, 4}, new_opts);
```

---

## 6. 视图机制详解

### 6.1 什么是视图？

**视图 (View)**: 共享底层 Storage 的 Tensor，只修改 metadata。

**视图操作**:
- `view()`, `reshape()` - 重塑形状
- `transpose()`, `permute()` - 转置/重排维度
- `narrow()`, `slice()` - 切片
- `select()` - 选择特定索引

### 6.2 视图 vs 拷贝

```python
# Python 示例
import torch

# 原始 Tensor
a = torch.randn(3, 4)  # 分配新内存

# 视图操作 (不复制数据)
b = a.view(4, 3)        # 共享内存
c = a.t()               # 共享内存
d = a[:, 1:3]           # 共享内存

# 拷贝操作 (分配新内存)
e = a.clone()           # 深拷贝
f = a.contiguous()      # 如果 a 非连续则拷贝
g = a * 2               # 计算结果存储在新 Tensor
```

### 6.3 C++ 实现

**View 操作**:
```cpp
// TensorImpl::set_sizes_and_strides
void TensorImpl::set_sizes_and_strides(
    IntArrayRef sizes,
    IntArrayRef strides) {
  
  // 更新 sizes 和 strides
  sizes_and_strides_.set_sizes(sizes);
  sizes_and_strides_.set_strides(strides);
  
  // Storage 不变，仍然共享同一块内存
}
```

**View 的安全性检查**:
```cpp
// 检查视图是否合法
TORCH_CHECK(
    numel() == storage().nbytes() / itemsize(),
    "view size is not compatible with input tensor's size and stride");
```

---

## 7. Tensor 生命周期

### 7.1 创建 Tensor

**C++ 创建**:
```cpp
// 1. 从 Storage 创建
auto storage = Storage(Storage::use_byte_size_t(), 1024, allocator);
auto impl = make_intrusive<TensorImpl>(
    std::move(storage),
    DispatchKeySet(DispatchKey::CPU),
    ScalarType::Float,
    Device(kCPU));
auto tensor = at::Tensor(impl);

// 2. 使用工厂函数
auto tensor = at::ones({3, 4}, TensorOptions().dtype(kFloat32));

// 3. 从已有数据创建 (不拥有所有权)
float data[] = {1.0, 2.0, 3.0, 4.0};
auto tensor = torch::from_blob(data, {2, 2}, kFloat32);
```

### 7.2 复制 Tensor

**浅拷贝 (默认)**:
```cpp
at::Tensor a = at::ones({3, 4});
at::Tensor b = a;  // 浅拷贝，共享 TensorImpl

b[0][0] = 2.0;
// a[0][0] 也会变成 2.0
```

**深拷贝**:
```cpp
at::Tensor a = at::ones({3, 4});
at::Tensor b = a.clone();  // 深拷贝，独立内存

b[0][0] = 2.0;
// a[0][0] 仍然是 1.0
```

### 7.3 销毁 Tensor

**引用计数归零时销毁**:
```cpp
// TensorImpl 使用引用计数
{
  auto a = at::ones({3, 4});
  auto b = a;  // ref_count = 2
}  // a 和 b 超出作用域，ref_count = 0，释放 Storage
```

**Storage 释放时机**:
```cpp
// 当所有共享该 Storage 的 Tensor 都被销毁时
// Storage 的 ~StorageImpl() 被调用
// DataPtr 的 deleter 被调用，释放实际内存
```

---

## 8. 特殊 Tensor 类型

### 8.1 未定义 Tensor (Undefined Tensor)

```cpp
at::Tensor t;  // 未定义 Tensor
TORCH_CHECK(!t.defined());  // true

// 未定义 Tensor 的特点:
// - 没有 TensorImpl
// - 没有 Storage
// - 不能访问数据
```

### 8.2 推断 Tensor (Inference Tensor)

```cpp
// 推断模式下的 Tensor 不跟踪版本计数器
at::InferenceMode guard;
auto tensor = at::ones({3, 4});
// tensor 是推断 Tensor，version_counter 被禁用
```

### 8.3 Meta Tensor

```cpp
// Meta Tensor 只有 metadata，没有实际数据
auto meta_tensor = at::empty({3, 4}, device(kMeta));
TORCH_CHECK(meta_tensor.device().type() == kMeta);

// 用途：
// - 延迟分配 (lazy allocation)
// - 形状推断 (shape inference)
// - 内存规划 (memory planning)
```

---

## 9. 内存格式

### 9.1 连续格式 (Contiguous)

```cpp
// Contiguous: 元素在内存中连续存储
auto t = torch::randn({3, 4});
TORCH_CHECK(t.is_contiguous());

// 内存布局 (行优先):
// [row0_col0, row0_col1, row0_col2, row0_col3,
//  row1_col0, row1_col1, ...]
```

### 9.2 Channels Last (NHWC)

```cpp
// 4D Tensor 的 Channels Last 格式
// NCHW (连续): [N, C, H, W], strides = [C*H*W, H*W, W, 1]
// NHWC (channels_last): [N, C, H, W], strides = [H*W*C, 1, W*C, C]

auto t = torch::randn({32, 64, 224, 224});
auto nhwc = t.to(memory_format=torch::ChannelsLast);

// NHWC 对 GPU 卷积操作更高效
```

### 9.3 内存格式转换

```cpp
// 转换为特定内存格式
auto contiguous = t.to(memory_format=torch::Contiguous);
auto channels_last = t.to(memory_format=torch::ChannelsLast);
auto channels_last_3d = t.to(memory_format=torch::ChannelsLast3d);
```

---

## 10. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `c10/core/TensorImpl.h` | L329-L419 | VariableVersion 结构 |
| `c10/core/TensorImpl.h` | L440-L500 | TensorImpl 类定义 |
| `c10/core/Storage.h` | L25-L150 | Storage 结构 |
| `c10/core/StorageImpl.h` | L20-L100 | StorageImpl 成员 |
| `c10/core/TensorOptions.h` | L15-L80 | TensorOptions 类 |
| `c10/core/impl/SizesAndStrides.h` | L1-L200 | SizesAndStrides 实现 |

---

## 11. 下一步

| Part | 主题 |
|------|------|
| [Part 8](./08-autograd-integration.md) | 自动微分集成 |
| [Part 9](./09-backend-extension.md) | 后端扩展指南 |

---

**参考资料**:
- `c10/core/TensorImpl.h` - TensorImpl 完整定义
- `c10/core/Storage.h` - Storage 完整定义
- `c10/core/TensorOptions.h` - TensorOptions 完整定义
- `aten/src/ATen/core/TensorBase.h` - TensorBase 定义
