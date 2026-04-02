# ATen - 核心算子库（十）：调试与测试

> **前序**: [Part 9 - 后端扩展指南](./09-backend-extension.md)  
> **核心源码**: `torch/testing/_internal/common_utils.py`, `test/test_ops.py`

---

## 1. ATen 测试框架概览

### 1.1 测试目录结构

```
test/
├── test_ops.py                   # 算子正确性测试
├── test_torch.py                 # 综合功能测试
├── test_autograd.py              # 自动微分测试
├── test_jit.py                   # JIT 编译测试
├── cpp/                          # C++ 测试
│   └── api/                      # C++ API 测试
└── inductor/                     # Inductor 测试
    └── test_torchinductor.py     # Inductor 算子测试
```

### 1.2 测试基础设施

**源码**: `torch/testing/_internal/common_utils.py`

```python
# 核心测试类
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    def setUp(self):
        super().setUp()
        # 测试前准备
        
    def tearDown(self):
        # 测试后清理
        super().tearDown()
    
    def test_example(self):
        # 测试逻辑
        pass

if __name__ == "__main__":
    run_tests()
```

---

## 2. 算子正确性测试

### 2.1 使用 OpInfo 进行测试

**OpInfo (Operator Information)** 是 PyTorch 的算子测试元数据系统。

**源码**: `test/test_ops.py`

```python
from torch.testing._internal.common_methods_invocations import op_db, OpInfo

# OpInfo 包含：
# - 算子名称
# - 测试样例生成器
# - 支持的 dtype
# - 装饰器 (skip/xfail)
```

### 2.2 编写算子测试

```python
import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import op_db

class TestATen(TestCase):
    def test_abs(self):
        # 基本测试
        a = torch.randn(3, 3)
        result = torch.abs(a)
        expected = a.abs()
        self.assertEqual(result, expected)
    
    def test_abs_cuda(self):
        # CUDA 测试
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        a = torch.randn(3, 3, device='cuda')
        result = torch.abs(a)
        self.assertEqual(result.device.type, 'cuda')
    
    def test_abs_backward(self):
        # 反向传播测试
        a = torch.randn(3, 3, requires_grad=True)
        result = torch.abs(a)
        result.sum().backward()
        
        # 检查梯度
        self.assertIsNotNone(a.grad)
        self.assertEqual(a.grad.shape, a.shape)

if __name__ == "__main__":
    run_tests()
```

### 2.3 使用 make_tensor 生成测试数据

```python
from torch.testing import make_tensor

# 生成随机 Tensor
a = make_tensor(
    (3, 3),           # 形状
    dtype=torch.float32,
    device='cpu',
    low=-10,          # 最小值
    high=10,          # 最大值
    requires_grad=False,
    noncontiguous=False,  # 是否非连续
)

# 生成特定 dtype
b = make_tensor((2, 2), dtype=torch.int64)

# 生成复数
c = make_tensor((2, 2), dtype=torch.complex64)
```

---

## 3. 精度测试与断言

### 3.1 assertEqual 断言

```python
from torch.testing._internal.common_utils import TestCase

class TestPrecision(TestCase):
    def test_tensor_equal(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        self.assertEqual(a, b)  # 精确相等
    
    def test_tensor_close(self):
        # 浮点数比较（考虑精度）
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0 + 1e-7, 2.0 + 1e-7, 3.0 + 1e-7])
        
        # 默认 rtol=1e-5, atol=1e-8
        self.assertEqual(a, b)
        
        # 自定义容差
        self.assertEqual(a, b, rtol=1e-6, atol=1e-9)
```

### 3.2 精度装饰器

```python
from torch.testing._internal.common_utils import TestCase, precisionOverride

class TestPrecisionOverride(TestCase):
    @precisionOverride(1e-6)  # 设置精度为 1e-6
    def test_low_precision(self):
        # 测试使用 1e-6 精度
        pass
```

### 3.3 不同 dtype 的测试

```python
from torch.testing._internal.common_dtype import get_all_dtypes

class TestDtypes(TestCase):
    def test_all_dtypes(self):
        for dtype in get_all_dtypes():
            a = torch.randn(3, 3, dtype=dtype)
            result = torch.abs(a)
            self.assertEqual(result.dtype, dtype)
```

---

## 4. 多设备测试

### 4.1 设备装饰器

```python
from torch.testing._internal.common_device_type import (
    onlyCPU,
    onlyCUDA,
    onlyMPS,
    instantiate_device_type_tests,
)

class TestDevice(TestCase):
    @onlyCPU
    def test_cpu_only(self):
        # 只在 CPU 上运行
        pass
    
    @onlyCUDA
    def test_cuda_only(self):
        # 只在 CUDA 上运行
        pass

# 参数化设备测试
@instantiate_device_type_tests(globals())
class TestMultiDevice(TestCase):
    def test_abs(self, device):
        # device 参数由框架注入
        a = torch.randn(3, 3, device=device)
        result = torch.abs(a)
        self.assertEqual(result.device.type, device)
```

### 4.2 跳过特定设备

```python
from torch.testing._internal.common_device_type import skipMPS, skipCUDA

class TestSkipDevice(TestCase):
    @skipMPS("MPS does not support this operation")
    def test_not_supported_mps(self):
        pass
    
    @skipCUDA("Requires ROCm")
    def test_rocm_only(self):
        pass
```

---

## 5. 性能测试

### 5.1 基准测试框架

```python
import torch
import time

def benchmark_op(op, *args, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        op(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时
    start = time.perf_counter()
    for _ in range(iterations):
        op(*args)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms

# 使用示例
a = torch.randn(1024, 1024, device='cuda')
ms = benchmark_op(torch.matmul, a, a)
print(f"matmul: {ms:.3f} ms")
```

### 5.2 使用 torch.utils.benchmark

```python
import torch.utils.benchmark as benchmark

# 创建 Timer
t0 = benchmark.Timer(
    stmt="torch.matmul(a, b)",
    setup="a = torch.randn(1024, 1024, device='cuda'); b = a.clone()",
    globals={"torch": torch},
)

# 运行基准测试
result = t0.blocked_autorange(min_run_time=1)
print(f"Mean: {result.median*1000:.3f} ms")
print(f"IQR: {result.iqr*1000:.3f} ms")
```

---

## 6. 调试工具

### 6.1 启用调试输出

```python
import os
import torch

# 通用调试
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'

# Dispatcher 调试
os.environ['TORCH_SHOW_DISPATCH_STACK'] = '1'

# 算子选择调试
os.environ['TORCH_SELECTIVE_BUILD'] = '0'

# 内存调试
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'

# 重新运行有问题的代码
x = torch.randn(3, 3)
y = torch.abs(x)
```

### 6.2 查看 Dispatch 表

```python
from torch._python_dispatcher import PythonDispatcher

dispatcher = PythonDispatcher()
dispatcher.register(["CPU", "CUDA", "MPS", "CompositeImplicitAutograd"])

# 查看算子的 dispatch 表
print(dispatcher.dispatchTable("aten::abs"))

# 输出示例:
# {
#   'CPU': <function abs_cpu at 0x...>,
#   'CUDA': <function abs_cuda at 0x...>,
#   'MPS': <function abs_mps at 0x...>,
# }
```

### 6.3 查看编译产物

```python
# Inductor 调试
import torch

# 保存调试信息
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_COMPILE_DEBUG_DIR'] = '/tmp/torch_debug'

# 使用 torch.compile
@torch.compile
def f(x):
    return x.abs() + 1

x = torch.randn(3, 3)
f(x)  # 会在 /tmp/torch_debug 生成调试文件
```

---

## 7. 常见错误排查

### 7.1 算子未实现

```
RuntimeError: Could not run 'aten::abs' with arguments from the 'CUDA' backend.

原因：CUDA 后端没有实现 abs 算子

排查步骤:
1. 检查 native_functions.yaml 中的 dispatch 配置
2. 查看 CUDA Kernel 是否注册
3. 使用 PythonDispatcher 查看 dispatch 表

解决:
- 实现缺失的 Kernel
- 或添加 BackendFallback
```

### 7.2 梯度计算错误

```
RuntimeError: element 0 of tensors does not require grad

原因：输入 Tensor 没有设置 requires_grad=True

排查步骤:
1. 检查输入是否需要梯度
2. 确认 derivatives.yaml 配置正确
3. 查看 autograd graph

解决:
x = torch.randn(3, 3, requires_grad=True)
```

### 7.3 精度问题

```
AssertionError: Tensor-likes are not close!

原因：浮点数精度不足

排查步骤:
1. 检查 dtype 是否匹配
2. 调整 rtol/atol 容差
3. 确认算法数值稳定性

解决:
self.assertEqual(a, b, rtol=1e-4, atol=1e-5)
```

### 7.4 设备不匹配

```
RuntimeError: Expected all tensors to be on the same device

原因：多 Tensor 参数在不同设备上

排查步骤:
1. 打印所有输入 Tensor 的 device
2. 检查代码中是否有隐式 CPU 创建

解决:
# 确保所有 Tensor 在同一设备
x = x.to(target_device)
y = y.to(target_device)
```

---

## 8. C++ 测试

### 8.1 C++ 测试框架

```cpp
// test/cpp/api/test_tensor.cpp
#include <gtest/gtest.h>
#include <torch/torch.h>

TEST(TensorTest, AbsTest) {
    auto a = torch::randn({3, 3});
    auto b = torch::abs(a);
    
    // 检查形状
    EXPECT_EQ(b.sizes(), std::vector<int64_t>({3, 3}));
    
    // 检查值
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_GE(b[i][j].item<float>(), 0);
        }
    }
}

TEST(TensorTest, CUDATest) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }
    
    auto a = torch::randn({3, 3}, torch::kCUDA);
    auto b = torch::abs(a);
    
    EXPECT_EQ(b.device().type(), torch::kCUDA);
}
```

### 8.2 运行 C++ 测试

```bash
# 构建测试
python setup.py develop

# 运行 C++ 测试
./build/bin/test_api

# 运行特定测试
./build/bin/test_api --gtest_filter="TensorTest.AbsTest"
```

---

## 9. OpInfo 测试数据库

### 9.1 OpInfo 结构

```python
from torch.testing._internal.common_methods_invocations import OpInfo

op_info = OpInfo(
    torch.abs,                          # 算子
    sample_inputs_func=sample_abs,      # 样例生成器
    dtypes=(torch.float32, torch.int32),  # 支持 dtype
    decorators=(onlyCUDA,),             # 装饰器
)
```

### 9.2 添加新的 OpInfo

```python
from torch.testing._internal.common_methods_invocations import OpInfo, make_test

def sample_my_op(op, device, dtype, requires_grad, **kwargs):
    # 生成测试样例
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    
    # 生成不同形状的输入
    for shape in ((3, 3), (2, 3, 4), (1, 2, 3, 4)):
        yield SampleInput(make_arg(shape))

# 添加到 op_db
op_db.append(
    OpInfo(
        "my_op",
        sample_inputs_func=sample_my_op,
        dtypes=(torch.float32, torch.float64),
    )
)
```

---

## 10. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `common_utils.py` | L1242-L1280 | run_tests 函数 |
| `common_utils.py` | L3153-L3250 | TestCase 类定义 |
| `test_ops.py` | L1-L100 | 算子测试入口 |
| `test_torchinductor.py` | L1-L200 | Inductor 测试示例 |
| `common_device_type.py` | L1-L100 | 设备装饰器 |
| `common_dtype.py` | L1-L100 | dtype 工具函数 |
| `common_methods_invocations.py` | L1-L200 | OpInfo 定义 |

---

## 11. 测试最佳实践

### 11.1 测试覆盖

```
- [ ] 基本功能测试
- [ ] 边界条件测试
- [ ] 多 dtype 测试
- [ ] 多设备测试
- [ ] 反向传播测试
- [ ] 非连续 Tensor 测试
- [ ] 广播测试
```

### 11.2 测试命名规范

```python
def test_{op_name}():           # 基本测试
def test_{op_name}_{dtype}():   # dtype 特定测试
def test_{op_name}_{device}():  # 设备特定测试
def test_{op_name}_backward():  # 反向传播测试
def test_{op_name}_gradcheck(): # 梯度检查
```

### 11.3 CI 集成

```yaml
# .github/workflows/test.yml
test:
  - name: test_ops
    command: python test/test_ops.py
    
  - name: test_autograd
    command: python test/test_autograd.py
    
  - name: test_cpp_api
    command: ./build/bin/test_api
```

---

## 12. 下一步

ATen 系列文档已全部完成！建议阅读顺序：

1. [Part 1 - 架构概览](./01-architecture.md)
2. [Part 2 - 算子声明系统](./02-native-functions-yaml.md)
3. [Part 3 - Dispatch 机制](./03-dispatch-mechanism.md)
4. [Part 4 - C++ Kernel 实现](./04-kernel-implementation.md)
5. [Part 5 - 注册机制](./05-registration.md)
6. [Part 6 - AT_DISPATCH 宏](./06-dispatch-macro.md)
7. [Part 7 - Tensor 数据结构](./07-tensor-structure.md)
8. [Part 8 - 自动微分集成](./08-autograd-integration.md)
9. [Part 9 - 后端扩展指南](./09-backend-extension.md)
10. [Part 10 - 调试与测试](./10-debugging-testing.md)

---

**参考资料**:
- `torch/testing/_internal/common_utils.py` - 测试框架核心
- `test/test_ops.py` - 算子测试示例
- `test/inductor/test_torchinductor.py` - Inductor 测试示例
- [PyTorch 测试指南](https://pytorch.org/docs/stable/community/test_policy.html)
