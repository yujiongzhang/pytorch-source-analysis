# Python 层 Tensor API（三）：自动微分集成

> **前序**: [Part 2 - Tensor 构造与工厂函数](./02-tensor-creation.md)
> **核心源码**: `torch/csrc/autograd/engine.cpp`, `torch/csrc/autograd/python_engine.cpp`, `torch/autograd/__init__.py`

---

## 1. Autograd 架构概览

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│              Python 层 (torch.autograd)                   │
│  - backward()                                            │
│  - grad()                                                │
│  - Function 类                                           │
│  - grad_mode (enable_grad/no_grad/inference_mode)        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              C++ 绑定层 (torch/csrc/autograd)            │
│  - python_engine.cpp - PythonEngine 实现                  │
│  - python_variable.cpp - Variable 绑定                    │
│  - python_cpp_function.cpp - C++ Function 绑定            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              C++ 引擎层 (torch/csrc/autograd)            │
│  - engine.cpp - 反向传播引擎                              │
│  - function.cpp - Autograd Function 基类                  │
│  - variable.cpp - Variable 实现                          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              ATen 层 (aten/src/ATen)                     │
│  - derivatives.yaml - 导数公式定义                        │
│  - AutogradComposite.cpp - 复合实现                       │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心概念

| 概念 | 说明 |
|------|------|
| **Variable** | 包装 Tensor，追踪计算历史 |
| **Function** | 计算图中的节点，定义前向和反向计算 |
| **Edge** | 连接 Function 的边，包含梯度和连接信息 |
| **GraphTask** | 一次反向传播任务的状态 |
| **ReadyQueue** | 待执行的 Function 队列 |
| **InputBuffer** | 累积输入梯度的缓冲区 |

---

## 2. Python 层 API

### 2.1 torch.autograd.backward()

**源码**: `torch/autograd/__init__.py` (L230-L300)

```python
def backward(
    tensors: _TensorOrTensorsOrGradEdge,
    grad_tensors: _TensorOrOptionalTensors = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrOptionalTensors] = None,
    inputs: Optional[_TensorOrTensorsOrGradEdge] = None,
) -> None:
    r"""Computes the sum of gradients of given tensors with respect to graph leaves.

    Args:
        tensors (Tensor or sequence of Tensors or GradientEdge): Tensors of which
            the derivative will be computed. GradientEdge is an output of a
            differentiable function with the corresponding output index that
            the derivative will be computed.
        grad_tensors (Tensor or sequence of Tensors, optional): The "vector"
            in the Jacobian-vector product. Usually gradients w.r.t.
            each element of corresponding tensors.
        retain_graph (bool, optional): If ``False``, the graph used to compute
            the grads will be freed. Defaults to ``None``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
        inputs (Tensor or sequence of Tensors or GradientEdge, optional): Inputs
            w.r.t. which the gradient will be accumulated into .grad.
    """
    # 处理 grad_variables 废弃参数
    if grad_variables is not None:
        warnings.warn(...)
        grad_tensors = grad_variables

    # 统一处理为序列
    tensors = (tensors,) if isinstance(tensors, (torch.Tensor, graph.GradientEdge)) else tensors

    # 创建 grad_tensors
    if grad_tensors is None:
        # 为每个输出创建全 1 梯度
        grad_tensors = torch._C._engine_base._make_default_grads(tensors)
    elif isinstance(grad_tensors, torch.Tensor):
        grad_tensors = (grad_tensors,)

    # 执行反向传播
    torch.autograd.graph._engine_run_backward(
        tensors=tensors,
        grad_tensors=grad_tensors,
        retain_graph=retain_graph,
        create_graph=create_graph,
        inputs=inputs,
        allow_unreachable=False,
        accumulate_grad=True
    )
```

### 2.2 torch.autograd.grad()

**源码**: `torch/autograd/__init__.py` (L350-L450)

```python
def grad(
    outputs: _TensorOrTensorsOrGradEdge,
    inputs: _TensorOrTensorsOrGradEdge,
    grad_outputs: _TensorOrOptionalTensors = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> tuple[torch.Tensor, ...]:
    r"""Computes and returns the sum of gradients of outputs with respect to inputs.

    Returns:
        A tuple of gradients w.r.t. each input.
    """
    inputs = (inputs,) if isinstance(inputs, (torch.Tensor, graph.GradientEdge)) else inputs
    outputs = (outputs,) if isinstance(outputs, (torch.Tensor, graph.GradientEdge)) else outputs

    if grad_outputs is None:
        grad_outputs = torch._C._engine_base._make_default_grads(outputs)
    elif isinstance(grad_outputs, torch.Tensor):
        grad_outputs = (grad_outputs,)

    # 执行反向传播并返回梯度
    return torch.autograd.graph._engine_run_backward(
        tensors=outputs,
        grad_tensors=grad_outputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        inputs=inputs,
        allow_unreachable=allow_unused,
        accumulate_grad=False
    )
```

### 2.3 _engine_run_backward()

**源码**: `torch/autograd/graph.py`

```python
def _engine_run_backward(
    *,
    tensors,
    grad_tensors,
    retain_graph,
    create_graph,
    inputs,
    allow_unreachable,
    accumulate_grad
):
    """调用 C++ 引擎执行反向传播"""
    return torch._C._engine_base.run_backward(
        tensors=tensors,
        grad_tensors=grad_tensors,
        keep_graph=retain_graph,
        create_graph=create_graph,
        inputs=inputs,
        allow_unreachable=allow_unreachable,
        accumulate_grad=accumulate_grad
    )
```

---

## 3. C++ 层引擎实现

### 3.1 PythonEngine 类

**源码**: `torch/csrc/autograd/python_engine.cpp` (L35-L54)

```cpp
namespace torch::autograd::python {

PythonEngine::PythonEngine() = default;

Engine& PythonEngine::get_python_engine() {
  static PythonEngine engine;
  if (_reinitialize_engine) {
    engine.release_workers();
    engine.~PythonEngine();
    new (&engine) torch::autograd::python::PythonEngine();
    _reinitialize_engine = false;
  }
  return engine;
}

// 线程初始化 - 获取 GIL
void PythonEngine::thread_init(
    int device,
    const std::shared_ptr<ReadyQueue>& ready_queue,
    bool should_increment) {

  if (should_increment) {
    increment_non_reentrant_thread_count();
  }

  // 创建 PyThreadState，但释放 GIL
  auto gil = std::make_unique<pybind11::gil_scoped_acquire>();
  pybind11::gil_scoped_release no_gil;

  Engine::thread_init(device, ready_queue, false);

  if (should_increment) {
    decrement_non_reentrant_thread_count();
  }
}

// 异常处理 - 保存 Python 异常状态
void PythonEngine::thread_on_exception(
    const std::shared_ptr<GraphTask>& graph_task,
    const std::shared_ptr<Node>& fn,
    std::exception& e) {

  auto python_err = dynamic_cast<python_error*>(&e);
  if (python_err) {
    python_err->persist();  // 保存 PyErr 状态
  }
  Engine::thread_on_exception(graph_task, fn, e);
}

} // namespace torch::autograd::python
```

### 3.2 run_backward() 实现

**源码**: `torch/csrc/autograd/python_engine.cpp` (L172-L280)

```cpp
static PyObject* THPEngine_run_backward(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {

  PyObject* tensors = nullptr;
  PyObject* grad_tensors = nullptr;
  unsigned char keep_graph = 0;
  unsigned char create_graph = 0;
  PyObject* inputs = nullptr;
  unsigned char allow_unreachable = 0;
  unsigned char accumulate_grad = 0;

  // 解析参数
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OObb|Obb",
          const_cast<char**>(accepted_kwargs),
          &tensors, &grad_tensors, &keep_graph, &create_graph,
          &inputs, &allow_unreachable, &accumulate_grad)) {
    return nullptr;
  }

  // 解析输入 tensors
  auto variables = pyobj_to_vars(tensors);
  auto input_vars = pyobj_to_vars(inputs);

  // 解析梯度
  auto grad_variables = pyobj_to_vars(grad_tensors);

  // 构建根节点边列表
  edge_list roots(variables.size());
  for (size_t i = 0; i < variables.size(); i++) {
    roots[i] = Edge(variables[i].grad_fn, variables[i].output_nr);
  }

  // 执行反向传播
  variable_list results;
  {
    pybind11::gil_scoped_release no_gil;  // 释放 GIL

    results = python::PythonEngine::get_python_engine().execute(
        roots,
        grad_variables,
        keep_graph,
        create_graph,
        accumulate_grad,
        input_vars);
  }

  // 检查 Python 异常
  if (PyErr_Occurred()) {
    throw python_error();
  }

  // 返回结果
  auto rg = pybind11::reinterpret_steal<PyObject*>(
      THPVariable_WrapList(results));

  return rg;
}
```

### 3.3 Engine::execute() 核心流程

**源码**: `torch/csrc/autograd/engine.cpp` (L800-L1000+)

```cpp
variable_list Engine::execute(
    const edge_list& roots,
    const variable_list& inputs,
    bool keep_graph,
    bool create_graph,
    bool accumulate_grad,
    const edge_list& outputs) {

  // 1. 创建 GraphTask
  auto graph_task = std::make_shared<GraphTask>(
      roots, inputs, outputs, keep_graph, create_graph, accumulate_grad);

  // 2. 初始化输入缓冲区
  InputBuffer input_buffer(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i].defined()) {
      input_buffer.add(input_edges[i].first, inputs[i].output_nr, inputs[i]);
    }
  }

  // 3. 创建图根节点
  auto graph_root = std::make_shared<GraphRoot>(roots, grad_variables);

  // 4. 执行计算图
  execute_with_graph_task(graph_task, graph_root, std::move(input_buffer));

  // 5. 收集结果
  variable_list results(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    results[i] = graph_task->output_values_[i];
  }

  return results;
}
```

---

## 4. Function 与 Node

### 4.1 Python Function 类

**源码**: `torch/autograd/function.py`

```python
class Function:
    """
    所有 autograd Function 的基类。

    子类需要实现:
    - forward(ctx, ...) -> Tensor
    - backward(ctx, grad_output) -> tuple of Tensors

    示例:
    >>> class Exp(Function):
    ...     @staticmethod
    ...     def forward(ctx, i):
    ...         result = i.exp()
    ...         ctx.save_for_backward(result)
    ...         return result
    ...     @staticmethod
    ...     def backward(ctx, grad_output):
    ...         result, = ctx.saved_tensors
    ...         return grad_output * result
    """

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        """执行 Function，设置 up 计算图"""
        # 获取 back 端实现
        backend = Function._get_backend(cls)

        # 保存输入 tensor 用于检查
        _dirty_tensors = set()
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                _dirty_tensors.add(arg)

        # 创建 context
        ctx = backend.create_context()

        # 执行前向计算
        ret = cls.forward(ctx, *args)

        # 设置 grad_fn
        if isinstance(ret, torch.Tensor) and ret.requires_grad:
            ret.grad_fn = backend
            ret._grad_fn_ctx = ctx

        return ret
```

### 4.2 C++ Node 类

**源码**: `torch/csrc/autograd/function.h`

```cpp
struct TORCH_API Node : public std::enable_shared_from_this<Node> {
  // 前向函数 (Python Function 或 C++ Function)
  virtual variable_list apply(variable_list&& inputs) = 0;

  // 访问相邻节点
  const std::vector<Edge>& next_edges() const {
    return next_edges_;
  }

  // 元数据
  std::string name() const { return name_; }
  void set_name(std::string name) { name_ = std::move(name); }

  // 输入元数据 (用于形状检查)
  struct InputMetadata {
    at::ScalarType dtype;
    std::vector<int64_t> shape;
    bool is_nested_tensor;
  };
  std::vector<InputMetadata> _input_metadata;

 protected:
  std::vector<Edge> next_edges_;  // 指向下一个节点
  std::string name_;              // 节点名称
};
```

### 4.3 CppFunction - C++ 实现的 Function

**源码**: `torch/csrc/autograd/python_cpp_function.cpp`

```cpp
// 将 C++ Function 包装为 Python 可调用的对象
struct THPCppFunction {
  PyObject_HEAD
  std::shared_ptr<torch::autograd::Node> cdata;
};

// 绑定到 Python
static PyTypeObject THPCppFunctionType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch._C._Function",                    /* tp_name */
    sizeof(THPCppFunction),                  /* tp_basicsize */
    // ... 方法定义
};
```

---

## 5. 梯度累积机制

### 5.1 InputBuffer

**源码**: `torch/csrc/autograd/input_buffer.h`

```cpp
class InputBuffer {
 public:
  InputBuffer(size_t size) : buffer_(size) {}

  // 添加梯度到缓冲区
  void add(std::shared_ptr<Node> owner, uint32_t output_nr, at::Tensor grad) {
    auto& slot = buffer_[owner->node_id()];
    if (!slot.defined()) {
      slot = grad;
    } else {
      // 累积梯度
      slot = slot + grad;
    }
  }

  // 获取累积后的梯度
  at::Tensor get(std::shared_ptr<Node> owner, uint32_t output_nr) {
    return buffer_[owner->node_id()];
  }

 private:
  std::vector<at::Tensor> buffer_;
};
```

### 5.2 梯度累积示例

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 第一次反向传播
y1 = x * 2
y1.sum().backward()
print(x.grad)  # tensor([2., 2., 2.])

# 第二次反向传播 (累积)
y2 = x * 3
y2.sum().backward()
print(x.grad)  # tensor([5., 5., 5.]) = [2,2,2] + [3,3,3]

# 清零梯度
x.grad.zero_()
```

---

## 6. 梯度模式

### 6.1 enable_grad / no_grad / inference_mode

**源码**: `torch/autograd/grad_mode.py`

```python
class enable_grad:
    """上下文管理器，启用梯度计算"""
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)


class no_grad:
    """上下文管理器，禁用梯度计算"""
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

    def __exit__(self, *args):
        torch.set_grad_enabled(self.prev)


class inference_mode:
    """推理模式 - 比 no_grad 更严格的优化"""
    def __init__(self, mode=True):
        self.mode = mode

    def __enter__(self):
        self.prev = torch.is_inference_mode_enabled()
        torch.set_inference_mode(self.mode)

    def __exit__(self, *args):
        torch.set_inference_mode(self.prev)
```

### 6.2 模式对比

| 模式 | 梯度计算 | 版本追踪 | 内存优化 |
|------|----------|----------|----------|
| 默认 | 启用 | 启用 | 无 |
| `no_grad` | 禁用 | 启用 | 中等 |
| `inference_mode` | 禁用 | 禁用 | 最大 |

---

## 7. 异常检测模式

### 7.1 detect_anomaly

**源码**: `torch/autograd/anomaly_mode.py`

```python
class detect_anomaly:
    """检测 autograd 中的异常 (NaN/Inf)"""

    def __init__(self, mode=True):
        self.mode = mode

    def __enter__(self):
        self.prev = torch.autograd.set_detect_anomaly(self.mode)

    def __exit__(self, *args):
        torch.autograd.set_detect_anomaly(self.prev)


# 使用示例
with torch.autograd.detect_anomaly():
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.log(x - 1)  # log(0) = -inf
    y.backward()  # 会抛出异常
```

### 7.2 C++ 实现

**源码**: `torch/csrc/autograd/anomaly_mode.cpp`

```cpp
namespace torch::autograd {

thread_local bool anomaly_mode_enabled = false;

bool set_detect_anomaly(bool mode) {
  bool prev = anomaly_mode_enabled;
  anomaly_mode_enabled = mode;
  return prev;
}

bool is_detect_anomaly_enabled() {
  return anomaly_mode_enabled;
}

} // namespace torch::autograd
```

---

## 8. 关键源码索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `torch/autograd/__init__.py` | L230-L300 | backward() 实现 |
| `torch/autograd/__init__.py` | L350-L450 | grad() 实现 |
| `torch/csrc/autograd/python_engine.cpp` | L172-L280 | run_backward() |
| `torch/csrc/autograd/engine.cpp` | L800-L1000+ | Engine::execute() |
| `torch/csrc/autograd/function.h` | - | Node 类定义 |
| `torch/autograd/grad_mode.py` | - | 梯度模式类 |

---

## 9. 下一步

| 章节 | 主题 |
|------|------|
| [Part 4](./04-storage-memory.md) | 存储与内存管理 |
| [Part 5](./05-factory-functions.md) | 工厂函数详解 |
| [Part 6](./06-dispatcher.md) | 分发机制 |

---

**参考资料**:
- `torch/autograd/__init__.py` - Python autograd API
- `torch/csrc/autograd/engine.cpp` - 反向传播引擎核心
- `torch/csrc/autograd/python_engine.cpp` - Python 引擎绑定
- `torch/autograd/grad_mode.py` - 梯度模式管理
