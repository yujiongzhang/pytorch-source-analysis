# PyTorch Inductor 源码解析（八）：性能优化技术

## 引言

PyTorch Inductor 通过多种性能优化技术显著提升深度学习模型的执行效率。本章详细介绍 Inductor 的核心性能优化技术，包括：

1. **内存优化**: Buffer 复用、峰值内存重排序
2. **算子融合**: Combo Kernel、外层循环融合
3. **通信优化**: 计算/通信重叠、DDP 融合
4. **Autotune**: 自动参数搜索、缓存机制
5. **并行优化**: 并行归约、混合顺序归约

**源码位置**: 分散在 `torch/_inductor/` 多个模块中

---

## 1. 内存优化技术

### 1.1 峰值内存重排序

**文件**: `torch/_inductor/memory.py`

**文件**: `torch/_inductor/memory.py:913-999`

```python
def reorder_for_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
    methods: list[Callable[..., list[BaseSchedulerNode]]] = [
        topological_sort_lpmf,
        topological_sort_bfs,
        topological_sort_dfs,
    ],
) -> list[BaseSchedulerNode]:
    """
    Try a few heuristics based topological sort algorithms, and pick the one whose
    resulting topological order has the lowest peak memory estimation.
    
    尝试多种基于启发式的拓扑排序算法，选择峰值内存估计最低的顺序
    
    Args:
        nodes: 调度节点列表
        name_to_buf: Buffer 名称到 SchedulerBuffer 的映射
        name_to_fused_node: Buffer 名称到融合节点的映射
        graph_inputs: 图输入集合
        graph_outputs: 图输出集合
        methods: 排序方法列表（默认 LPMF、BFS、DFS）
    
    Returns:
        优化后的节点顺序
    """
    torch_log.info("Reordering for peak memory -- %d nodes", len(nodes))

    # L932-938: 准备内存规划信息
    estimated_peak_memory, name_to_freeable_input_buf = prepare_planning_info(
        nodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )

    # L941-948: 导出图用于模拟器（调试模式）
    if config.reorder_for_peak_memory_debug:
        export_graph_for_simulator(
            nodes,
            name_to_freeable_input_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
        )

    # L951-957: 验证图的无环性
    try:
        validate_graph_acyclic(nodes)
        validate_unique_buffer_names(nodes, name_to_buf, name_to_freeable_input_buf)
    except RuntimeError:
        torch_log.exception("Memory planning validation failed")
        if not is_fbcode():
            raise

    # L960-963: 记录基线峰值内存
    peak_memory_diff_methods: list[PeakMemoryResult] = []
    peak_memory_diff_methods.append(
        PeakMemoryResult(nodes, estimated_peak_memory, "baseline")
    )
    torch_log.info("Baseline peak memory: %d", estimated_peak_memory)

    # L967-986: 尝试多种排序方法
    for method in methods:
        try:
            if method is topological_sort_lpmf:
                # LPMF 需要额外参数
                order = method(
                    nodes, name_to_freeable_input_buf, name_to_buf, graph_outputs
                )
            else:
                order = method(nodes)
            assert len(order) == len(nodes)
            
            # L976-978: 估计峰值内存
            peak_memory, _ = estimate_peak_memory(
                order, name_to_freeable_input_buf, graph_outputs
            )
            peak_memory_diff_methods.append(
                PeakMemoryResult(order, peak_memory, method.__name__)
            )
            torch_log.info("%s peak memory: %d", method.__name__, peak_memory)
        except Exception:
            torch_log.exception("Failed to reorder for %s", method.__name__)
            if not is_fbcode():
                raise

    # L988-994: 记录指标
    signpost_event(
        category="inductor",
        name="memory",
        parameters={
            "orm": {elem.method: elem.peak_memory for elem in peak_memory_diff_methods},
        },
    )

    # L997-999: 返回最优结果
    best_result = min(peak_memory_diff_methods, key=lambda x: x.peak_memory)
    return best_result.order
```

**关键点**:
- 第 919-923 行：默认使用三种排序方法（LPMF、BFS、DFS）
- 第 932-938 行：`prepare_planning_info` 计算 Buffer 大小和依赖关系
- 第 976-978 行：`estimate_peak_memory` 估算给定顺序的峰值内存
- 第 997 行：选择峰值内存最低的方案

### 1.2 内存规划信息

**文件**: `torch/_inductor/memory.py:40-87`

```python
@dataclasses.dataclass
class MemoryPlanningInfoForBuffer:
    """
    Buffer 内存规划信息
    
    Attributes:
        size_alloc: 分配时的内存大小（字节）
        size_free: 释放时的内存大小（字节）
        succ_nodes: 后继节点集合（用于 Buffer 生命周期管理）
        succ_nodes_for_ordering: 用于节点排序的后继集合（包含 fake WeakDeps）
    """
    size_alloc: int = 0
    size_free: int = 0
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    succ_nodes_for_ordering: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )

    def __post_init__(self) -> None:
        # L53-57: 验证 succ_nodes 是 succ_nodes_for_ordering 的子集
        torch._check(
            len(self.succ_nodes) <= len(self.succ_nodes_for_ordering),
            lambda: f"succ_nodes must be a subset of succ_nodes_for_ordering. "
            f"len(succ_nodes)={len(self.succ_nodes)}, "
            f"len(succ_nodes_for_ordering)={len(self.succ_nodes_for_ordering)}",
        )


@dataclasses.dataclass
class MemoryPlanningInfoForNode:
    """
    节点内存规划信息
    
    Attributes:
        index: 节点索引
        size: 节点产生的内存大小
        pred_buffers: 前驱 Buffer 集合
        pred_nodes: 前驱节点集合
        succ_nodes: 后继节点集合
    """
    index: int = 0
    size: int = 0
    pred_buffers: OrderedSet[Union[SchedulerBuffer, FreeableInputBuffer]] = (
        dataclasses.field(default_factory=OrderedSet)
    )
    pred_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )


@dataclasses.dataclass
class FreeableInputBuffer:
    """
    可释放的输入 Buffer
    
    某些输入 Tensor 在使用后可以提前释放，降低峰值内存
    """
    name: str
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def get_name(self) -> str:
        return self.name
```

### 1.3 拓扑排序方法

```python
def topological_sort_lpmf(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_buf: dict[str, SchedulerBuffer],
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:
    """
    LPMF: Largest Peak Memory First
    
    优先处理峰值内存贡献最大的节点
    """
    # 1. 计算每个节点的峰值内存贡献
    # 2. 使用优先队列按峰值内存降序处理
    # 3. 生成拓扑顺序


def topological_sort_bfs(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    BFS: Breadth-First Search
    
    按 BFS 顺序处理节点
    """


def topological_sort_dfs(
    nodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    DFS: Depth-First Search
    
    按 DFS 顺序处理节点
    """
```

---

## 2. Combo Kernel 优化

### 2.1 Combo Kernel 决策

**文件**: `torch/_inductor/scheduler.py`

**文件**: `torch/_inductor/scheduler.py:6892-6970`

```python
def speedup_by_combo_kernel(self, nodes: list[BaseSchedulerNode]) -> bool:
    """
    If config.benchmark_fusion is False, always return True.
    Otherwise, return True if fusion can brings speedup.
    
    判断 Combo Kernel 是否能带来性能提升
    
    Args:
        nodes: 待融合的节点列表
    
    Returns:
        True 表示融合能带来加速
    """
    subkernel_nodes = nodes
    device = subkernel_nodes[0].get_device()

    # L6901-6903: 验证所有节点在同一设备上
    assert all(node.get_device() == device for node in subkernel_nodes), (
        "All nodes in a combo kernel group must be on the same device"
    )

    # L6905-6906: 如果不启用 benchmark，直接返回 True
    if not config.benchmark_combo_kernel:
        return True

    from triton.compiler.errors import CompilationError

    ms1, path1_list = 0.0, []
    node_benchmark_results = {}
    
    # L6912-6940: 分别 benchmark 每个子 Kernel
    for i, snode in enumerate(subkernel_nodes):
        node_list = snode.get_nodes()
        
        # L6914-6919: atomic_add 的 benchmark 可能不准确
        if self._any_atomic_add(node_list):
            fusion_log.debug(
                "ComboKernel: benchmarking may not accurate due to atomic_add"
            )

        try:
            ms, path = self.benchmark_fused_nodes(node_list)
            node_benchmark_results[snode] = (ms, path)
            
            # L6924-6929: 寄存器溢出检测
            if math.isinf(ms):
                fusion_log.debug(
                    "ComboKernel benchmark: register spilling of %d-th subkernel",
                    i,
                )
                return False
        except CompilationError as e:
            # L6930-6938: 处理 Loop-carried variable 错误
            if "Loop-carried variable" in str(e):
                fusion_log.debug(
                    "ComboKernel benchmark: return True because of loop-carried variable"
                )
                return True  # allow fusion
            else:
                raise
        ms1 += ms
        path1_list.append(path)

    # L6942-6954: Benchmark 融合后的 Kernel
    try:
        ms2, ms2_clone, _path2_list = self.benchmark_combo_kernel(
            subkernel_nodes, node_benchmark_results
        )
    except CompilationError as e:
        if "Loop-carried variable" in str(e):
            fusion_log.debug(
                "ComboKernel benchmark: return True because of loop-carried variable"
            )
            return True
        else:
            raise

    # L6957-6968: 判断是否加速
    # small_kernel: 小 Kernel benchmark 误差大，倾向于融合
    small_kernel = ms2 - ms2_clone < 0.3 or ms1 < 0.3
    
    if fusion_log.isEnabledFor(logging.DEBUG):
        if ms1 > ms2 or small_kernel:
            fusion_log.debug(
                "can fuse (benchmark): fusing causes %sx speedup",
                green_text(f"{ms1 / ms2:.3f}"),
            )
        else:
            fusion_log.debug(
                "cannot fuse (benchmark): fusing causes %sx slowdown",
                red_text(f"{ms1 / ms2:.3f}"),
            )
    
    # L6969-6970: ms1 已扣除 clone 时间
    return ms2 - ms2_clone < ms1 or small_kernel
```

**关键点**:
- 第 6905-6906 行：不启用 benchmark 时默认允许融合
- 第 6924-6929 行：寄存器溢出（inf）时拒绝融合
- 第 6930-6938 行：Loop-carried variable 错误时允许融合（Triton 限制）
- 第 6957 行：小 Kernel（<0.3ms）倾向于融合（benchmark 误差大）
- 第 6969-6970 行：比较融合前后的执行时间

### 2.2 外层循环融合

**文件**: `torch/_inductor/metrics.py:43-50`

```python
@dataclasses.dataclass
class CppOuterLoopFusedCount:
    """
    C++ 外层循环融合计数
    
    Attributes:
        inner_kernel_number: 内部 Kernel 数量
        local_buffer_number: 本地 Buffer 数量
    """
    inner_kernel_number: int
    local_buffer_number: int = 0


# L49-50: 记录外层循环融合
cpp_outer_loop_fused_inner_counts: list[CppOuterLoopFusedCount] = []
```

---

## 3. 通信优化技术

### 3.1 计算/通信重叠

**文件**: `torch/_inductor/comms.py`

**文件**: `torch/_inductor/comms.py:2219-2232`

```python
def reorder_compute_and_comm_for_overlap(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    重排序节点以实现计算和通信的重叠
    
    通过调整节点顺序，使得通信操作可以与计算操作并行执行，
    从而隐藏通信延迟
    
    Args:
        snodes: 调度节点列表
    
    Returns:
        重排序后的节点列表
    """
    order = snodes
    
    # L2224-2230: 应用配置的重排序 Passes
    for p in config.reorder_for_compute_comm_overlap_passes:
        if isinstance(p, str) and p in globals():
            p = globals()[p]  # it is a builtin pass
        assert callable(p), (
            f"Invalid reorder_compute_and_comm_for_overlap pass: {p} is not callable"
        )
        order = p(order)  # type: ignore[operator]
    
    return order
```

### 3.2 FSDP2 通信优化

**文件**: `torch/_inductor/comms.py:2235-2317`

```python
def remove_fsdp2_unsharded_param_graph_input_usage(graph: torch.fx.Graph):
    """
    This FX graph pass replaces uses of FSDP2 unsharded params with their corresponding
    graph intermediates that were fsdp.copy_ into the unsharded params.
    
    FSDP2 通信优化 Pass:
    - 移除 FSDP2 unsharded 参数的冗余 resize 和 copy 操作
    - 仅适用于具有特定模式的参数：resize_(full) -> copy_ -> resize_(0)
    
    NOTE: 只能应用于具有特定模式的 FSDP2 参数
    """
    node_list = list(graph.nodes)

    # L2252-2268: 查找所有图输入及其 resize 计数
    graph_input_to_resized_to_full_node_idxes = defaultdict(list)
    graph_input_to_resized_to_0_node_idxes = defaultdict(list)
    
    for idx, node in enumerate(node_list):
        if (
            node.op == "call_function"
            and node.target is torch.ops.inductor.resize_storage_bytes_.default
        ):
            assert node.args[0].op == "placeholder"
            graph_input = node.args[0]
            new_size = node.args[1]
            if new_size > 0:
                graph_input_to_resized_to_full_node_idxes[graph_input].append(idx)
            else:
                graph_input_to_resized_to_0_node_idxes[graph_input].append(idx)

    # L2269-2303: 检查 resize 模式
    def check_resize_pattern(graph_input):
        resized_to_full_idxes = graph_input_to_resized_to_full_node_idxes.get(
            graph_input, []
        )
        resized_to_0_idxes = graph_input_to_resized_to_0_node_idxes.get(graph_input, [])

        # L2280-2288: 检查数量相等
        if len(resized_to_full_idxes) != len(resized_to_0_idxes):
            log.warning(
                f"Unequal number of resize-to-full and resize-to-0 nodes for {graph_input}"
            )
            return False

        # L2290-2302: 检查顺序：(resize_to_full -> resize_to_0)+
        for resize_to_full_idx, resize_to_0_idx in zip(
            resized_to_full_idxes, resized_to_0_idxes
        ):
            if resize_to_full_idx >= resize_to_0_idx:
                log.warning(
                    f"resize-to-full at {resize_to_full_idx} happens after "
                    f"resize-to-0 at {resize_to_0_idx}"
                )
                return False
        return True

    # L2306-2316: 查找符合条件的 unsharded 参数
    unsharded_param_to_fsdp_copy_node_idxes = defaultdict(list)
    for idx, node in enumerate(node_list):
        if node.op == "call_function" and node.target is torch.ops.fsdp.copy_.default:
            fsdp_copy_node = node
            unsharded_param = node.args[0]
            assert unsharded_param.op == "placeholder"
            
            if check_resize_pattern(unsharded_param):
                unsharded_param_to_fsdp_copy_node_idxes[unsharded_param].append(idx)
```

---

## 4. 指标收集系统

### 4.1 核心指标

**文件**: `torch/_inductor/metrics.py`

**文件**: `torch/_inductor/metrics.py:24-92`

```python
# L24-25: Kernel 生成计数
generated_kernel_count = 0
generated_cpp_vec_kernel_count = 0

# L27: 内存访问字节数
num_bytes_accessed = 0

# L28-33: 节点元素数统计
nodes_num_elem: list[
    tuple[
        BaseSchedulerNode,
        int,
    ]
] = []

# L34: 节点运行时间
node_runtimes: list[tuple[BaseSchedulerNode, float]] = []

# L37-38: 融合前后节点计数
ir_nodes_pre_fusion = 0

# L40: C++ 类型转换计数
cpp_to_dtype_count = 0

# L43-46: C++ 外层循环融合计数
@dataclasses.dataclass
class CppOuterLoopFusedCount:
    inner_kernel_number: int
    local_buffer_number: int = 0

cpp_outer_loop_fused_inner_counts: list[CppOuterLoopFusedCount] = []

# L52-56: 其他优化计数
num_comprehensive_padding = 0
num_matches_for_scatter_upon_const_tensor = 0
num_loop_reordering = 0
num_auto_chunking: int = 0

# L59: 并行归约计数
parallel_reduction_count = 0

# L61: 混合顺序归约计数
codegen_mix_order_reduction = 0


# L65-92: 重置所有计数器
def reset() -> None:
    global generated_kernel_count
    global generated_cpp_vec_kernel_count
    global num_bytes_accessed, nodes_num_elem
    global ir_nodes_pre_fusion
    global cpp_to_dtype_count
    global cpp_outer_loop_fused_inner_counts
    global num_comprehensive_padding
    global num_matches_for_scatter_upon_const_tensor
    global num_loop_reordering
    global parallel_reduction_count
    global codegen_mix_order_reduction
    global num_auto_chunking

    generated_kernel_count = 0
    generated_cpp_vec_kernel_count = 0
    num_bytes_accessed = 0
    nodes_num_elem.clear()
    node_runtimes.clear()
    ir_nodes_pre_fusion = 0
    cpp_to_dtype_count = 0
    cpp_outer_loop_fused_inner_counts.clear()
    num_comprehensive_padding = 0
    num_matches_for_scatter_upon_const_tensor = 0
    num_loop_reordering = 0
    parallel_reduction_count = 0
    codegen_mix_order_reduction = 0
    num_auto_chunking = 0
```

### 4.2 性能指标表

```python
def get_metric_table() -> dict[str, Any]:
    """
    获取性能指标表
    
    Returns:
        包含所有性能指标的字典
    """
    return {
        "generated_kernel_count": generated_kernel_count,
        "generated_cpp_vec_kernel_count": generated_cpp_vec_kernel_count,
        "num_bytes_accessed": num_bytes_accessed,
        "ir_nodes_pre_fusion": ir_nodes_pre_fusion,
        "cpp_to_dtype_count": cpp_to_dtype_count,
        "parallel_reduction_count": parallel_reduction_count,
        "codegen_mix_order_reduction": codegen_mix_order_reduction,
        # ...
    }
```

---

## 5. Benchmark 系统

### 5.1 Benchmarker 类

**文件**: `torch/_inductor/runtime/benchmarking.py`

**文件**: `torch/_inductor/runtime/benchmarking.py:100+`

```python
class Benchmarker:
    """
    性能测试基准类
    
    提供 GPU/CPU Kernel 的精确计时功能
    """
    
    @may_distort_benchmarking_result
    @time_and_count
    def benchmark_gpu(self, fn, *args, **kwargs):
        """
        GPU Kernel Benchmark
        
        Args:
            fn: 待测试的函数
            args: 位置参数
            kwargs: 关键字参数
        
        Returns:
            执行时间（毫秒）
        """
        # 1. 预热执行（warmup）
        # 2. 多次执行取中位数
        # 3. 返回执行时间
    
    @may_distort_benchmarking_result
    @time_and_count
    def benchmark_cpu(self, fn, *args, **kwargs):
        """
        CPU Kernel Benchmark
        """
```

### 5.2 测试失真功能

**文件**: `torch/_inductor/runtime/benchmarking.py:30-61`

```python
def may_distort_benchmarking_result(
    fn: Callable[Concatenate[Any, P], T]
) -> Callable[Concatenate[Any, P], T]:
    """
    根据配置扭曲 benchmark 结果（用于测试）
    
    支持的失真方法:
    - inverse: 返回倒数
    - random: 返回随机值
    """
    from torch._inductor import config

    if config.test_configs.distort_benchmarking_result == "":
        return fn

    def distort(
        ms: list[float] | tuple[float, ...] | float,
    ) -> list[float] | tuple[float, ...] | float:
        if isinstance(ms, (list, tuple)):
            return type(ms)(distort(val) for val in ms)

        distort_method = config.test_configs.distort_benchmarking_result
        assert isinstance(ms, float)
        
        if distort_method == "inverse":
            return 1.0 / ms if ms else 0.0
        elif distort_method == "random":
            import random
            return random.random()
        else:
            raise RuntimeError(f"Unrecognized distort method {distort_method}")

    @functools.wraps(fn)
    def wrapper(
        *args: list[Any], **kwargs: dict[str, Any]
    ) -> list[float] | tuple[float, ...] | float:
        ms = fn(*args, **kwargs)
        return distort(ms)

    return wrapper
```

---

## 6. 配置项

### 6.1 内存优化配置

```python
import torch._inductor.config as config

# 启用峰值内存重排序
config.reorder_for_peak_memory = True

# 启用调试模式（导出图用于模拟器）
config.reorder_for_peak_memory_debug = True

# 自定义重排序方法
config.reorder_for_peak_memory_methods = [
    "topological_sort_lpmf",
    "topological_sort_bfs",
]
```

### 6.2 通信优化配置

```python
# 启用计算/通信重叠
config.reorder_for_compute_comm_overlap = True

# 自定义重叠 Passes
config.reorder_for_compute_comm_overlap_passes = [
    "sink_comm_pass",
    "overlap_pass",
]

# 融合 DDP 通信
config._fuse_ddp_communication = True
```

### 6.3 Combo Kernel 配置

```python
# 启用 Combo Kernels
config.combo_kernels = True

# 启用 Combo Kernel Autotune
config.combo_kernels_autotune = 1

# 启用 Benchmark（默认禁用）
config.benchmark_combo_kernel = True
```

### 6.4 测试配置

```python
# 扭曲 benchmark 结果（测试用）
config.test_configs.distort_benchmarking_result = "inverse"  # 或 "random"

# 禁用某些优化（调试用）
config.reorder_for_peak_memory = False
config.combo_kernels = False
```

---

## 7. 源码阅读指南

### 7.1 核心文件索引

| 文件 | 行号范围 | 内容 |
|------|----------|------|
| `memory.py` | L913-999 | `reorder_for_peak_memory` |
| `memory.py` | L40-87 | 内存规划数据类 |
| `scheduler.py` | L6892-6970 | `speedup_by_combo_kernel` |
| `comms.py` | L2219-2232 | `reorder_compute_and_comm_for_overlap` |
| `comms.py` | L2235-2317 | `remove_fsdp2_unsharded_param_graph_input_usage` |
| `metrics.py` | L24-92 | 指标计数器定义 |
| `benchmarking.py` | L30-61 | 测试失真功能 |
| `benchmarking.py` | L100+ | `Benchmarker` 类 |

### 7.2 推荐阅读顺序

```
1. torch/_inductor/memory.py (内存优化)
2. torch/_inductor/scheduler.py (Combo Kernel)
3. torch/_inductor/comms.py (通信优化)
4. torch/_inductor/metrics.py (指标收集)
5. torch/_inductor/runtime/benchmarking.py (Benchmark)
```

---

## 8. 总结

本章详细介绍了 PyTorch Inductor 的核心性能优化技术：

1. **内存优化**: 
   - 峰值内存重排序（LPMF、BFS、DFS）
   - Buffer 生命周期管理
   - 可释放输入 Buffer 追踪

2. **Combo Kernel**:
   - 自动融合决策
   - Benchmark 驱动的融合判断
   - 小 Kernel 特殊处理

3. **通信优化**:
   - 计算/通信重叠
   - FSDP2 通信优化
   - DDP 融合

4. **指标收集**:
   - Kernel 生成计数
   - 内存访问统计
   - 融合效果追踪

5. **Benchmark 系统**:
   - GPU/CPU 精确计时
   - 测试失真功能
   - 确定性模式支持

这些优化技术共同作用，使得 Inductor 能够生成高度优化的执行代码，显著提升深度学习模型的训练和推理性能。

---

**下一篇**: [PyTorch Inductor 源码解析（九）：max-autotune 深度解析](./09-max-autotune.md)
