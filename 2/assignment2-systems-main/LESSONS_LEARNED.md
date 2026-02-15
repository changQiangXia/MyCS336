# 作业2 - Systems 经验总结

## 作业2完成情况

| 模块 | 测试情况 | 状态 |
|------|---------|------|
| FlashAttention2 PyTorch | 2/2 passed | ✅ 完成 |
| FlashAttention2 Triton | 2/2 passed | ✅ 完成 |
| DDP Individual | 2/2 passed | ✅ 完成 |
| DDP Bucketed | 6/6 passed | ✅ 完成 |
| Sharded Optimizer | 2/2 failed | ❌ Windows限制 |

**总计**: 14/16 passed (87.5%)

---

## 1. FlashAttention2 算法实现

### 核心算法：Online Softmax + Tiling

FlashAttention2 的关键创新是使用 tiling 和 online softmax 来避免存储完整的 N×N attention 矩阵。

#### 错误实现
```python
# 错误：在循环内归一化
for j in range(Tc):
    Sij = Qi @ Kj^T * scale
    # ...
    oi = oi / li  # ❌ 错误！不应该在这里归一化
```

#### 正确实现
```python
# 正确：循环结束后才归一化
mi = -inf  # running max
li = 0     # running sum
oi = 0     # running numerator (未归一化)

for j in range(Tc):
    Sij = Qi @ Kj^T * scale
    
    # Online softmax 更新
    m_new = max(mi, max(Sij))
    li = exp(mi - m_new) * li + sum(exp(Sij - m_new))
    oi = exp(mi - m_new) * oi + exp(Sij - m_new) @ Vj
    mi = m_new

output = oi / li  # ✅ 最后才归一化
```

### 关键细节

1. **数值稳定性**：中间计算使用 fp32
2. **Causal Mask**：`col <= row` 表示可以 attend
3. **Backward 重计算**：重新计算 attention 分数而不是存储

---

## 2. Windows PyTorch 分布式训练问题

### 问题 1：libuv 不支持

**错误**：`RuntimeError: use_libuv was requested but PyTorch was build without libuv support`

**解决**：
```python
os.environ["USE_LIBUV"] = "0"
```

### 问题 2：ReduceOp.AVG 不支持

**错误**：`RuntimeError: Cannot use ReduceOp.AVG with Gloo`

**原因**：Gloo 后端（CPU 分布式）不支持 `ReduceOp.AVG`，只支持 `ReduceOp.SUM`

**解决**：
```python
# 错误的：
dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

# 正确的：
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
tensor.div_(world_size)
```

### 问题 3：堆损坏 (Sharded Optimizer)

**错误**：`process terminated with exit code 3221226505` (STATUS_HEAP_CORRUPTION)

**原因**：Windows PyTorch 多进程分布式存在稳定性问题

**状态**：未解决，确定是 PyTorch Windows 版本底层问题

### 环境变量设置

在 `tests/conftest.py` 顶部添加：
```python
import os
os.environ["USE_LIBUV"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
```

---

## 3. PyTorch CUDA 版本安装与 uv 冲突

### 问题描述

安装 CUDA 版本 PyTorch 后，运行 `uv run python` 时发现 CUDA 仍然不可用：`torch.cuda.is_available()` 返回 `False`。

### 问题成因

**uv 依赖管理冲突**：
1. `pyproject.toml` 中指定了 `torch~=2.6.0`（CPU 版本）
2. 手动安装 `torch==2.6.0+cu118`（CUDA 版本）成功
3. 运行 `uv run` 时，uv 检测到 pyproject.toml 的依赖与实际安装不符
4. uv **自动重新安装** CPU 版本，覆盖 CUDA 版本

```bash
# 现象：安装 CUDA 版本成功
$ uv pip install torch==2.6.0+cu118
+ torch==2.6.0+cu118

# 但运行时被覆盖
$ uv run python -c "import torch; print(torch.cuda.is_available())"
# uv 自动重装...
CUDA available: False  # 被覆盖回 CPU 版本
```

### 解决方式

**需要修改两个 pyproject.toml 文件**：

1. **主项目** `pyproject.toml`：
```toml
# 原配置
dependencies = [
    "torch~=2.6.0; sys_platform != 'darwin' or platform_machine != 'x86_64'",
]

# 修改为：注释掉 torch 依赖
dependencies = [
    # "torch~=2.6.0; sys_platform != 'darwin' or platform_machine != 'x86_64'",
]
```

2. **cs336-basics** `cs336-basics/pyproject.toml`：
```toml
# 同样注释掉 torch 依赖
dependencies = [
    # "torch~=2.6.0; sys_platform != 'darwin' or platform_machine != 'x86_64'",
]
```

3. **重新安装 CUDA 版本**：
```bash
uv pip install torch==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 探索路径

1. **初次安装 CUDA 版本成功**：显示 `+ torch==2.6.0+cu118`
2. **验证时失败**：`torch.cuda.is_available()` 仍为 `False`
3. **观察 uv 行为**：发现 `uv run` 自动重新安装依赖
4. **检查 pyproject.toml**：发现两处都声明了 torch 依赖
5. **注释依赖并重新安装**：问题解决

### 未来启发

- **uv 会强制同步 pyproject.toml 与实际环境**：如果两者不符，会自动重新安装
- **手动安装 PyTorch 版本前，先注释 pyproject.toml 中的依赖**
- **检查所有子模块的 pyproject.toml**：cs336-basics 也有 torch 依赖
- **验证时使用 `uv run`**：确保 uv 不会自动覆盖已安装的包

---

## 4. DDP 实现要点

### Individual Parameters 版本

每个参数单独进行 all-reduce：
```python
for param in self.module.parameters():
    if param.requires_grad and param.grad is not None:
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(self.world_size)
```

### Bucketed 版本

将参数分组到 buckets 中，减少通信次数：
```python
# 1. 构建 buckets（按参数大小分组）
self.buckets = self._build_buckets()

# 2. 将同一 bucket 的梯度 flatten 后 all-reduce
for bucket in self.buckets:
    grads = [param.grad for name, param in bucket if param.grad is not None]
    flat_grads = _flatten_dense_tensors(grads)
    
    dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
    flat_grads.div_(self.world_size)
    
    # unflatten 并复制回原始梯度
    unflattened = _unflatten_dense_tensors(flat_grads, grads)
    for grad, unflat_grad in zip(grads, unflattened):
        grad.copy_(unflat_grad)
```

### 参数广播

初始化时从 rank 0 广播参数到所有 ranks：
```python
for param in self.module.parameters():
    dist.broadcast(param.data, src=0)
```

---

## 5. Sharded Optimizer (ZeRO-1) 实现

### 核心思想

- 每个 rank 只存储部分参数的优化器状态
- 参数按 `i % world_size` 分配到不同 rank
- 梯度需要 all-reduce 聚合
- 更新后的参数需要 broadcast 到所有 ranks

### 实现代码
```python
class ShardedOptimizer:
    def __init__(self, params, optimizer_cls, world_size, rank, **kwargs):
        # 分配参数到不同 rank
        for i, param in enumerate(all_params):
            assigned_rank = i % world_size
            if assigned_rank == rank:
                local_params.append(param)
        
        # 只创建本地优化器
        self.optimizer = optimizer_cls(local_params, **kwargs)
    
    def step(self):
        # 1. 聚合本地参数的梯度
        for param in self.local_params:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)
        
        # 2. 更新本地参数
        self.optimizer.step()
        
        # 3. 广播更新后的参数到所有 ranks
        for param in self.all_params:
            owner = self.param_to_rank[id(param)]
            dist.broadcast(param.data, src=owner)
```

---

## 6. 测试命令

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行特定测试
uv run pytest tests/test_attention.py -v
uv run pytest tests/test_ddp.py -v
uv run pytest tests/test_sharded_optimizer.py -v

# 保存输出到文件
uv run pytest tests/ -v > test_output.txt 2>&1
```

---

## 7. 经验总结

### Windows 分布式训练建议

1. **优先在 Linux 上开发分布式代码**：Windows 支持是次要的
2. **使用 WSL2**：Windows Subsystem for Linux 可以解决大部分问题
3. **或者使用 Docker**：在 Linux 容器中运行
4. **Gloo vs NCCL**：
   - Gloo：CPU 分布式，功能有限，跨平台
   - NCCL：GPU 分布式，功能完整，仅 Linux

### uv 包管理注意事项

1. **修改 pyproject.toml 前先备份**
2. **手动安装 PyTorch 前注释掉依赖声明**
3. **检查所有子模块的 pyproject.toml**
4. **使用 `uv pip install` 而非 `pip install`**：确保安装在 uv 管理的虚拟环境中

### 调试技巧

1. **设置环境变量**：在 conftest.py 中设置，确保子进程继承
2. **使用 127.0.0.1**：比 localhost 更可靠
3. **检查错误码**：3221226505 表示堆损坏，通常是底层问题
4. **单进程测试**：先确保单进程逻辑正确，再测试多进程

### 实现要点

1. **理解算法后再写代码**：FlashAttention2 的 paper 要仔细阅读
2. **数值稳定性**：中间计算使用 fp32
3. **梯度同步**：SUM 后除以 world_size，不要依赖 AVG
4. **参数广播**：初始化时确保所有 ranks 有相同的初始权重
