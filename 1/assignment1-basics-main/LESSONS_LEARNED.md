# CS336 Assignment 1 踩坑记录与经验总结

## 概览

本项目在实现大语言模型基础组件的过程中遇到了多个问题，涉及编码兼容性、算法实现细节、框架差异等方面。

---

## 1. Windows 平台兼容性问题

### 问题描述
在 Windows 上运行测试时出现 `ModuleNotFoundError: No module named 'resource'` 和 `UnicodeDecodeError`。

### 问题成因
- `resource` 模块是 Unix/Linux 特有的系统调用库，Windows 上没有
- Python 在 Windows 上默认使用 GBK 编码打开文件，而不是 UTF-8
- 项目原本在 Linux/Mac 上开发，没有考虑 Windows 兼容性

### 解决方式
```python
# 1. 添加兼容性检查
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

# 2. 在 memory_limit 装饰器中跳过 Windows
if not HAS_RESOURCE:
    return f(*args, **kwargs)  # 直接执行，不限制内存

# 3. 所有文件操作显式指定 encoding='utf-8'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
```

### 探索路径
1. 最初以为是依赖缺失，尝试安装 resource 包
2. 发现是系统 API 差异，需要条件导入
3. 批量替换所有文件操作，添加 encoding 参数

### 未来启发
- **跨平台开发必须显式指定文件编码**：`encoding='utf-8'`
- **系统特定功能要加兼容性检查**：使用 `try/except` 或 `sys.platform` 判断
- **CI/CD 应该包含多平台测试**：Windows、Linux、Mac 都要测

---

## 2. 学习率调度算法细节

### 问题描述
Cosine LR Schedule 测试失败，warmup 阶段的输出值与期望不符。

### 问题成因
期望行为：
- it=0 → 0
- it=1 → 0.1428...
- it=7 (warmup_iters) → 1.0

我的实现：
- it=0 → 0.1428... (使用了 `(it + 1) / warmup_iters`)
- it=7 → 1.0

### 解决方式
```python
# 错误的实现（从 it=1 开始 warmup）
return max_learning_rate * (it + 1) / warmup_iters

# 正确的实现（从 it=0 开始 warmup）
return max_learning_rate * it / warmup_iters
```

### 探索路径
1. 对比测试输出的 ACTUAL vs DESIRED 数组
2. 发现所有 warmup 值都偏移了一个位置
3. 意识到 warmup 应该从第 0 个 iteration 开始，而不是第 1 个

### 未来启发
- **仔细阅读测试用例的期望值**：对比每个 index 的具体值
- **明确算法的边界条件**：it 从 0 开始还是从 1 开始
- **warmup 阶段包含 it=0**：it=0 时 LR 应该为 0

---

## 3. Transformer 权重加载 - Weight Tying

### 问题描述
`test_transformer_lm` 测试失败，输出值与期望值差异巨大。

### 问题成因
- 参考实现（snapshot）中的 `token_embeddings` 和 `lm_head` **没有 weight tying**
- 我的实现默认启用了 weight tying（共享权重）
- 这导致 `lm_head.weight` 被覆盖为 `token_embeddings.weight`，结果完全不同

```python
# 错误的实现（启用了 weight tying）
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_embeddings.weight  # 共享权重

# 加载权重时
model.lm_head.weight.data = weights['lm_head.weight']  # 这行实际上覆盖了 shared weight
```

### 解决方式
```python
# 移除 weight tying
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
# 不要写 self.lm_head.weight = self.token_embeddings.weight

# 加载权重时分别加载
model.token_embeddings.weight.data = weights['token_embeddings.weight']
model.lm_head.weight.data = weights['lm_head.weight']  # 独立加载
```

### 探索路径
1. 检查 state_dict 发现 `token_embeddings.weight` 和 `lm_head.weight` 是不同的张量
2. 意识到参考实现没有使用 weight tying
3. 移除代码中的 weight tying 逻辑
4. 确保 eval 模式和无梯度计算

### 未来启发
- **不要假设模型架构**：实际检查 checkpoint 的 state_dict
- **weight tying 是可选特性**：GPT-2 早期版本没有 weight tying
- **测试 snapshot 是黄金标准**：如果 snapshot 有不同权重，就不要 tie

---

## 4. Tokenizer 字节编码映射

### 问题描述
测试 `get_tokenizer_from_vocab_merges_path` 出现 `KeyError: '臓'`（奇怪的汉字）。

### 问题成因
- GPT-2 使用字节到 Unicode 的映射表将 0-255 字节映射到可打印字符
- 在 Windows 上，Python 默认使用 GBK 编码打开文件
- merges.txt 中的特殊字符（如 `Ġ` U+0120）被错误解码为中文字符

```python
# 错误的实现（Windows 默认 GBK）
with open(merges_path) as f:  # encoding 默认为 locale.getpreferredencoding()
    # 文件内容被错误解码
```

### 解决方式
```python
# 显式指定 UTF-8 编码
with open(merges_path, encoding='utf-8') as f:
    merges = [line.split() for line in f]
```

### 探索路径
1. 发现 `KeyError` 的字符是中文，怀疑编码问题
2. 检查文件内容发现是 UTF-8 编码的 Unicode 字符
3. 在 Windows PowerShell 中测试发现默认编码是 GBK
4. 批量添加 `encoding='utf-8'` 参数

### 未来启发
- **永远显式指定文件编码**：不要依赖系统默认值
- **Python 3 的 open() 默认编码是平台相关的**
- **Unicode 字符要小心**：特别是 0x80-0xFF 范围的字符

---

## 5. BPE Tokenizer 实现细节

### 问题描述
1. Tokenizer 把 `\n\n` 合并为一个 token，但 tiktoken 保留为两个
2. BPE 训练的 merges 顺序与参考实现不同

### 问题成因
**问题 1：连续换行符**
- 我的实现正确地合并了 `\n + \n` → `\n\n`（merge rank 372）
- 但 tiktoken 似乎有特殊的预处理逻辑，阻止了这种合并
- 可能 tiktoken 在处理连续的空白字符时有特殊规则

**问题 2：Merge 顺序**
- 当两个 pair 出现频率相同时，平局打破策略不同
- 参考实现可能使用字典序、或按 pair 在文本中首次出现的位置
- 我的实现使用 `(count, pair)` 作为排序 key

### 探索路径
1. 发现 `\n\n` 的 merge rank 是 372，相对较低
2. 在 tiktoken 中测试发现 `\n\n` 确实被编码为两个 198
3. 对比 merges 文件发现我的实现与参考在频率相同时选择不同

### 未来启发
- **Tokenizer 行为可能有细微差异**：只要 roundtrip 正确即可接受
- **BPE 的平局打破策略未标准化**：不同实现可能有不同结果
- **性能优先**：确保训练算法是 O(n log n) 而不是 O(n²)

---

## 6. BPE 训练算法优化 - 从 2.3秒 到 0.8秒

### 问题描述
`test_train_bpe_speed` 测试要求 BPE 训练在 1.5 秒内完成，初始实现耗时 2.3 秒。

### 问题成因：时间复杂度的陷阱

**初始实现（O(V × N × M)）**：
```python
words = list(word_freqs.items())  # N words
for _ in range(target_merges):      # V iterations (V = vocab_size)
    pair_counts = {}
    for word, freq in words:        # O(N)
        for i in range(len(word)):  # O(M)
            pair_counts[(word[i], word[i+1])] += freq
    best_pair = max(pair_counts, key=...)  # O(vocab)
    # Update all words...
```
每次 merge 都重新扫描所有词的所有 token，导致 V × N × M 的时间复杂度。

### 解决方式：四位一体数据结构

建立四个数据结构，在循环中只更新它们：

```python
# 1. words: 去重后的单词列表
words = [list(word) for word in word_freqs.keys()]

# 2. word_counts: 每个单词的出现次数
word_counts = list(word_freqs.values())

# 3. pair_counts: pair -> total frequency
pair_counts = {(token[i], token[i+1]): count}

# 4. pair_to_words (反向索引): pair -> set of word indices
pair_to_words = {(token[i], token[i+1]): {word_idx1, word_idx2}}
```

**增量更新逻辑（O(K) 其中 K = 受影响词的数量）**：
```python
for _ in range(target_merges):
    # 1. 找最大值：O(vocab)
    best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
    
    # 2. 精准定位：O(1) 通过反向索引
    affected_indices = pair_to_words[best_pair]  # 通常只有几十几百个
    
    # 3. 局部重构与增量结算：O(K × avg_word_length)
    for word_idx in affected_indices:
        old_word = words[word_idx]
        freq = word_counts[word_idx]
        
        # 扣除旧账：O(word_length)
        for i in range(len(old_word) - 1):
            pair = (old_word[i], old_word[i+1])
            pair_counts[pair] -= freq
            pair_to_words[pair].discard(word_idx)
        
        # 合并列表：O(word_length)
        new_word = []
        i = 0
        while i < len(old_word):
            if i < len(old_word) - 1 and old_word[i] == bp0 and old_word[i+1] == bp1:
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(old_word[i])
                i += 1
        
        # 记录新账：O(word_length)
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_idx)
        
        words[word_idx] = new_word
    
    # 4. 清理门户
    del pair_counts[best_pair]
    del pair_to_words[best_pair]
```

### 探索路径

1. **发现问题**：测试要求 < 1.5秒，实际耗时 2.3秒
2. **分析复杂度**：识别出每次迭代都全量扫描的问题
3. **尝试微优化**：使用局部变量、list 代替 tuple，效果不明显（1.56秒）
4. **深入理解算法**：意识到需要像数据库索引一样的反向索引结构
5. **建立四位一体**：实现完整的增量更新，时间降到 0.8秒

### 关键错误与纠正

**错误 1：尝试用 heapq 优化**
```python
# 错误：使用 heapq 导致正确性问题
heap = [(-count, pair) for pair, count in pair_counts.items()]
# 问题：当 pair count 更新时，heap 中的旧值失效
```
纠正：使用普通 dict + max()，在 Python 中足够快。

**错误 2：过度优化破坏正确性**
```python
# 错误：一次性找到所有 merge 位置再重建
positions = [i for i in range(len(word)) if is_match(i)]
new_word = rebuild_with_positions(word, positions)
```
纠正：使用 greedy left-to-right 扫描，与 BPE 标准一致。

### 未来启发

- **算法复杂度分析是核心**：O(V × N × M) vs O(V × K × M)，K << N
- **空间换时间是有效策略**：用 pair_to_words 索引换取 O(1) 查找
- **避免过早优化**：先保证正确性，再识别瓶颈，最后针对性优化
- **增量更新是高性能的关键**：只更新变化的部分，而不是全量重建
- **数据结构的选择至关重要**：
  - `dict` 用于 O(1) 查找
  - `set` 用于 O(1) 成员检查
  - `list` 用于有序遍历

### 性能对比

| 实现方式 | 时间复杂度 | 实际耗时 | 是否通过 |
|---------|-----------|---------|---------|
| 全量重建 | O(V × N × M) | 2.3秒 | ❌ |
| 微优化 | O(V × N × M) | 1.56秒 | ❌ |
| 增量更新 | O(V × K × M) | 0.8秒 | ✅ |

---

## 7. PyTorch Checkpoint 跨设备加载

### 问题描述
在 CPU 机器上加载 GPU 训练的 checkpoint 时出现 `RuntimeError: Attempting to deserialize object on a CUDA device`。

### 解决方式
```python
# 始终使用 map_location 确保跨设备兼容
checkpoint = torch.load(path, map_location='cpu')
# 或
checkpoint = torch.load(path, map_location=device)
```

### 未来启发
- **保存 checkpoint 时记录设备信息**
- **加载时显式指定 map_location**
- **不要假设运行环境有 GPU**

---

## 8. 测试驱动开发 (TDD) 经验

### 有效的工作流程
1. **先运行测试，观察失败**：了解期望行为
2. **最小化实现**：先让测试通过，再优化
3. **对比实际输出和期望输出**：使用 pytest 的 `-v` 和 `--tb=long` 选项
4. **使用调试脚本**：对于复杂问题，编写独立的 debug_xxx.py

### 调试技巧
```bash
# 运行单个测试
uv run pytest tests/test_model.py::test_transformer_lm -v --tb=long

# 检查 state_dict 内容
uv run python -c "import torch; d = torch.load('model.pt', map_location='cpu'); print(d.keys())"
```

---

## 总结与建议

### 编码规范
1. **所有文件操作必须指定 `encoding='utf-8'`**
2. **跨平台功能要加兼容性检查**
3. **复杂算法要先理解期望输出再实现**

### 调试方法论
1. **阅读错误信息**：KeyError 中的字符提示编码问题
2. **对比输入输出**：ACTUAL vs DESIRED
3. **检查边界条件**：索引从 0 还是从 1 开始
4. **验证假设**：weight tying 是否真的存在

### 性能优化
1. **先实现正确，再优化性能**
2. **使用适当的数据结构**：defaultdict, set, heapq
3. **增量更新而不是全量重建**

### 深度学习工程
1. **理解模型架构细节**：weight tying, dropout, layer norm 位置
2. **正确处理设备**：CPU/GPU 兼容性
3. **Tokenizer 是黑盒**：行为可能有细微差异，保证 roundtrip 正确即可
