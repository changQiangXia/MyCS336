# CS336 Assignment 4: Data Processing 经验教训

## 概览

本次作业实现了大语言模型数据预处理管道，包括 HTML 文本提取、语言识别、PII 遮蔽、内容分类、质量过滤和文档去重。

### 完成情况

| 模块 | 功能 | 实现方式 | 状态 |
|------|------|---------|------|
| Text Extraction | HTML 转纯文本 | resiliparse | ✅ 通过 |
| Language ID | 176 种语言识别 | fasttext | ✅ 通过 |
| PII Masking | 邮箱/电话/IP 遮蔽 | 正则表达式 | ✅ 通过 |
| Content Classification | NSFW/毒性检测 | 关键词规则 | ✅ 通过 |
| Quality Filter | Gopher 规则过滤 | 规则实现 | ✅ 通过 |
| Deduplication | 精确 + MinHash | LSH | ✅ 通过 |

**测试覆盖**: 21 个测试全部通过 ✅

---

## 1. Windows 上 fasttext 编译问题

### 问题描述
运行 `uv sync` 时 fasttext 编译失败：
```
error: Microsoft Visual C++ 14.0 or greater is required.
```

### 问题成因
fasttext 官方包 `fasttext>=0.9.3` 需要从源码编译 C++ 扩展，Windows 上需要 Visual Studio Build Tools。

### 解决方案
使用预编译的 wheel 包替代：

```toml
# pyproject.toml
[project]
dependencies = [
    # "fasttext>=0.9.3",  # ❌ 需要编译
    "fasttext-wheel>=0.9.2",  # ✅ 预编译版本
]
```

修改后需要重新生成锁文件：
```bash
rm uv.lock
uv lock
uv sync
```

### 经验
- **Windows 优先使用 wheel 包**：避免编译依赖
- **修改依赖后必须重新生成 uv.lock**：否则旧依赖仍会被使用
- **fasttext-wheel 是 drop-in 替代**：API 完全相同

---

## 2. resilientparse HTML 提取 API

### 问题描述
`extract_plain_text` 函数接收 `bytes` 时抛出异常：
```
TypeError: Parameter "html" is neither string nor HTMLTree.
```

### 问题成因
resiliparse 的 `extract_plain_text` 需要字符串输入，不是 bytes。

### 解决方案
显式解码 bytes 为字符串：

```python
from resiliparse.extract.html2text import extract_plain_text

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        # 先解码 bytes 为字符串
        html_str = html_bytes.decode('utf-8', errors='replace')
        text = extract_plain_text(html_str)
        return text
    except Exception:
        return None
```

### 经验
- **仔细阅读库文档**：确认参数类型
- **使用 `errors='replace'`**：处理损坏的编码
- **返回 None 而不是抛出异常**：符合测试期望

---

## 3. 精确行级去重的语义理解

### 问题描述
实现 `exact_line_deduplication` 时，对"去重"的理解有误，导致测试失败。

### 两种去重语义

**语义 A：保留首次出现，删除后续重复**
```
File1: [A, B, C]    → File1: [A, B, C]
File2: [A, D]       → File2: [D]  (A 被删除)
```

**语义 B：跨文件重复的行全部删除**
```
File1: [A, B, C]    → File1: [B, C]  (A 在 File2 中也出现)
File2: [A, D]       → File2: [D]     (A 被删除)
```

### 正确实现（语义 B）

```python
def exact_line_deduplication(input_files, output_directory):
    # 第一遍：统计每行出现在多少文件中
    line_counts = {}
    for filepath in input_files:
        seen_in_file = set()
        with open(filepath) as f:
            for line in f:
                stripped = line.rstrip('\n\r')
                if stripped not in seen_in_file:
                    seen_in_file.add(stripped)
                    line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    # 出现在多个文件中的行是"模板/页眉页脚"
    duplicate_lines = {line for line, count in line_counts.items() if count > 1}
    
    # 第二遍：写入文件，删除重复行
    for filepath in input_files:
        output_path = output_directory / Path(filepath).name
        with open(filepath) as infile, open(output_path, 'w') as outfile:
            for line in infile:
                stripped = line.rstrip('\n\r')
                if stripped not in duplicate_lines:
                    outfile.write(line)
```

### 经验
- **理解业务语义**：这里是去除跨文件的重复模板内容
- **常见于网页数据**：页眉、导航栏、版权信息会重复出现
- **与 MinHash 的区别**：精确匹配 vs 模糊匹配

---

## 4. MinHash + LSH 模糊去重实现

### 核心算法流程

```python
def minhash_deduplication(input_files, num_hashes, num_bands, ngrams, 
                          jaccard_threshold, output_directory):
    # 1. 为每个文档生成 shingles
    for doc in documents:
        shingles = set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
    
    # 2. 计算 MinHash 签名
    for shingle in shingles:
        for i in range(num_hashes):
            hash_val = mmh3.hash(shingle, seed=i)
            signature[i] = min(signature[i], hash_val)
    
    # 3. LSH：将签名分桶
    rows_per_band = num_hashes // num_bands
    for band_idx in range(num_bands):
        band = signature[band_idx*rows_per_band : (band_idx+1)*rows_per_band]
        band_id = hash(tuple(band))
        buckets[band_id].append(doc_idx)
    
    # 4. 只比较同一桶中的文档对
    for bucket in buckets:
        for doc_i, doc_j in pairs_in_bucket:
            jaccard = len(shingles_i & shingles_j) / len(shingles_i | shingles_j)
            if jaccard >= threshold:
                mark_as_duplicate(doc_j)
```

### 关键参数

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `num_hashes` | MinHash 签名长度 | 100-500 |
| `num_bands` | LSH 桶数量 | 10-50 |
| `ngrams` | shingle 大小 | 5 |
| `jaccard_threshold` | 相似度阈值 | 0.8 |

### 经验
- **LSH 显著减少比较次数**：从 O(N²) 降到 O(N × bucket_size)
- **参数需要调优**：更多 hashes = 更精确但更慢
- **使用 mmh3 库**：高性能 MurmurHash3 实现

---

## 5. Gopher 质量过滤规则

### Gopher 规则实现

```python
def gopher_quality_filter(text: str) -> bool:
    words = text.split()
    lines = text.split('\n')
    
    # 规则 1: 50-100,000 个非符号词
    non_symbol = [w for w in words if any(c.isalnum() for c in w)]
    if not (50 <= len(non_symbol) <= 100000):
        return False
    
    # 规则 2: 平均词长 3-10 字符
    alpha_words = [w for w in words if any(c.isalpha() for c in w)]
    mean_len = sum(len(w) for w in alpha_words) / len(alpha_words)
    if not (3 <= mean_len <= 10):
        return False
    
    # 规则 3: <30% 行以省略号结尾
    ellipsis_lines = sum(1 for l in lines if l.rstrip().endswith('...'))
    if ellipsis_lines / len(lines) > 0.3:
        return False
    
    # 规则 4: ≥80% 词包含字母
    words_with_alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if words_with_alpha / len(words) < 0.8:
        return False
    
    return True
```

### 经验
- **规则是启发式的**：基于统计观察而非绝对真理
- **需要平衡**：太严格会过滤好内容，太宽松会保留垃圾
- **测试用例是最好的文档**：边界条件一目了然

---

## 6. PII 遮蔽的边界情况

### 邮箱遮蔽的挑战

**场景 1：多个邮箱**
```python
text = "contact pl@ai.com and spl@ai.com"
# 需要正确替换两个邮箱
```

**场景 2：已经遮蔽的文本**
```python
text = "email |||EMAIL_ADDRESS||| or new@test.com"
# 只替换新的邮箱，保留已有的标记
```

### 实现技巧

```python
import re

EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

def mask_emails(text: str) -> tuple[str, int]:
    matches = list(EMAIL_PATTERN.finditer(text))
    num_masked = len(matches)
    
    # 从后往前替换，避免位置偏移
    masked_text = text
    for match in reversed(matches):
        masked_text = (masked_text[:match.start()] + 
                      "|||EMAIL_ADDRESS|||" + 
                      masked_text[match.end():])
    
    return masked_text, num_masked
```

### 经验
- **从后往前替换**：避免字符串长度变化导致的位置偏移
- **正则需要足够宽松**：匹配各种合法邮箱格式
- **统计替换数量**：返回 `(masked_text, count)` 元组

---

## 7. 语言识别模型加载

### 模型路径处理

```python
import os
import fasttext

# 获取相对于当前文件的路径
_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "models", "lid.176.bin"
)

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = fasttext.load_model(_MODEL_PATH)
    return _model
```

### 经验
- **使用懒加载**：避免导入时立即加载大模型
- **相对路径要处理正确**：`__file__` 是可靠的参考点
- **模型文件单独存放**：126MB 文件不应放在代码目录

---

## 8. 测试驱动开发技巧

### 快速迭代流程

```bash
# 1. 运行单个测试，快速验证
uv run pytest tests/test_pii.py::test_mask_emails_single -v

# 2. 运行相关模块的所有测试
uv run pytest tests/test_pii.py -v

# 3. 运行全部测试
uv run pytest tests/ -v
```

### 调试技巧

```python
# 在测试失败时打印详细信息
print(f"Expected: {expected!r}")
print(f"Actual:   {actual!r}")
```

### 经验
- **小步快跑**：每次只改一个功能，立即测试
- **关注错误信息**：测试框架会显示 ACTUAL vs DESIRED
- **使用 `!r` 格式**：显示字符串的原始表示，便于发现空白字符差异

---

## 9. 依赖管理最佳实践

### uv 工作流

```bash
# 添加依赖
uv add package_name

# 添加开发依赖
uv add --dev pytest

# 同步环境
uv sync

# 运行命令
uv run pytest

# 重新生成锁文件（依赖变更后）
rm uv.lock && uv lock
```

### 关键原则
1. **pyproject.toml 是源头**：所有依赖声明在这里
2. **uv.lock 是快照**：确保环境可复现
3. **修改依赖后重新 lock**：否则旧依赖仍生效
4. **提交 uv.lock 到 git**：确保团队环境一致

---

## 总结

### 关键技术点
1. **HTML 提取**: resiliparse 库需要字符串输入
2. **语言识别**: fasttext-wheel 是 Windows 友好选择
3. **去重算法**: 精确去重删模板，MinHash 去重相似文档
4. **质量过滤**: Gopher 规则基于启发式统计
5. **PII 遮蔽**: 正则 + 从后往前替换

### 工程实践
1. **测试先行**：理解测试期望再实现
2. **小步迭代**：单个测试 -> 模块 -> 全部
3. **文档化**：LESSONS_LEARNED 记录关键决策
4. **跨平台**：Windows 上注意编码和编译问题
