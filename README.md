# CS336 Spring 2025 Assignments

Stanford CS336: Language Modeling from Scratch - 作业合集

## 📚 课程信息

- **课程**: CS336: Language Modeling from Scratch (Spring 2025)
- **内容**: 从0开始构建大语言模型
- **语言**: Python 3.11+
- **框架**: PyTorch 2.6

## 📁 仓库结构

```
CS336/
├── 1/                          # 作业1: Basics
│   └── assignment1-basics-main/
│       ├── cs336_basics/       # 核心实现
│       │   ├── nn_utils.py     # 神经网络基础组件
│       │   ├── model.py        # Transformer模型
│       │   ├── optimizer.py    # 优化器和学习率调度
│       │   ├── tokenizer.py    # BPE Tokenizer
│       │   ├── data.py         # 数据加载
│       │   └── serialization.py # 模型保存/加载
│       ├── tests/              # 测试用例
│       └── LESSONS_LEARNED.md  # 作业1经验总结
│
├── 2/                          # 作业2: Systems (待完成)
│   └── assignment2-systems-main/
│
├── 3/                          # 作业3: Scaling (待完成)
│   └── assignment3-scaling-main/
│
├── 4/                          # 作业4: Data (待完成)
│   └── assignment4-data-main/
│
├── 5/                          # 作业5: Alignment (待完成)
│   └── assignment5-alignment-main/
│
├── .gitignore                  # Git忽略配置
├── README.md                   # 本文件
└── LESSONS_LEARNED.md          # 全课程经验总结
```

## ✅ 作业完成情况

| 作业 | 主题 | 测试通过率 | 状态 |
|------|------|-----------|------|
| Assignment 1 | Basics (基础组件) | 46/46 (100%) | ✅ 完成 |
| Assignment 2 | Systems (分布式训练) | - | ⏳ 待开始 |
| Assignment 3 | Scaling (扩展法则) | - | ⏳ 待开始 |
| Assignment 4 | Data (数据处理) | - | ⏳ 待开始 |
| Assignment 5 | Alignment (模型对齐) | - | ⏳ 待开始 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) 包管理器（推荐）
- Git

### 安装依赖

进入任意作业目录：

```bash
cd 1/assignment1-basics-main
uv sync  # 自动安装所有依赖
```

### 运行测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行单个测试
uv run pytest tests/test_model.py::test_transformer_lm -v
```

## 📝 作业1 详细内容

### 实现的功能

#### 1. 神经网络基础组件 (`nn_utils.py`)
- ✅ Linear 层
- ✅ Embedding 层
- ✅ RMSNorm
- ✅ SiLU / SwiGLU 激活函数
- ✅ Softmax (数值稳定版)
- ✅ Cross Entropy Loss
- ✅ Gradient Clipping

#### 2. 注意力机制 (`nn_utils.py`)
- ✅ Scaled Dot-Product Attention
- ✅ Multi-Head Self-Attention
- ✅ RoPE (Rotary Position Embedding)

#### 3. Transformer 模型 (`model.py`)
- ✅ TransformerBlock (Pre-Norm)
- ✅ TransformerLM (完整语言模型)

#### 4. 训练相关 (`optimizer.py`, `data.py`, `serialization.py`)
- ✅ AdamW 优化器
- ✅ Cosine Learning Rate Schedule (带 Warmup)
- ✅ 数据采样 (`get_batch`)
- ✅ 模型检查点保存/加载

#### 5. Tokenizer (`tokenizer.py`)
- ✅ BPE 编码/解码
- ✅ BPE 训练算法 (增量更新优化)
- ✅ 特殊 Token 处理

### 性能优化亮点

**BPE 训练算法优化**：
- 时间复杂度：O(V × N × M) → O(V × K × M)
- 实际耗时：2.3秒 → 0.8秒
- 关键优化：四位一体数据结构 + 增量更新

详见 [LESSONS_LEARNED.md](LESSONS_LEARNED.md)

## 🐛 踩坑记录

项目开发过程中遇到的各类问题及解决方案，详见各作业的 `LESSONS_LEARNED.md`：

- [作业1 经验总结](1/assignment1-basics-main/LESSONS_LEARNED.md)

## 📄 License

本仓库为个人学习笔记，仅供学习交流使用。
