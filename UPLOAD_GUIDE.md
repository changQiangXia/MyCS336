# GitHub 上传指南

本文档指导如何将本仓库上传到 GitHub。

## 方式一：命令行上传（推荐）

### 1. 初始化 Git 仓库

在 `D:\pythonProjects\CS336` 目录下打开终端，执行：

```bash
# 初始化 Git 仓库
git init

# 配置你的信息（如果还没配置过）
git config user.name "你的名字"
git config user.email "你的邮箱@example.com"
```

### 2. 添加文件到 Git

```bash
# 添加所有文件（除了 .gitignore 中列出的）
git add .

# 检查状态
git status
```

### 3. 提交（Commit）

```bash
# 创建第一个提交
git commit -m "feat: 完成作业1 - Basics

- 实现神经网络基础组件: Linear, Embedding, RMSNorm, SiLU, SwiGLU
- 实现注意力机制: SDPA, Multi-Head Attention, RoPE
- 实现 Transformer 模型: TransformerBlock, TransformerLM
- 实现优化器: AdamW, Cosine LR Schedule
- 实现 BPE Tokenizer 及训练算法
- 通过所有 46 个测试用例

性能优化:
- BPE 训练从 O(V×N×M) 优化到 O(V×K×M)
- 训练速度从 2.3s 提升到 0.8s
- 使用四位一体数据结构实现增量更新"
```

### 4. 创建 GitHub 仓库并推送

1. 打开 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `CS336` （或你喜欢的名字）
   - **Description**: `Stanford CS336: Language Modeling from Scratch - 作业合集`
   - **Visibility**: 选择 Private（私密）或 Public（公开）
   - **不要勾选** "Initialize this repository with a README" （因为我们本地已有）
3. 点击 **Create repository**
4. 复制页面上显示的推送命令（类似下面）：

```bash
# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/CS336.git

# 推送代码
git branch -M main
git push -u origin main
```

## 方式二：GitHub Desktop 上传（图形界面）

1. 下载安装 [GitHub Desktop](https://desktop.github.com/)
2. 点击 "File" → "Add local repository"
3. 选择 `D:\pythonProjects\CS336` 文件夹
4. 填写提交信息，点击 "Commit to main"
5. 点击 "Publish repository"
6. 填写仓库名，选择是否公开，点击 "Publish"

## 方式三：直接拖拽上传（最简单但不推荐）

1. 打开 https://github.com/new 创建空仓库
2. 不要初始化 README
3. 在仓库页面点击 "uploading an existing file"
4. 拖拽文件上传
5. **缺点**: 没有 Git 历史记录，不适合后续更新

## 后续作业更新流程

做完作业2后，按以下流程更新：

```bash
# 进入项目目录
cd D:\pythonProjects\CS336

# 查看修改了哪些文件
git status

# 添加作业2的文件
git add 2/

# 提交
git commit -m "feat: 完成作业2 - Systems

- 实现 DDP (Distributed Data Parallel)
- 实现分片优化器
- xxxxx"

# 推送到 GitHub
git push
```

## 常见问题

### Q: 忘记配置用户名和邮箱
```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱@example.com"
```

### Q: 推送时报错 "rejected"
```bash
# 先拉取远程更新
git pull origin main

# 然后再推送
git push origin main
```

### Q: 不想上传某些文件
编辑 `.gitignore` 文件，添加要忽略的文件/文件夹。

## 验证上传成功

上传完成后，在浏览器访问 `https://github.com/YOUR_USERNAME/CS336` 应该能看到所有文件。

---

**建议**: 第一次使用 Git 的话，推荐用 **方式二（GitHub Desktop）**，有图形界面比较直观。
