@echo off
chcp 65001 >nul
echo ==========================================
echo   CS336 Git 仓库初始化脚本
echo ==========================================
echo.

REM 检查是否在正确的目录
if not exist "1" (
    echo 错误：请在 CS336 根目录运行此脚本
    pause
    exit /b 1
)

echo [1/5] 初始化 Git 仓库...
git init

echo.
echo [2/5] 添加文件到 Git...
git add .

echo.
echo [3/5] 创建提交...
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

echo.
echo [4/5] 设置分支名为 main...
git branch -M main

echo.
echo ==========================================
echo   初始化完成！
echo ==========================================
echo.
echo 下一步：
echo 1. 在 GitHub 上创建仓库：https://github.com/new
echo 2. 不要勾选 "Initialize this repository with a README"
echo 3. 创建后，复制页面上的推送命令执行
echo.
echo 或者直接运行以下命令（替换 YOUR_USERNAME）：
echo   git remote add origin https://github.com/YOUR_USERNAME/CS336.git
echo   git push -u origin main
echo.
pause
