# CLAUDE.md — parameter-golf 项目指引

## 项目概述

OpenAI Parameter Golf 挑战赛：在 16MB artifact + 10分钟 8×H100 约束下，训练最优语言模型（评估指标：FineWeb val_bpb）。

## 仓库结构

- `train_gpt.py` — 原版训练脚本（标准9层堆叠 + U-Net skip）
- `plan1.py` — 改进方案（Depth Recurrence + QAT + Outlier Isolation）
- `train_gpt_origin.py` — 原始 baseline 备份，不要修改
- `train_gpt_mlx.py` — Apple MLX 版，非主力方案
- `data/` — 数据集和 tokenizer
- `records/` — 历史实验记录（只读参考，不要修改已有记录）
- `experiments.md` — 实验对比记录表，每次跑完实验必须更新
- `修改记录.md` — 修改留痕，每次改动必须更新

## 硬约束（不可违反）

1. **16MB 上限**：代码字节 + 压缩模型字节 ≤ 16,000,000 bytes（十进制，非 MiB）
2. **10分钟训练**：`max_wallclock_seconds=600`，不可放宽
3. **评估不可访问训练数据**：推理时禁止读取训练集
4. **验证集不可用于训练**：不可在评估前对验证集做 test-time training
5. **自包含**：评估时禁止外部下载或网络调用
6. **SOTA 提交门槛**：必须比当前 SOTA 好 ≥ 0.005 nats，p < 0.01（通常需要 3 个 seed）

## 当前状态

- 当前 SOTA：val_bpb = **1.1428**（10L Int5MLP + BigramHash + SWA）
- 基于 `train_gpt.py` 的最佳配置
- `plan1.py` 尚未跑出结果，待验证

## 工作流约束

### 修改代码前

- 先读 `experiments.md` 了解历史实验和已验证的改进方向
- 先读目标文件，理解现有逻辑再动手
- 确认改动不会突破 16MB 限制（可用 `wc -c` 估算）

### 修改代码时

- `train_gpt.py` 和 `plan1.py` 是两套独立方案，改一个不要影响另一个
- 不要动 `records/` 下的已有实验记录
- 不要动 `train_gpt_origin.py`（原始备份）
- 新实验配置优先通过环境变量控制，不要硬编码到脚本里
- 量化相关改动必须验证 roundtrip（量化→反量化→推理）不会 break

### 修改代码后

- 更新 `修改记录.md`
- 语法检查：`python -m py_compile <文件>`
- 行数检查：确认不超过合理范围（当前 train_gpt.py ~1287行，plan1.py ~1476行）
- 如果跑了训练，把结果填入 `experiments.md`

### 跑实验

- 本地 GPU：NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)，CUDA 12.7，支持本地训练
- 单次训练+验证约 12~13 分钟
- 3 seed 串行约 40 分钟
- 启动命令模板：
  ```bash
  RUN_ID=<实验名> \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```

## 关键技术栈

- PyTorch（CUDA）、Muon + Adam 混合优化器
- Int5/Int6/Int8 量化 + zstd/zlib 压缩
- Sliding Window Eval（stride=64）
- SWA（Stochastic Weight Averaging）
- SmearGate、BigramHash、OrthoInit

## 改进方向优先级

1. 验证 plan1.py（Depth Recurrence + QAT + OI）是否超越当前 SOTA
2. 调优 plan1 的超参（loop数、shared block数、OI topk）
3. 探索更激进的量化（Int4 + OI）
4. 架构创新（新的参数共享方式、更高效的注意力机制）
