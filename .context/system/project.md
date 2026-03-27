---
description: "Project: parameter-golf"
limit: 10000
---

# parameter-golf

## Purpose
- OpenAI Parameter Golf 挑战赛参赛项目。
- 目标：在 16MB artifact 约束下，训练最优语言模型。
- 评估指标：FineWeb val_bpb（bits per byte），越低越好。

## Critical Constraints (不可违反)
1. **16MB 上限**：代码字节 + 压缩模型字节 ≤ 16,000,000 bytes（十进制，非 MiB）
2. **评估不可访问训练数据**：推理时禁止读取训练集
3. **验证集不可用于训练**：不可在评估前对验证集做 test-time training
4. **自包含**：评估时禁止外部下载或网络调用
5. **SOTA 提交门槛**：必须比当前 SOTA 好 ≥ 0.005 nats，p < 0.01（通常需要 3 个 seed）

## Local Training Config (本地训练配置)
- **无时间限制**：`max_wallclock_seconds=0`，不设上限
- **默认 iterations=6000**
- **默认 train_batch_tokens=8192, train_seq_len=1024**（适配 8GB VRAM）

## Current Status (当前状态)
- 当前 SOTA：val_bpb = **1.1428**（10L Int5MLP + BigramHash + SWA）
- 基于配置：`train_gpt.py`
- `plan1.py` 已验证失败（容量砍太狠，效果变差）
- `plan2.py` 已验证失败（同时改太多变量，无法归因）
- 当前实验方向：`plan3a.py` ~ `plan3d.py`（渐进式验证）

## Key Files
- `train_gpt.py` — 主力训练脚本（标准9层堆叠 + U-Net skip）
- `train_gpt_origin.py` — 原始 baseline 备份（不可修改）
- `plan1.py` — Depth Recurrence 方案（已验证失败）
- `plan2.py` — Muon WD 方案（已验证失败，变量混杂）
- `plan3a.py` ~ `plan3d.py` — 当前实验矩阵（渐进式改进）
- `train_gpt_mlx.py` — Apple MLX 版，非主力方案
- `data/` — 数据集和 tokenizer
- `records/` — 历史实验记录（只读）
- `experiments.md` — 实验对比记录表（每次实验后必须更新）
- `修改记录.md` — 修改留痕（每次改动必须更新）
- `LESSONS_LEARNED.md` — 踩坑记录与改进建议

## Local Hardware
- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- CUDA: 12.7
- 单次训练约 12~13 分钟（6000 步）
- Windows 特化：禁用 GQA、flash SDPA、torch.compile

## External Dependencies
- 中央层：`D:\Repository\opencontext-central`
- 经验库：`D:\Repository\agents-exp`

## Project Memory Contract
- `.context/memory/long-term.md` — 项目专属长期稳定知识
- `.context/memory/decisions.md` — 重要技术决策
- `.context/memory/todo.md` — 待办事项
- `.context/memory/handoff.md` — 会话交接信息