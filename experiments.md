# 实验记录

## 训练环境

- 本地 GPU：NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)，CUDA 12.7
- 单次训练+验证约 12~13 分钟
- 3 seed 串行约 40 分钟

## 训练方案说明

本仓库有两套训练方案：

| 方案 | 文件 | 架构 | 核心特点 |
|------|------|------|----------|
| 原版 (baseline) | `train_gpt.py` | 9层独立Block + U-Net skip | 标准堆叠Transformer，每层独立参数 |
| Plan1 | `plan1.py` | 2 shared blocks × 4 loops | Depth Recurrence + QAT + Outlier Isolation |

### Plan1 相对原版的改动

来源：修改记录 2026-03-22 21:50:00

1. **Depth Recurrence（深度循环）**
   - 原版：9个独立Block + `skip_weights` U-Net skip connection
   - Plan1：2个共享Block循环4次（等效8层），新增 `loop_scales[4,2,dim]` 作为 per-loop scale adapter
   - 目的：用更少参数量换取等效深度，在16MB限制下争取更多表达能力

2. **STE Fake-Quant / QAT（量化感知训练）**
   - 原版：无，训练时不考虑量化
   - Plan1：新增 `_FakeQuantSTE`，训练时对 shared block 的 CastedLinear 权重做 per-row int8 fake quantize，backward 直通梯度
   - 目的：让训练时的权重分布更贴近最终 int8 导出形态，减少量化损失
   - 开关：`ENABLE_QAT` 环境变量，默认开启

3. **Outlier Column 隔离量化**
   - 原版：统一 per-row int8 量化
   - Plan1：新增 `isolate_outlier_cols`，top-16 列存 fp16 sidecar，其余走 per-row int8
   - 目的：减少异常列对量化精度的破坏
   - 参数：`OI_TOPK=16`

4. **移除的模块**
   - `skip_weights`、encoder/decoder 分半、skip stack 全部移除
   - `CONTROL_TENSOR_NAME_PATTERNS` 中 `skip_weight,skip_weights` 替换为 `loop_scale,loop_scales`

5. **未改动的部分**
   - 模型基础参数：`model_dim=512`, `num_heads=8`, `num_kv_heads=4`, `mlp_mult=2`, `vocab_size=1024`
   - 训练超参：学习率、优化器、batch size、序列长度、迭代数全部一致
   - 评估方式：`eval_val` 函数完全相同

### Plan3a-d 独立实验矩阵（已实现，待正式跑分）

说明：
- `train_gpt.py` 与 `train_gpt_origin.py` 保持不动。
- `plan3a.py`、`plan3b.py`、`plan3c.py`、`plan3d.py` 都是从当前 `train_gpt.py` 独立拷出，各自单独维护。
- 四个脚本统一默认：`iterations=6000`、`max_wallclock_seconds=0`、`train_batch_tokens=8192`、`train_seq_len=1024`。
- 训练日志统一带 `lr_scale`、`grad_norm`、`tokens`，并保留 `step_avg`、验证行、roundtrip 与 `quant_gap`。

| 版本 | 文件 | 累计 bundle | 固定日志文件 | 备注 |
|------|------|-------------|--------------|------|
| Plan3a | `plan3a.py` | `tok_emb` fp16 passthrough + `warmdown_iters=3600` | `logs/plan3a_v1.txt` | 保持 baseline 9 层、U-Net skip、baseline optimizer split |
| Plan3b | `plan3b.py` | Plan3a + Muon WD bundle | `logs/plan3b_v1.txt` | `muon_wd=0.04`, `matrix_lr=0.02`, `scalar_lr=0.02`, `tied_embed_lr=0.03`, `momentum=0.99`, warmup=1500, clip=0.3 |
| Plan3c | `plan3c.py` | Plan3b + 3x MLP + SmearGate + BigramHash + OrthoInit | `logs/plan3c_v1.txt` | `bigram_hash_buckets=4096`, `bigram_hash_dim=128`, 仍保持 9 层 |
| Plan3d | `plan3d.py` | Plan3c + mixed int5/int6 export | `logs/plan3d_v1.txt` | MLP 走 int5，Attention 走 int6；本轮明确不加第 10 层、late QAT、EMA、GPTQ-lite |

| Launcher | 对应脚本 | 说明 |
|----------|----------|------|
| `run_plan3a.bat` | `plan3a.py` | 最薄 launcher，只负责调用脚本 |
| `run_plan3b.bat` | `plan3b.py` | 最薄 launcher，只负责调用脚本 |
| `run_plan3c.bat` | `plan3c.py` | 最薄 launcher，只负责调用脚本 |
| `run_plan3d.bat` | `plan3d.py` | 最薄 launcher，只负责调用脚本 |

---

## 历史实验结果（原版 train_gpt.py）

按时间顺序，所有实验基于 `train_gpt.py` 的不同配置变体。

### track_10min_16mb（本地 GPU 训练）

| 日期 | 实验名称 | val_bpb | 模型大小 | 训练步数 | 关键配置 |
|------|----------|---------|----------|----------|----------|
| 03-17 | Naive Baseline | 1.2244 | 15.9MB | 13,780 | 9L 512dim, seq1024, int8+zlib |
| 03-17 | LoRA TTT | 1.1929 | 15.9MB | 同上 | 同baseline + 推理时 rank-8 LoRA TTT |
| 03-18 | FP16 Embed + WD3600 | 1.2197 | 15.9MB | 13,692 | FP16 embed, MLP_HIDDEN=992, WARMDOWN=3600 |
| 03-18 | Long Context Seq2048 | 1.2058 | 15.9MB | 11,564 | seq2048, LR调整 |
| 03-18 | Lower LR | 1.2230 | 15.9MB | 14,421 | MATRIX_LR=0.02, SCALAR_LR=0.02 |
| 03-19 | 10L Mixed Precision | 1.2147 | 15.9MB | 13,100 | 10L, int6中间层(3-6) |
| 03-19 | Training Opt Seq4096 | 1.2014 | 15.9MB | 8,394 | seq4096, Muon momentum=0.99 |
| 03-19 | Sliding Window Eval | 1.1925 | 15.9MB | 13,450 | 同baseline + sliding window stride=64 |
| 03-19 | Warmdown Quant Int6 MLP3x | 1.1574 | 16.0MB | ~12,200 | WARMDOWN=20000, FP16 embed, MLP3x, seq2048, SW |
| 03-19 | Mixed Quant Int6+Int8 SW | 1.1630 | 15.4MB | 12,395 | MLP3x, int6 block + int8 embed, SW |
| 03-19 | Int6 QAT MLP2.6x Muon SW | 1.1586 | 15.6MB | 8,319 | 10L, STE int6 QAT, zstd-22, MLP_HIDDEN=1344, seq2048, SW |
| 03-19 | SW FP16Emb 10L MuonWD OvertoneInit | 1.1748 | 15.4MB | ~10,500 | 10L, FP16 embed, Muon WD=0.02, overtone init, SW |
| 03-19 | SmearGate OrthoInit MuonWD | 1.1556 | 15.9MB | 12,047 | 9L MLP3x, SmearGate, BigramHash, OrthoInit, int6+zstd-22, SW |
| 03-19 | Seq2048 FP16Emb TunedLR | 1.1502 | 15.4MB | 10,070 | 11L MLP3x, QAT int6, zstd-22, FP16 embed, SW, Muon WD=0.04 |
| 03-20 | Int6 MLP3x SmearGate BigramHash SWA | 1.1458 | 15.9MB | 7,379 | 9L MLP3x, SmearGate, BigramHash, SWA/50步, int6+zstd-22, seq2048 |
| 03-20 | **10L Int5MLP MuonWD04 SWA50** | **1.1428** | ~15.9MB | — | 10L, int5 MLP + int6 attn, BigramHash 10240, SWA frac=0.4, seq2048 |

### track_non_record_16mb（远程 GPU 训练）

| 日期 | 实验名称 | val_bpb | 模型大小 | 训练步数 | 关键配置 |
|------|----------|---------|----------|----------|----------|
| 03-18 | 4h Quasi-10B (无时间限制) | 1.2074 | 15.8MB | 329,430 | 9L 512dim, 8×H100, 4小时 |
| 03-19 | SwiGLU 1×5090 | 1.3281 | 15.3MB | 3,773 | SwiGLU, warmdown=0.2, batch=131K, 单卡5090 |

---

## Plan1 实验结果

| 日期 | 版本 | val_bpb | 模型大小 | 关键配置 | 备注 |
|------|------|---------|----------|----------|------|
| — | — | — | — | — | 尚未运行 |

---

## 改进趋势总结

- Baseline → 当前最佳：1.2244 → **1.1428**，累计改善 0.082 bpb
- 最大单项贡献：Sliding Window Eval（约 -0.032 bpb，零训练成本）
- Int6 QAT + zstd-22 解锁了更宽 MLP（3x）和更多层（10-11L）
- SmearGate + BigramHash + SWA 是近期叠加的主要架构改进
- Int5 MLP 混合量化进一步压缩模型体积，释放参数空间

---

## Plan3 系列（2026-03-25）

### 设计思路

Plan1 失败原因：depth-recurrent 砍掉 3/4 参数量，去解决不存在的 size 问题。
Plan2 失败原因：同时改了 batch/seq_len/optimizer，结果不可归因。

Plan3 策略：分阶段递进验证，每步只加一个 bundle，保持可比 regime。

### 四个分支

| 分支 | 基于 | 新增改动 | 关键超参 |
|------|------|----------|----------|
| plan3a | baseline | fp16 tok_emb + warmdown 900 | LR=0.04, momentum=0.95, no WD |
| plan3b | plan3a | Muon WD bundle | LR=0.02, momentum=0.99, WD=0.04, clip=0.3 |
| plan3c | plan3b | 3x MLP + SmearGate + BigramHash(4096) + OrthoInit | 同 plan3b |
| plan3d | plan3c | 10th layer + mixed int5/int6 quant | 同 plan3c, 10 layers |

全部 6000 步，batch=8192，seq=1024，val_loss_every=1000。

### 本地 vs H100 的可信度分层

- 架构改动（C 的 3x MLP、SmearGate、BigramHash）：本地最可信
- 优化器超参（B 的 LR/momentum/WD）：本地结果要打折，batch 差 96 倍
- 量化改动（A 的 fp16、D 的 mixed quant）：看 artifact 大小和 roundtrip 误差

### 代码审查修复（3/25）

1. val_batch_size：524288 → 8192（全部，防 OOM）
2. warmdown_iters：3600 → 900（全部，6000 步下 15% 比例）
3. momentum_warmup_steps：1500 → 450（plan3b/c/d，按比例缩放）
4. num_layers：9 → 10（plan3d，原代码漏改）
5. SmearGate gate init：3.0 → 0（plan3c/d，对齐 SOTA）
   - SOTA 用 gate=0（sigmoid=0.5）
   - 原代码 gate=3.0（sigmoid=0.95）太激进，几乎完全用前一个 token
   - Agent 建议 gate=-4（sigmoid=0.018）保守冷启动，但选择跟 SOTA 一致

### 关键风险预判

- plan3b 大概率在本地表现差（WD×LR×6000步 → 参数衰减到 0.8%），不代表 H100 上也差
- 如果 B 差，用梯度累积 ×96 跑 200 步验证大 batch 行为
- plan3d 不会 OOM（估算总显存 ~1.6GB）

## 待验证

- [ ] plan3a 6000 步 val_bpb（预期 ≈ baseline）
- [ ] plan3b 6000 步 val_bpb（预期可能比 a 差，batch 差异）
- [ ] plan3c 6000 步 val_bpb（最可信的架构验证）
- [ ] plan3d 6000 步 val_bpb + artifact 大小
- [ ] EMA 实现（下一步优先）
- [ ] 11L 可行性验证
- [ ] Partial RoPE 实现
