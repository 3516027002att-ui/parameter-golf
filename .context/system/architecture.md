---
description: "Architecture: parameter-golf"
limit: 8000
---

# Architecture Overview

## Model Architecture (train_gpt.py)
- **Baseline**: 标准 Transformer decoder-only 架构
- **Layers**: 9层堆叠 + U-Net skip connections
- **Attention**: 支持 SWA (Sliding Window Attention)
- **MLP**: Int5/Int6/Int8 量化支持
- **Special**: BigramHash, SmearGate, OrthoInit

## Model Architecture (plan1.py)
- **Core Innovation**: Depth Recurrence（深度循环）
- **Quantization**: QAT (Quantization-Aware Training)
- **Optimization**: Outlier Isolation（离群值隔离）
- **Status**: 待验证，尚未跑出结果

## Optimizer
- Muon + Adam 混合优化器
- SWA (Stochastic Weight Averaging) 支持

## Compression Pipeline
1. 训练后量化：Int5/Int6/Int8
2. 压缩算法：zstd / zlib
3. 总大小约束：代码 + 压缩模型 ≤ 16MB

## Evaluation
- Sliding Window Eval (stride=64)
- 指标：FineWeb val_bpb（越低越好）
- 当前最佳：1.1428

## Data Flow
```
data/datasets/fineweb10B_sp1024/ 
  → Tokenizer (fineweb_1024_bpe.model) 
  → Model Training 
  → Quantization 
  → Compression 
  → 16MB Artifact
```

## Key Technical Decisions
1. 量化层级选择：Int5MLP 当前最优
2. 注意力机制：BigramHash + SWA 组合
3. 参数共享：U-Net skip connections
4. 训练时间分配：600秒硬约束

## Known Limitations
- 本地 GPU 仅 8GB VRAM，无法跑完整 8×H100 配置
- 需要适配本地环境进行验证
- 部分超参调优受限于硬件