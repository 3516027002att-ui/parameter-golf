# Long-term Memory

## Stable Facts (长期稳定事实)
- 项目参赛目标：OpenAI Parameter Golf 挑战赛
- 硬约束：16MB artifact 上限（无时间限制）
- 评估指标：FineWeb val_bpb（越低越好）
- 当前 SOTA：1.1428（10L Int5MLP + BigramHash + SWA）
- 本地训练配置：`max_wallclock_seconds=0`，`iterations=6000`，`train_batch_tokens=8192`，`train_seq_len=1024`

## Verified Improvements (已验证有效)
- Int5MLP 量化 + Int6 Attention（当前最优量化组合）
- BigramHash(10240) 嵌入提升效果
- SWA (Stochastic Weight Averaging) frac=0.4 稳定最终性能
- Sliding Window Eval (stride=64) 标准评估方式
- SmearGate + OrthoInit 架构增益
- 3x MLP 扩展（公开记录最大贡献）
- FP16 tok_emb passthrough 减少量化损失

## Failed Experiments (已验证无效)
- **plan1.py**：Depth Recurrence + QAT + Outlier Isolation — 容量砍太狠（17M→4.2M参数），效果系统性落后 baseline
- **plan2.py**：Muon WD 方案 — 同时改 seq_len/batch/LR，变量混杂无法归因
- **depth recurrence 方向**：PR #38, #298 都比 baseline 差，未验证能赢

## 六条避坑原则 (核心教训)
1. 不要提前为 size 付训练税 — baseline 已经能进 16MB
2. 不要把不同 seq_len 的日志拿来直接比较
3. 不要把 LR=0.02 当成普适真理 — 它是 WD bundle 的一部分
4. 不要做全程 QAT — 收益抵不过开销
5. 不要先上 10240 BigramHash — 先用小桶验证
6. 不要忽视 eval-only 增益（sliding eval）

## Current Focus (当前重点)
- plan3a：fp16 tok_emb + 保持 9 层 + warmdown 重调
- plan3b：Muon WD bundle（0.02 LR / 0.99 momentum / long warmup / clip=0.3）
- plan3c：SmearGate + BigramHash(4096) + OrthoInit + 3x MLP
- plan3d：mixed int5/int6 + 10th layer（本轮不加 late QAT/EMA/GPTQ-lite）

## Local Environment Notes (本地环境备注)
- GPU: RTX 4060 Laptop 8GB VRAM, CUDA 12.7
- Windows 特化：禁用 GQA、flash SDPA、torch.compile
- 单次训练约 12-13 分钟（6000 步）
- 启动命令：`.venv\Scripts\python.exe plan3a.py`（或 plan3b/c/d）

## Cross-Project References (跨项目引用)
- 中央层：`D:\Repository\opencontext-central`
- 经验库：`D:\Repository\agents-exp`
- 本项目不承担跨项目知识沉淀

## Memory Update Rules (记忆更新规则)
- 长期稳定事实写本文件
- 重要决策写 `decisions.md`
- 待办事项写 `todo.md`
- 会话交接写 `handoff.md`
- 跨项目可复用知识写回中央层