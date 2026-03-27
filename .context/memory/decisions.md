# Decisions

## 2026-03-25: 更新记忆系统配置
- **决策**: 根据实际项目状态更新记忆文档
- **修正内容**:
  - 训练时间：无限制（`max_wallclock_seconds=0`），非 10 分钟
  - plan1/plan2：已验证失败，非"待验证"
  - 当前实验方向：plan3a-d 渐进式验证

## 2026-03-25: 创建项目记忆系统
- **决策**: 为 parameter-golf 项目创建独立的 `.context/` 记忆系统
- **原因**: 
  - 项目有严格约束和复杂技术栈
  - 需要跨会话保持实验状态和决策历史
  - 与中央层分离，专注项目专属知识
- **影响**: 
  - 长期记忆存于 `.context/memory/long-term.md`
  - 不影响中央层 `opencontext-central` 结构

## 2026-03-23: plan3a-d 实验矩阵设计
- **决策**: 采用渐进式验证，每次只改一个 bundle
- **原因**: plan1/plan2 同时改太多变量，无法归因
- **路径**:
  1. plan3a：fp16 tok_emb + 保持 baseline 架构
  2. plan3b：Muon WD bundle 成套带入
  3. plan3c：SmearGate + BigramHash + 3x MLP
  4. plan3d：mixed int5/int6 + 第 10 层

## 2026-03-23: Windows 本地训练适配
- **决策**: 为 Windows 环境添加特化兼容代码
- **内容**: 禁用 GQA、flash SDPA、torch.compile
- **原因**: Windows + PyTorch 2.10 不支持这些特性
- **技术债**: 使用 `sys.platform != "win32"` 条件分支，不影响 Linux 生产环境

## Historical Decisions (来自 experiments.md)
- Int5MLP + Int6 Attention 混合量化（当前最优）
- BigramHash(10240) + SWA(frac=0.4) 组合
- U-Net skip connections 架构
- Sliding Window Eval (stride=64) 评估方式