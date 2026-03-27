# parameter-golf 记忆入口

## 系统文件
- [project.md](system/project.md) — 项目概述、约束、关键文件
- [architecture.md](system/architecture.md) — 架构、技术栈、数据流

## 记忆文件
- [long-term.md](memory/long-term.md) — 长期稳定事实、六条避坑原则
- [decisions.md](memory/decisions.md) — 重要技术决策
- [todo.md](memory/todo.md) — 待办事项
- [handoff.md](memory/handoff.md) — 会话交接信息

## 快速入口
- 主训练脚本: `train_gpt.py`
- 当前实验: `plan3a.py` ~ `plan3d.py`
- 实验记录: `experiments.md`
- 修改留痕: `修改记录.md`
- 踩坑记录: `LESSONS_LEARNED.md`（必读！）

## 当前目标
运行 plan3a-d 渐进式验证，超越当前 SOTA (val_bpb = 1.1428)

## 本地训练配置
- 无时间限制（`max_wallclock_seconds=0`）
- iterations=6000, batch_tokens=8192, seq_len=1024
- Windows 特化：禁用 GQA、flash SDPA、torch.compile