# Handoff

## 2026-03-25: 更新记忆系统配置
- **更新者**: 小野
- **状态**: 已根据实际项目状态更新记忆文档
- **修正内容**:
  - 训练时间：无限制，非 10 分钟
  - plan1/plan2：已验证失败
  - 当前实验：plan3a-d
  - 六条避坑原则已记录到 long-term.md

## 关键文件阅读清单
- `experiments.md` — 完整实验历史
- `修改记录.md` — 所有改动留痕
- `LESSONS_LEARNED.md` — 踩坑记录（重要！）
- `CLAUDE.md` — 项目指引

## 留意事项
- `records/` 目录只读，不要修改历史记录
- `train_gpt_origin.py` 不要动
- `train_gpt.py` 和 `plan3*.py` 是独立方案，改一个不影响另一个
- 每次改动必须更新 `修改记录.md`
- 每次实验后必须更新 `experiments.md`
- 参考 `LESSONS_LEARNED.md` 避免重蹈覆辙

## 下一步
- 正式运行 plan3a-d 并记录结果
- 根据结果确定最优配置
- 不要同时改太多变量