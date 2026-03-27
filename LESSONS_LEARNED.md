# 踩坑记录与改进建议

## plan1 失败分析

### 核心结论

**plan1 变差不是一个点的问题，而是"容量砍太狠 + 架构归纳偏置变弱 + 共享权重的优化没重新配平 + QAT 开得太早太重"叠加出来的结果。**

从日志看，它**不是在 6000 步突然坏掉**，而是从 1000 步开始就一直系统性落后：
- baseline 在 1000/6000/8000 步的 `val_loss` 大约是 3.4604 / 2.7712 / 2.7040
- plan1 是 3.6619 / 3.0188 / 2.9531
- 参数量从 17059912 直接降到 4202512，约只剩四分之一

这个走势更像**长期欠拟合/表达力不足**，不像中途数值崩盘。

---

### 主因分析（按重要性排序）

#### 1. 优化错了主要瓶颈（最关键）

**比赛限制的是 artifact 大小，不是原始参数量，所以没必要把训练模型直接砍到 4.2M。**

官方 FAQ 说得很清楚，计入上限的是"代码字节 + 压缩后的模型字节"。baseline 本地日志里，`train_gpt` 的 `final_model.int8.ptz` 约 15.38MB，总提交约 15.43MB，**已经在 16MB 内**。

也就是说，**baseline 17.1M 参数本来就能过体积线**。在这种前提下，plan1 用 2 个共享 block 循环 4 次，把训练态参数砍到 4.2M，本质是在用"训练表达力"去换一个其实并不缺的"压缩空间"，这几乎注定会输掉训练曲线。

现在排行榜上真正有效的路线，主流也不是"把模型做得更小"，而是靠更激进的量化/压缩去**容纳 10-11 层、甚至 3x MLP 的更大模型**。

#### 2. 同时改了太多变量

plan1 不是"同容量换结构"，而是同时做了：
- 全块共享（2 个 shared blocks × 4 loops）
- 去掉原版 U-Net skip
- 共享权重的优化没重新配平
- 全程 QAT

这让"为什么差"混在一起，无法定位具体问题。

**教训**：不要一次性同时改四个轴（结构、skip、优化器、QAT）。

#### 3. QAT 开得太早、太重、不匹配

plan1 里 `ENABLE_QAT=1` 默认打开，从 step 0 就走 fake-quant：
- 范围：shared blocks 所有 CastedLinear 都被打 `_fake_quant`
- 量化器不一致：训练时是 per-row `amax/127` round-trip，导出时是 `isolate_outlier_cols`（top-k outlier columns 侧带 fp16 + 其余 per-row int8）
- 噪声不匹配：训练时注入的噪声与真正部署时的失真并不一致

更重要的是，现有强 PR 里，QAT 的定位普遍更保守：
- 1.1502 把 int6 QAT 当作组件，但建立在更深模型、更低 LR、weight decay、zstd-22 的强底座上
- 1.1248 明确写了：他们原本打算开 Late QAT，但 `torch.compile` 把 flag 常量折叠掉了，**QAT 根本没生效**，最后得分主要来自 Partial RoPE 和 LN Scale

**结论**：在这个赛题里，**QAT 目前更像后期微调器，而不是主架构武器**。

#### 4. 共享权重后没有重新配平 LR

plan1 的 shared block 参数还是走 Muon，`matrix_lr=0.04`，`scalar_lr=0.04`，没有因为 `num_loops=4` 重新缩放。

相关 PR #38 里，作者加了 `shared_matrix_lr = args.matrix_lr / sqrt(num_loops)`，理由是 **weight tying 让 shared params 在一次 forward/backward 中累计更多梯度**。

**注意**：这条建议来源是 PR #38，但它本身不是强记录（val_bpb 比 baseline 差）。所以这是"理论合理但未验证"的建议。

#### 5. loop_scales 补偿不足

plan1 给 shared recurrence 的"去同质化"手段，只有每个 loop、每个 block 一个逐通道乘法 `loop_scales`。这能调幅，但很难产生真正的层间功能分化。

相关 PR #38 加了 **per-iteration LoRA deltas**，PR #298 堆了 **per-pass control params、U-Net skip、adaptive depth** 等一整套补偿机制，但它给出的 `val_bpb` 仍然只有 1.2271，略差于 README baseline 1.2244。

**结论**：**在这个 challenge 里，depth recurrence 目前不是"加上就赚"的成熟方向**，必须带一整套补偿件，且即便如此也还没证明能赢。

---

### plan1 教训总结

1. **不要把"缩 raw params"当第一目标** - 赛题目标是低 `val_bpb`，体积靠量化/压缩解决
2. **不要一次性同时改多个轴** - 改得越大，越难定位问题
3. **不要全程开 QAT** - 如果要开，做 late-QAT，且必须验证真的跑到了
4. **如果做共享权重，要重新配平 LR** - 至少考虑 shared params 被多次使用的影响
5. **recurrence 方向目前未验证能赢** - 即使带补偿件，也没超过 baseline

---

## plan2 失败分析

### 问题本质

**plan2 变差的主因不是 Muon WD / SWA 这条路错了，而是没有在可比 regime 下验证它。**

### 具体原因

plan2 同时改了太多变量：
- train_batch_tokens: 524288 → 65536 → 32768
- train_seq_len: 1024 → 256 → 128
- 学习率仍然是 matrix_lr=0.04 / scalar_lr=0.04

**这不是在验证"Muon WD / SWA / fp16 emb"，而是在测 short-context 新 regime。**

### 无法归因

plan2 的日志不能拿来得出"WD/SWA 没用"的结论，因为同时改了优化器、上下文长度、batch、warmup/warmdown 形态。

### 学习率误用

0.02 不是"普适更好"的学习率：
- 公开记录里，在 9L 路线上，早期有效的是 warmdown 变长 + MATRIX_LR=0.06
- 0.02 是 3x MLP / WD=0.04 / 更深模型那一整套 bundle 的一部分
- 不能把"10L+3x+WD 路线里的 LR"直接拿来当所有路线的最优 LR

### plan2 教训总结

1. **不要把不同 seq_len 的日志拿来直接比较** - 1024 和 128/256 测到的是不同任务代理
2. **不要把 LR=0.02 当成普适真理** - 它是 WD + 大模型 bundle 里的值
3. **不要同时改太多变量** - 要按 bundle 搬，逐步验证

---

## 公开经验总结

### 已验证的方向

| 方向 | 验证状态 | 备注 |
|------|----------|------|
| depth recurrence/looping | 未验证能赢 | PR #38, #298 都比 baseline 差 |
| 全程 QAT | 不推荐 | 收益抵不过开销，容易被 torch.compile 优化掉 |
| late QAT | 小收益 | 1.1233 阈值从 0.1 调到 0.15，只有 -0.0001 BPB |
| int5 MLP + 第 10 层 | 已验证 | 一起出现、一起成立，int5 节省的空间资助第 10 层 |
| U-Net skip | 有争议 | 有些强 PR 保留，有些不用 |

### 公开 SOTA 路线

| val_bpb | 架构 | 关键技术 |
|---------|------|----------|
| 1.1502 | 11L MLP3x | int6 QAT, WD=0.04, zstd-22 |
| 1.1271 | 11L | EMA, XSA |
| 1.1248 | 11L | Partial RoPE, LN Scale |
| 1.1233 | 11L | EMA, GPTQ-lite |
| 1.1428 | 10L | Int5MLP, BigramHash(10240), SWA(frac=0.4), WD=0.04 |

**趋势**：加深 + 更宽 MLP + 更低 LR + decoupled WD + EMA + smarter quantization

---

## 推荐改进路径

### 总原则

**"保容量、保 skip、轻量共享、晚开 QAT、先赚后处理"**

### plan3a：低风险、强确定性改动（保持 baseline 架构不变）

**保持不变**：
- 9 个独立 block
- U-Net skip
- seq_len=1024
- train_batch_tokens=8192（本地环境）

**只加**：
- **FP16 tied embedding export**：早期 PR 明确说 embedding 是最敏感张量，把 post-quant degradation 从约 0.007 压到约 0.0005
- **更长 warmdown**：1200 → 3000
- **不做 recurrence、不做全程 QAT、不改 short-context**

---

### plan3b：WD bundle（必须成套带）

**必须一起带的超参**：
- matrix_lr / scalar_lr 下到 0.02 左右
- tied_embed_lr 下到 0.03 左右
- Muon momentum 提到 0.99
- warmup 从 0.92 → 0.99 拉长到 1500 左右
- grad_clip=0.3

**注意**：单独加 WD 而不降 LR，很容易把训练推得过猛。

---

### plan3c：架构增益（按顺序）

1. **3x MLP**：公开记录里被作者直接标成最大贡献
2. **SmearGate**：参数极少，基本属于该拿的便宜增益
3. **BigramHash**：
   - 先从 2048 或 4096 起（本地小卡验证更稳）
   - 10240 是后期 squeeze 最后几点万分位的活
4. **OrthoInit**：和上面几个是高频共现组合

**暂不做**：第 10 层（等 mixed quant 稳定后再考虑）

---

### plan3d：量化 + 第 10 层

**在量化预算稳定之后**：
- mixed int6/int5
- 第 10 层
- late QAT / EMA / GPTQ-lite

---

## 六条避坑原则

1. **不要再提前为 size 付训练税**
   - baseline 已经能进 16MB，就别先上 recurrence / QAT 去"省空间"
   - 先把质量拉上去，再决定字节怎么换

2. **不要把不同 seq_len 的日志拿来直接比较**
   - 1024 和 128/256 测到的是不同任务代理
   - plan2 最大的问题就是这个

3. **不要把 LR=0.02 当成普适真理**
   - 它是 WD + 大模型 bundle 里的值，不是 baseline 路线的一般最优

4. **不要做全程 QAT**
   - 如果没有 late-stage gating、EMA/GPTQ-lite 之类一起配，通常不划算
   - 还要验证 QAT 真的跑到了（别被 torch.compile 优化掉）

5. **不要先上 10240 BigramHash**
   - 先用小桶把方向跑通
   - 大桶是后期 squeeze 最后几点万分位的活

6. **不要忽视 eval-only 增益**
   - 可以先不做 sliding eval，但最后和 leaderboard 对数时要记得它本身就是一条收益轴
   - 但 **sliding eval 是 leaderboard 技巧，不是训练曲线修复器**，别把主要精力放在它前面

---

## 本地环境约束

- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- 公开 record 基本都是 8×H100、seq_len=2048、batch≈786K tokens
- 直接一锅炖，会得到"改了很多、但不知道到底谁起作用"的结果
- **必须按 bundle 搬，逐步验证**

---

## 最稳下一步

1. 基于 baseline，先做 **fp16 tok_emb passthrough + 保持 9 个独立层 + 不改 seq_len/batch + warmdown 重调**
2. 然后再做 **Muon WD bundle (0.02 LR / 0.99 momentum / long warmup / clip=0.3)**
3. 然后再做 **SmearGate + BigramHash(2048 或 4096) + OrthoInit + 3x MLP**
4. 最后才是 **mixed int6/int5 + 10th layer + late QAT/EMA/GPTQ-lite**

这条路径比"一次性全搬 1.1428"慢一点，但更不容易重演 plan1 / plan2 那种"改得很大，结果更差，还说不清为什么"的情况。

---

## 如果坚持做 recurrence

**不要用 plan1 的方案（full-block sharing + 无 skip + always-on QAT）**，改成：

1. **部分共享**：只共享 V / value embedding / 某些 late-layer projection
2. **保留 U-Net skip**
3. **加 per-pass adapter**：小 LoRA / small delta / per-pass bias/scale
4. **shared 部分 LR 单独缩小**
5. **只做 late-QAT**

但请注意：**即使带补偿件，recurrence 方向目前也未验证能赢 baseline**。

---

## 后处理优先于训练期噪声

plan1 的主矛盾不是"导出后精度掉太多"，而是"训练出来的 float 模型就弱"。

所以更适合走：
- baseline/11L 主体不动
- 导出时做 **GPTQ-lite / percentile clip search / outlier sidecar** 这类零训练成本优化
- embedding 保持 fp16
- 再加 EMA/SWA

这是现在最符合优秀 PR 经验的路线。