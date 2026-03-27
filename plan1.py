"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.

中文说明：
- 这个脚本实现了一个尽量精简、但包含完整训练闭环的 GPT 训练基线。
- 代码结构从上到下依次是：超参数定义、Muon 优化器、与 tokenizer 解耦的验证指标、
  训练后量化导出、数据加载、Transformer 模块实现，以及训练/导出主流程。
- 它的目标不是追求最强配置，而是给参赛者或新读者一个"能完整看懂并能继续改"的出发点。
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap


class Hyperparameters:
    """集中管理训练脚本用到的全部超参数。

    设计上这里没有使用 dataclass，而是直接定义类属性，原因是：
    - 读取环境变量的写法足够直接，适合脚本型入口。
    - 后续通过 `args = Hyperparameters()` 访问时语义也很清晰。
    - 训练日志打印时更容易把"当前运行真正生效的默认值"集中到一处查看。

    大部分参数都允许通过环境变量覆盖，这样在不改脚本的情况下就能进行实验。
    """

    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
    )
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_shared_blocks = int(os.environ.get("NUM_SHARED_BLOCKS", 2))
    num_loops = int(os.environ.get("NUM_LOOPS", 4))
    num_layers = num_shared_blocks * num_loops  # effective depth for logging
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    oi_topk = int(os.environ.get("OI_TOPK", 16))
    enable_qat = bool(int(os.environ.get("ENABLE_QAT", "1")))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(
        os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85)
    )
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))


# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/


def zeropower_via_newtonschulz5(
    G: Tensor, steps: int = 10, eps: float = 1e-7
) -> Tensor:
    """用 Newton-Schulz 迭代近似正交化二维梯度矩阵，供 Muon 优化器使用。"""
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon 优化器的简化实现。

    这里的参数分组假定调用方已经把"适合 Muon 的矩阵参数"筛选好了。对于每个参数，
    优化器会：
    1. 维护动量缓冲区；
    2. 计算 Nesterov 形式的梯度；
    3. 对二维梯度做 Newton-Schulz 正交化；
    4. 在分布式场景下把各 rank 负责的更新拼平后汇总；
    5. 再把更新切回原参数形状并执行一步更新。
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
    ):
        """初始化 Muon。

        Args:
            params: 要优化的参数迭代器，通常只包含二维矩阵参数。
            lr: 学习率。
            momentum: 动量系数。
            backend_steps: Newton-Schulz 后端迭代次数。
            nesterov: 是否启用 Nesterov 动量。
        """
        super().__init__(
            params,
            dict(
                lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步参数更新。

        Returns:
            loss: 如果提供了 closure，则返回 closure 的结果；否则返回 None。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(
                total_params, device=params[0].device, dtype=torch.bfloat16
            )

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """为 tokenizer 无关的 BPB 评估构造查找表。

    训练损失是"每 token 的交叉熵"，但挑战真正关心的是"每字节压缩效果"。
    为了把 token 级损失换算到 byte 级，我们需要知道每个 token 对应多少 UTF-8 字节，
    以及它是否隐含前导空格、是否属于边界 token。

    Args:
        sp: 已加载好的 SentencePiece tokenizer。
        vocab_size: 模型侧认为的词表大小。
        device: 查找表要放置到的设备。

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - base_bytes_lut: 每个 token 本体对应的字节数。
            - has_leading_space_lut: token 是否包含隐式前导空格。
            - is_boundary_token_lut: token 是否可视为边界/特殊 token。
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    """读取并拼接验证集 token。

    这里会把匹配到的所有验证 shard 顺序拼起来，然后裁到刚好适合按 `seq_len`
    切成整批序列的长度。多保留的最后 1 个 token 用来在后面构造 `(x, y)` 位移对。

    Args:
        pattern: 验证集 shard 的 glob 模式。
        seq_len: 训练/验证使用的序列长度。

    Returns:
        Tensor: 一维连续 token 张量，长度满足"可按 seq_len 切分并额外带 1 个目标 token"。
    """
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """在完整验证集上评估模型。

    这个函数会返回两个指标：
    - `val_loss`: 常规 token 级交叉熵；
    - `val_bpb`: bits-per-byte，挑战更关心的 tokenizer 无关压缩指标。

    评估流程是先按 rank 切分验证序列范围，分别计算本地 loss、token 数与 byte 数，
    最后在分布式场景下做 all-reduce 汇总。

    Args:
        args: 运行超参数。
        model: 待评估模型，可能是原模型，也可能是 DDP 包装后的模型。
        rank: 当前分布式进程编号。
        world_size: 总进程数。
        device: 当前设备。
        grad_accum_steps: 梯度累积步数。这里用于反推出每个 rank 的本地评估 batch 大小。
        val_tokens: 已预加载好的完整验证 token。
        base_bytes_lut: token 到基础字节数的查找表。
        has_leading_space_lut: token 是否自带前导空格的查找表。
        is_boundary_token_lut: token 是否为边界 token 的查找表。

    Returns:
        tuple[float, float]:
            - 第一个值是验证集平均交叉熵；
            - 第二个值是验证集 BPB。
    """
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            # prev_ids/tgt_ids 用来恢复"前一个 token + 当前 token"的局部上下文，
            # 从而判断当前 token 的前导空格是否应计入本次字节数。
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,loop_scale,loop_scales",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    """返回张量占用的原始字节数。"""
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(
    name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]
) -> Tensor:
    """决定某个浮点张量是否直接保留浮点格式而不量化。

    小张量或控制张量往往对数值精度更敏感，但自身占空间很小，因此直接保留更划算。

    Args:
        name: 参数名，用于按模式匹配控制张量。
        t: 待处理张量。
        passthrough_orig_dtypes: 记录原始 dtype，便于反量化后恢复。

    Returns:
        Tensor: 直接保留的张量，必要时会下转到更省空间的存储 dtype。
    """
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    """把浮点张量量化成 int8 载荷和对应 scale。

    - 对二维矩阵使用逐行 scale，因为每一行常常对应一个输出通道，动态范围差异大；
    - 对向量/标量使用整张量一个 scale，逻辑更简单且足够省空间。

    Args:
        t: 浮点张量。

    Returns:
        tuple[Tensor, Tensor]:
            - q: int8 量化结果；
            - scale: 用于反量化的尺度张量。
    """
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = (
            torch.clamp(torch.round(clipped / scale[:, None]), -127, 127)
            .to(torch.int8)
            .contiguous()
        )
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item())
        if t32.numel()
        else 0.0
    )
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = (
        torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127
        )
        .to(torch.int8)
        .contiguous()
    )
    return q, scale


class _FakeQuantSTE(torch.autograd.Function):
    """STE fake-quant: forward 做 per-row int8 round-trip，backward 直通。"""

    @staticmethod
    def forward(ctx, t):
        if t.ndim != 2:
            return t
        t32 = t.float()
        amax = t32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 127.0
        return (torch.clamp(torch.round(t32 / scale), -127, 127) * scale).to(t.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad


fake_quantize_ste = _FakeQuantSTE.apply


def isolate_outlier_cols(
    t: Tensor, topk: int, reuse_factor: float = 1.0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """把 top-k 异常列分离为 fp16 sidecar，其余走 per-row int8。"""
    q0, s0 = quantize_float_tensor(t)
    s0f = s0.float()
    if s0f.ndim > 0:
        w0 = (q0.float() * s0f.view(t.shape[0], *([1] * (q0.ndim - 1)))).to(t.dtype)
    else:
        w0 = (q0.float() * s0f.item()).to(t.dtype)
    residual = (t.float() - w0.float()).abs()
    col_score = residual.mean(dim=0) * reuse_factor
    k = min(topk, t.shape[1])
    _, idx = col_score.topk(k)
    idx = idx.sort().values
    sidecar = t[:, idx].to(torch.float16).contiguous()
    bulk = t.clone()
    bulk[:, idx] = 0.0
    q_bulk, s_bulk = quantize_float_tensor(bulk)
    return q_bulk, s_bulk, idx.to(torch.int16).contiguous(), sidecar


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    """把模型 state_dict 转成统一的 int8 导出格式。

    导出策略如下：
    - 大浮点矩阵：量化为 int8；
    - 小浮点张量/控制张量：直接保留浮点，减少精度损失；
    - 非浮点张量：原样透传；
    - 同时记录必要元数据，保证后续能无损恢复结构、近似恢复数值。

    Args:
        state_dict: 原始模型参数字典。

    Returns:
        tuple[dict[str, object], dict[str, int]]:
            - 第一个值是可序列化的量化对象；
            - 第二个值是量化过程统计信息，便于日志里分析压缩收益。
    """
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    outlier_idx: dict[str, Tensor] = {}
    outlier_vals: dict[str, Tensor] = {}
    oi_topk = Hyperparameters.oi_topk
    num_loops = Hyperparameters.num_loops
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        is_shared_matrix = (
            t.ndim == 2
            and "shared_blocks" in name
            and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        )
        if is_shared_matrix and oi_topk > 0:
            q, s, oi_i, oi_v = isolate_outlier_cols(
                t, oi_topk, reuse_factor=float(num_loops)
            )
            qmeta[name] = {"scheme": "per_row_int8_outlier_cols_v1", "axis": 1}
            outlier_idx[name] = oi_i
            outlier_vals[name] = oi_v
            stats["int8_payload_bytes"] += (
                tensor_nbytes(q)
                + tensor_nbytes(s)
                + tensor_nbytes(oi_i)
                + tensor_nbytes(oi_v)
            )
        else:
            q, s = quantize_float_tensor(t)
            if s.ndim > 0:
                qmeta[name] = {"scheme": "per_row", "axis": 0}
            stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_oi_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if outlier_idx:
        obj["outlier_idx"] = outlier_idx
        obj["outlier_vals"] = outlier_vals
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    """把 int8 导出对象恢复成普通 state_dict。

    Args:
        obj: 由 `quantize_state_dict_int8` 生成的量化对象。

    Returns:
        dict[str, Tensor]: 反量化后的参数字典，可直接用于 `load_state_dict`。
    """
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    outlier_idx = obj.get("outlier_idx", {})
    outlier_vals = obj.get("outlier_vals", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        scheme = qmeta.get(name, {}).get("scheme", "")
        if scheme in ("per_row_int8_outlier_cols_v1", "per_row") or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            w = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype)
            if name in outlier_idx:
                idx = outlier_idx[name].long()
                w[:, idx] += outlier_vals[name].to(dtype=dtype)
            out[name] = w.contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING
# -----------------------------


def load_data_shard(file: Path) -> Tensor:
    """从单个二进制 shard 读取 token。

    这个仓库的数据预处理流程会在 shard 开头写一个固定长度的 header，
    包含魔数、版本和 token 数。读取时先校验 header，再按 header 描述的长度
    读取真实 token，避免静默读到损坏数据。

    Args:
        file: shard 文件路径。

    Returns:
        Tensor: 一维 `uint16` token 张量。
    """
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(
            f"Shard size mismatch for {file}: expected {expected_size} bytes"
        )
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    """顺序、循环地从 shard 流中取 token。

    这个类故意保持极简：不打乱、不多进程、不做复杂预取，只按文件顺序一直往前读。
    好处是训练样本流动方式非常透明，出了问题也更容易复现和定位。
    """

    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        """初始化 token 流。"""
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        """切到下一个 shard，读到末尾后会自动回绕。"""
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        """连续取出 n 个 token，必要时跨 shard 拼接。

        Args:
            n: 需要取出的 token 数。

        Returns:
            Tensor: 长度为 n 的一维 token 张量。
        """
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    """把共享 token 流切成各 rank 的本地批次。

    这里不是"每个 rank 各自随机采样"，而是先从同一条顺序 token 流中切出一大块，
    再按 rank 切成互不重叠的子区间。这样不同进程看到的数据不会重复，同时整体顺序
    仍然是可追踪、可复现的。
    """

    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(
        self, global_tokens: int, seq_len: int, grad_accum_steps: int
    ) -> tuple[Tensor, Tensor]:
        """生成当前 rank 的下一批 `(x, y)` 训练样本。

        Args:
            global_tokens: 全局总 token batch 大小，所有 rank 与所有 micro-step 共享。
            seq_len: 单条训练序列长度。
            grad_accum_steps: 梯度累积步数，用于拆分本地 micro-batch 大小。

        Returns:
            tuple[Tensor, Tensor]:
                - x: 输入 token，形状约为 `[batch, seq_len]`
                - y: 目标 token，为 x 向右平移一位后的结果
        """
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


# -----------------------------
# TRANSFORMER MODULES
# -----------------------------


class RMSNorm(nn.Module):
    """RMSNorm 的极简封装。

    这里直接委托给 `torch.nn.functional.rms_norm`，主要是为了让调用侧写法统一，
    同时保留一个可插拔的模块边界。
    """

    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """对输入最后一维执行 RMSNorm。"""
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """线性层的"存 fp32、算低精度"封装。

    权重在参数存储和优化器状态里保持更高精度，有利于训练稳定性；真正做矩阵乘法时，
    再把权重 cast 到输入张量 dtype，以利用 bf16 计算吞吐。
    """

    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        """执行线性变换，并让计算 dtype 跟随输入。"""
        w = self.weight.to(x.dtype)
        if getattr(self, "_fake_quant", False) and self.training:
            w = fake_quantize_ste(w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    """把低维参数和控制参数恢复为 fp32。

    训练主体张量可以用 bf16 跑得更快，但很多标量、向量或门控系数参数更容易受到
    精度影响，因此这里强制把它们保留为 fp32。
    """
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (
                param.ndim < 2
                or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            ) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    """RoPE 位置编码缓存器。

    它不会直接改写输入，而是根据 `seq_len` 生成并缓存对应的 cos/sin 表，
    供注意力层中的 query/key 旋转使用。
    """

    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """返回给定序列长度下的 RoPE cos/sin 缓存。"""
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """把 RoPE 旋转应用到最后一维的偶/奇半区。

    Args:
        x: 待旋转张量，最后一维必须能均分为两半。
        cos: 余弦缓存。
        sin: 正弦缓存。

    Returns:
        Tensor: 应用旋转后、形状不变的张量。
    """
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """带 GQA 和 RoPE 的因果自注意力模块。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        """初始化注意力层的投影、RoPE 和可学习 q 增益。"""
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(
            torch.full((num_heads,), qk_gain_init, dtype=torch.float32)
        )
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        """执行一次因果自注意力前向传播。

        关键步骤依次是：
        1. 线性投影得到 q/k/v；
        2. 对 q/k 先做 RMSNorm，再加 RoPE；
        3. 给 q 乘上每头单独的可学习增益；
        4. 调用 PyTorch 的 fused `scaled_dot_product_attention`；
        5. 把多头结果拼回模型维度并投影输出。
        """
        bsz, seqlen, dim = x.shape
        q = (
            self.c_q(x)
            .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.c_k(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.c_v(x)
            .reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    """使用 `relu(x)^2` 激活的前馈网络。"""

    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        """先升维，再做 `relu^2`，最后投影回残差维度。"""
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    """Transformer block，包含注意力、MLP 与若干可学习残差控制项。"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        """初始化一个 block 的规范化层、子模块和残差缩放参数。"""
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init
        )
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        """执行一个 block 的前向传播。

        Args:
            x: 当前层输入。
            x0: 最初的 token embedding 表示，用于跨层残差混合。

        Returns:
            Tensor: 经过注意力和 MLP 更新后的隐藏状态。
        """
        # resid_mix 让当前层表示与初始表示之间可以做逐通道混合，
        # 类似一种轻量的跨层信息回注机制。
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x)
        )
        return x


class GPT(nn.Module):
    """Depth-recurrent GPT：少数共享 block 循环展开，配合 per-loop scale adapter。"""

    def __init__(
        self,
        vocab_size: int,
        num_shared_blocks: int,
        num_loops: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        """构建完整模型并初始化权重。"""
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_shared_blocks = num_shared_blocks
        self.num_loops = num_loops
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.shared_blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for _ in range(num_shared_blocks)
            ]
        )
        self.loop_scales = nn.Parameter(
            torch.ones(num_loops, num_shared_blocks, model_dim, dtype=torch.float32)
        )
        self.final_norm = RMSNorm()
        self.lm_head = (
            None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        """初始化嵌入层以及标记为 `_zero_init` 的线性层。"""
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        """计算语言模型交叉熵损失。"""
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        for loop in range(self.num_loops):
            for bi, block in enumerate(self.shared_blocks):
                x = block(x, x0)
                x = x * self.loop_scales[loop, bi].to(dtype=x.dtype)[None, None, :]

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


# -----------------------------
# TRAINING
# -----------------------------


def main() -> None:
    """训练脚本主入口。

    主流程可以概括为：
    1. 读取超参数与当前脚本源码；
    2. 初始化分布式/CUDA 环境；
    3. 加载 tokenizer 与验证集；
    4. 构建模型、优化器与数据加载器；
    5. 执行 warmup 和正式训练；
    6. 导出原始模型与量化模型；
    7. 对量化模型做反序列化回灌并再次评估，确认导出结果可用。
    """
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if sys.platform != "win32":
        zeropower_via_newtonschulz5 = (
            torch.compile(zeropower_via_newtonschulz5)
            if sys.platform != "win32"
            else zeropower_via_newtonschulz5
        )

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(
            f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral"
        )
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_cudnn_sdp,
        enable_flash_sdp,
        enable_math_sdp,
        enable_mem_efficient_sdp,
    )

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        """只在主进程打印/写日志，避免多卡输出相互打架。"""
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    # Workaround for GQA on Windows
    if sys.platform == "win32":
        enable_flash_sdp(False)
        enable_math_sdp(True)
        log0("GQA workaround: switched to math SDPA backend on Windows")

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Script only setup for SentencePiece .model file: {args.tokenizer_path}"
        )
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
        build_sentencepiece_luts(sp, args.vocab_size, device)
    )
    log0(
        f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}"
    )
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = (
        GPT(
            vocab_size=args.vocab_size,
            num_shared_blocks=args.num_shared_blocks,
            num_loops=args.num_loops,
            model_dim=args.model_dim,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings,
            tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap,
            rope_base=args.rope_base,
            qk_gain_init=args.qk_gain_init,
        )
        .to(device)
        .bfloat16()
    )
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    if args.enable_qat:
        for m in base_model.shared_blocks.modules():
            if isinstance(m, CastedLinear):
                m._fake_quant = True
    compiled_model = (
        torch.compile(base_model, dynamic=False, fullgraph=True)
        if sys.platform != "win32"
        else base_model
    )
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False)
        if distributed
        else compiled_model
    )

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.shared_blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2
        or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params.append(base_model.loop_scales)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [
        optimizer_tok,
        optimizer_muon,
        optimizer_scalar,
    ]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [
                {
                    "params": [base_model.lm_head.weight],
                    "lr": args.head_lr,
                    "base_lr": args.head_lr,
                }
            ],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(
        f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}"
    )
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        """把所有优化器的梯度清零。"""
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = (
        1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    )

    def lr_mul(step: int, elapsed_ms: float) -> float:
        """根据迭代进度或剩余墙钟时间返回学习率缩放系数。

        这里的 warmdown 有两种模式：
        - 如果没有时间上限，就按迭代数做线性收尾；
        - 如果设置了墙钟时间上限，就按"剩余时间是否进入 warmdown 窗口"来收尾。
        """
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return (
                max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < args.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return (
            remaining_ms / max(warmdown_ms, 1e-9)
            if remaining_ms <= warmdown_ms
            else 1.0
        )

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        # warmup 的目的不是"从更差的初始化开始训练"，而是提前把
        # torch.compile、前向、反向和优化器路径都编译/预热好。
        # 所以 warmup 完成后会把模型与优化器状态恢复到真正初始值。
        initial_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        }
        initial_optimizer_states = [
            copy.deepcopy(opt.state_dict()) for opt in optimizers
        ]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1
                    )
                x, y = train_loader.next_batch(
                    args.train_batch_tokens, args.train_seq_len, grad_accum_steps
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if (
                args.warmup_steps <= 20
                or (warmup_step + 1) % 10 == 0
                or warmup_step + 1 == args.warmup_steps
            ):
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        # 这里把"到达最终 step"和"因墙钟时间提前停止"统一折叠为 last_step，
        # 这样验证、日志与收尾逻辑都能复用同一条控制流。
        last_step = step == args.iterations or (
            stop_after_step is not None and step >= stop_after_step
        )

        should_validate = last_step or (
            args.val_loss_every > 0 and step > 0 and step % args.val_loss_every == 0
        )
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(
                args.train_batch_tokens, args.train_seq_len, grad_accum_steps
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = (
            min(step / args.muon_momentum_warmup_steps, 1.0)
            if args.muon_momentum_warmup_steps > 0
            else 1.0
        )
        muon_momentum = (
            1 - frac
        ) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        # Muon 的动量也有单独 warmup，避免训练最早期更新过于激进。
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        grad_norm_val = torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=float('inf')).item()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = args.train_log_every > 0 and (
            step <= 10
            or step % args.train_log_every == 0
            or stop_after_step is not None
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"lr_scale:{scale:.4f} grad_norm:{grad_norm_val:.4f} tokens:{step * args.train_batch_tokens} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = (
            max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        )
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(
            quant_stats["int8_payload_bytes"], 1
        )
        log0(
            f"Serialized model int8+zlib: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    # 无论训练阶段模型用的是原始权重还是 DDP 包装，这里都强制从磁盘重新读回量化产物，
    # 验证"真正提交出去的文件"能否解压、反量化并恢复到可评估状态。
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(
        io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu"
    )
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(
        f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}"
    )
    log0(f"quant_gap:{q_val_bpb - val_bpb:.6f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
