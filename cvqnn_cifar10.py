"""
================================================================================
 CVQNN - Complex-Valued Quantized Neural Network
 Weights hard-constrained to the 4th roots of unity: {+1, -1, +i, -i}
================================================================================

 The idea:
   Every weight is a unit-modulus complex number sitting at one of the four
   corners of the "phase square", i.e. exactly 2 bits per weight (log2 4).
   Unlike binary {+1,-1} networks, this buys an extra degree of freedom:
   multiplying by i is a 90-degree rotation that MIXES the real and imaginary
   channels of the signal. The hypothesis is that this phase coupling partly
   compensates for the loss of weight precision.

   Training runs on latent FP32 copies (w_real, w_imag); the forward pass
   projects them onto the nearest corner and the gradient passes straight
   through (STE).

 Architecture: a compact complex-valued CIFAR-style ResNet.
   ResNet rather than ViT: trained from scratch on 50k images it is far more
   stable, needs no long warmup or heavy augmentation, and tolerates
   aggressive quantization much better - the skip connection gives the
   gradient a clean path around the quantized layers.

 To run: paste into a Kaggle notebook (Accelerator: GPU T4) and Run All.
 Dependencies: torch, torchvision, numpy, matplotlib - all preinstalled there.
================================================================================
"""

import os
import math
import time
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


# ==============================================================================
# 0. CONFIG
# ==============================================================================
class CFG:
    # --- reproducibility / hardware ---
    seed            = 1337
    device          = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers     = 2               # more than 2-4 buys nothing on Kaggle
    use_amp         = True            # fp16 autocast; critical parts stay fp32

    # --- data ---
    data_root       = "./data"
    batch_size      = 128
    val_batch_size  = 512

    # --- model ---
    widths          = (48, 96, 192)   # channels per stage
    blocks          = (2, 2, 2)       # residual blocks per stage
    num_classes     = 10

    # --- QUANTIZATION SWITCH (the main experimental knob) ---
    # True  -> weights projected onto {+1,-1,+i,-i}: the hypothesis under test
    # False -> plain complex FP32 network: the control arm of the ablation
    quantize        = True
    quantize_stem   = True            # quantize the first conv (BNN practice: keep it FP)
    quantize_head   = True            # quantize the final linear
    per_channel_scale = True          # scale as a per-output-channel vector (False -> scalar)
    weight_clip     = 1.0             # clip latent weights after each step; None disables

    # --- training ---
    epochs          = 40
    lr              = 2e-3
    weight_decay    = 5e-2            # applied ONLY to non-quantized parameters
    label_smoothing = 0.1
    grad_clip       = 5.0
    log_every       = 100             # steps between intermediate log lines

    # --- what to run ---
    # "both"  - quantized network + FP32 control, plus the comparison between
    #           them. The delta between the two runs is what actually answers
    #           the research question, so this is the default.
    # "quant" / "fp32" - a single arm only.
    # Overridable via the CVQNN_MODE environment variable; on Kaggle env vars
    # cannot be set, so editing this line is the primary way to configure it.
    mode            = "both"

    # --- output ---
    out_dir         = "./cvqnn_out"


def check_gpu_compat(cfg=CFG):
    """
    Fail fast if the installed PyTorch build has no kernels for this GPU.

    Otherwise the very first CUDA op dies with cudaErrorNoKernelImageForDevice
    somewhere deep inside the network, and the traceback points at our code
    rather than at the environment. Real case: Kaggle hands out a Tesla P100
    (sm_60) while its preinstalled torch is built for sm_70+ - Pascal support
    was dropped from the wheels.
    """
    if cfg.device != "cuda":
        return
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    name = torch.cuda.get_device_name(0)

    if sm not in arch_list:
        raise RuntimeError(
            f"\n{'!' * 70}\n"
            f"GPU {name} has compute capability {sm}, but the installed\n"
            f"PyTorch {torch.__version__} was built only for: {' '.join(arch_list)}.\n"
            f"Every CUDA op would fail with cudaErrorNoKernelImageForDevice.\n\n"
            f"Fix: request a different accelerator (on Kaggle use T4 instead of\n"
            f"P100 via machine_shape in kernel-metadata.json), install a torch\n"
            f"build targeting {sm}, or fall back to CPU (CFG.device = 'cpu').\n"
            f"{'!' * 70}"
        )
    print(f"[gpu] {name} ({sm}) is compatible with torch {torch.__version__}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# 1. QUANTIZER: PhaseQuant + Straight-Through Estimator
# ==============================================================================
class PhaseQuantSTE(torch.autograd.Function):
    """
    Forward: the complex weight W = w_real + i*w_imag is projected onto the
             nearest corner of the phase square {+1, -1, +i, -i}.

             The rule is "larger magnitude wins":
               |Re| >= |Im|  ->  q = sign(Re) * 1     (real axis)
               |Re| <  |Im|  ->  q = sign(Im) * i     (imaginary axis)

             Geometrically this really is the nearest corner: the decision
             boundaries are the diagonals Re = +-Im, which are exactly the
             bisectors between adjacent corners of the square.

    Backward: plain STE - the incoming gradient is passed to the latent FP32
              copies untouched. The projection is piecewise constant, so its
              true derivative is zero almost everywhere; we substitute the
              identity instead.
    """

    @staticmethod
    def forward(ctx, w_real, w_imag):
        # sign with the convention sign(0) = +1, so we never mint dead zero weights
        ones = torch.ones_like(w_real)
        s_re = torch.where(w_real >= 0, ones, -ones)
        s_im = torch.where(w_imag >= 0, ones, -ones)

        real_dominant = w_real.abs() >= w_imag.abs()
        zeros = torch.zeros_like(w_real)

        q_re = torch.where(real_dominant, s_re, zeros)   # {+1,-1, 0, 0}
        q_im = torch.where(real_dominant, zeros, s_im)   # { 0, 0,+1,-1}
        return q_re, q_im

    @staticmethod
    def backward(ctx, g_re, g_im):
        # Straight-through: pass the gradient along unchanged.
        return g_re, g_im


def phase_quantize(w_real, w_imag):
    return PhaseQuantSTE.apply(w_real, w_imag)


@torch.no_grad()
def phase_code(w_real, w_imag):
    """Integer corner id: 0:+1, 1:-1, 2:+i, 3:-i. Used for diagnostics only."""
    real_dominant = w_real.abs() >= w_imag.abs()
    z = torch.zeros_like(w_real, dtype=torch.int8)
    code = torch.where(
        real_dominant,
        torch.where(w_real >= 0, z + 0, z + 1),
        torch.where(w_imag >= 0, z + 2, z + 3),
    )
    return code


# ==============================================================================
# 2. COMPLEX ALGEBRA FOR THE LAYERS
# ==============================================================================
def complex_op(op, x_re, x_im, w_re, w_im, cat_dim, chunk_dim):
    """
    Apply a linear operator (linear / conv2d) with complex arithmetic:

        Out_real = X_real * W_real - X_imag * W_imag
        Out_imag = X_real * W_imag + X_imag * W_real

    Naively that is 4 calls to op(). We concatenate [W_real; W_imag] along the
    output dimension and issue 2 calls with twice the outputs instead - same
    FLOPs, half the kernel launches, which matters for small convolutions.
    """
    w = torch.cat([w_re, w_im], dim=cat_dim)          # (2*out, in, ...)

    a = op(x_re, w)                                    # -> [X_re*W_re , X_re*W_im]
    b = op(x_im, w)                                    # -> [X_im*W_re , X_im*W_im]

    a_rr, a_ri = a.chunk(2, dim=chunk_dim)
    b_ir, b_ii = b.chunk(2, dim=chunk_dim)

    out_re = a_rr - b_ii
    out_im = a_ri + b_ir
    return out_re, out_im


class _ComplexQuantBase(nn.Module):
    """Shared plumbing: latent weights, quantization, learnable scale."""

    def __init__(self, weight_shape, out_features, fan_in, quantize=True,
                 per_channel_scale=True):
        super().__init__()
        self.quantize = quantize
        self.fan_in = fan_in

        # Latent FP32 copies, initialised ~ N(0, 1/sqrt(fan_in)). The projection
        # itself only cares about signs and relative magnitudes, but this scale
        # sits comfortably inside the [-1, 1] clipping range.
        std = 1.0 / math.sqrt(fan_in)
        self.w_real = nn.Parameter(torch.randn(weight_shape) * std)
        self.w_imag = nn.Parameter(torch.randn(weight_shape) * std)

        # Learnable REAL scale: quantized weights have unit modulus, so output
        # variance grows with fan_in. The scale pulls the signal back into a
        # sane range (the analogue of alpha in XNOR-Net).
        n_scale = out_features if per_channel_scale else 1
        self.scale = nn.Parameter(torch.full((n_scale,), 1.0 / math.sqrt(fan_in)))

    def effective_weights(self):
        if self.quantize:
            return phase_quantize(self.w_real, self.w_imag)
        return self.w_real, self.w_imag

    @torch.no_grad()
    def clip_latent_(self, limit):
        if self.quantize and limit is not None:
            self.w_real.clamp_(-limit, limit)
            self.w_imag.clamp_(-limit, limit)


class ComplexQuantLinear(_ComplexQuantBase):
    def __init__(self, in_features, out_features, quantize=True, per_channel_scale=True):
        super().__init__(
            weight_shape=(out_features, in_features),
            out_features=out_features,
            fan_in=in_features,
            quantize=quantize,
            per_channel_scale=per_channel_scale,
        )

    def forward(self, x_re, x_im):
        w_re, w_im = self.effective_weights()
        out_re, out_im = complex_op(F.linear, x_re, x_im, w_re, w_im,
                                    cat_dim=0, chunk_dim=-1)
        s = self.scale.to(out_re.dtype)
        return out_re * s, out_im * s


class ComplexQuantConv2d(_ComplexQuantBase):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 quantize=True, per_channel_scale=True):
        super().__init__(
            weight_shape=(out_ch, in_ch, kernel_size, kernel_size),
            out_features=out_ch,
            fan_in=in_ch * kernel_size * kernel_size,
            quantize=quantize,
            per_channel_scale=per_channel_scale,
        )
        self.stride = stride
        self.padding = padding

    def forward(self, x_re, x_im):
        w_re, w_im = self.effective_weights()
        op = lambda x, w: F.conv2d(x, w, stride=self.stride, padding=self.padding)
        out_re, out_im = complex_op(op, x_re, x_im, w_re, w_im,
                                    cat_dim=0, chunk_dim=1)
        s = self.scale.to(out_re.dtype).view(1, -1, 1, 1)
        return out_re * s, out_im * s


# ==============================================================================
# 3. ACTIVATION AND NORMALIZATION
# ==============================================================================
class ComplexSplitReLU(nn.Module):
    """Component-wise ReLU: relu(Re) + i*relu(Im)."""

    def forward(self, x_re, x_im):
        return F.relu(x_re), F.relu(x_im)


class ComplexAmpNorm(nn.Module):
    """
    Amplitude normalization: both components are divided by THE SAME mean
    modulus of the signal (over batch and space, per channel).

        amp   = sqrt(Re^2 + Im^2 + eps)
        m_c   = mean_{N,H,W} amp
        Re,Im = Re/m_c, Im/m_c

    The key property is that dividing by a shared modulus PRESERVES PHASE - we
    only change the length of the vector, never rotate it. A plain BatchNorm
    applied to Re and Im separately would destroy the phase.

    Running statistics are tracked BatchNorm-style so that eval is deterministic.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.register_buffer("running_amp", torch.ones(num_features))
        if affine:
            # gamma is shared by Re and Im: a phase-preserving rescale
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta_re = nn.Parameter(torch.zeros(num_features))
            self.beta_im = nn.Parameter(torch.zeros(num_features))

    def forward(self, x_re, x_im):
        dims = [0] + list(range(2, x_re.dim()))          # everything but the channel axis
        shape = [1, -1] + [1] * (x_re.dim() - 2)

        if self.training:
            # statistics in fp32: under fp16 autocast this sqrt easily yields inf/0
            amp = torch.sqrt(x_re.float() ** 2 + x_im.float() ** 2 + self.eps)
            m = amp.mean(dim=dims)
            with torch.no_grad():
                self.running_amp.mul_(1 - self.momentum).add_(self.momentum * m.detach())
        else:
            m = self.running_amp

        m = m.clamp_min(self.eps).view(shape).to(x_re.dtype)
        x_re = x_re / m
        x_im = x_im / m

        if self.affine:
            g = self.gamma.view(shape).to(x_re.dtype)
            x_re = x_re * g + self.beta_re.view(shape).to(x_re.dtype)
            x_im = x_im * g + self.beta_im.view(shape).to(x_im.dtype)
        return x_re, x_im


class ComplexSequential(nn.Sequential):
    """nn.Sequential for modules that take and return an (Re, Im) pair."""

    def forward(self, x_re, x_im):
        for module in self:
            x_re, x_im = module(x_re, x_im)
        return x_re, x_im


# ==============================================================================
# 4. MODEL: complex-valued ResNet
# ==============================================================================
class ComplexBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, cfg=CFG):
        super().__init__()
        self.conv1 = ComplexQuantConv2d(in_ch, out_ch, 3, stride, 1,
                                        cfg.quantize, cfg.per_channel_scale)
        self.norm1 = ComplexAmpNorm(out_ch)
        self.act = ComplexSplitReLU()
        self.conv2 = ComplexQuantConv2d(out_ch, out_ch, 3, 1, 1,
                                        cfg.quantize, cfg.per_channel_scale)
        self.norm2 = ComplexAmpNorm(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ComplexSequential(
                ComplexQuantConv2d(in_ch, out_ch, 1, stride, 0,
                                   cfg.quantize, cfg.per_channel_scale),
                ComplexAmpNorm(out_ch),
            )

    def forward(self, x_re, x_im):
        id_re, id_im = (x_re, x_im) if self.downsample is None \
            else self.downsample(x_re, x_im)

        r, i = self.conv1(x_re, x_im)
        r, i = self.norm1(r, i)
        r, i = self.act(r, i)
        r, i = self.conv2(r, i)
        r, i = self.norm2(r, i)

        # Complex residual: component-wise addition is addition in C
        return self.act(r + id_re, i + id_im)


class CVQResNet(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        w = cfg.widths
        b = cfg.blocks

        # --- Stem ---
        self.stem = ComplexSequential(
            ComplexQuantConv2d(3, w[0], 3, 1, 1,
                               cfg.quantize and cfg.quantize_stem,
                               cfg.per_channel_scale),
            ComplexAmpNorm(w[0]),
            ComplexSplitReLU(),
        )

        # --- Stages ---
        stages = []
        in_ch = w[0]
        for si, (out_ch, n_blocks) in enumerate(zip(w, b)):
            for bi in range(n_blocks):
                stride = 2 if (bi == 0 and si > 0) else 1
                stages.append(ComplexBasicBlock(in_ch, out_ch, stride, cfg))
                in_ch = out_ch
        self.stages = ComplexSequential(*stages)

        # --- Head ---
        self.head = ComplexQuantLinear(in_ch, cfg.num_classes,
                                       cfg.quantize and cfg.quantize_head,
                                       cfg.per_channel_scale)
        # The magnitude is non-negative with a narrow dynamic range, so a
        # learnable temperature plus a per-class shift give softmax room to work.
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.logit_bias = nn.Parameter(torch.zeros(cfg.num_classes))

    def forward(self, x):
        # INPUT: the real part is the normalized image, the imaginary part is zero.
        # All phase is born inside the network, from multiplications by +-i.
        x_re = x
        x_im = torch.zeros_like(x)

        x_re, x_im = self.stem(x_re, x_im)
        x_re, x_im = self.stages(x_re, x_im)

        # Complex global average pooling
        x_re = x_re.mean(dim=(2, 3))
        x_im = x_im.mean(dim=(2, 3))

        out_re, out_im = self.head(x_re, x_im)

        # OUTPUT: complex vector -> real logits via the magnitude |z|
        mag = torch.sqrt(out_re.float() ** 2 + out_im.float() ** 2 + 1e-8)
        return self.logit_scale * mag + self.logit_bias


# ==============================================================================
# 5. DATA (CIFAR-10, with an offline fallback for Kaggle without internet)
# ==============================================================================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def resolve_cifar_root(default_root):
    """Look for an already-downloaded CIFAR-10 (Kaggle datasets), else fetch it."""
    candidates = [default_root]
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for name in os.listdir(kaggle_input):
            candidates.append(os.path.join(kaggle_input, name))
    for root in candidates:
        if os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
            print(f"[data] found a local CIFAR-10 copy: {root}")
            return root, False
    print("[data] no local copy -> downloading (needs Internet: ON)")
    return default_root, True


def build_loaders(cfg=CFG):
    root, need_download = resolve_cifar_root(cfg.data_root)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(root, train=True,
                                             transform=train_tf, download=need_download)
    test_set = torchvision.datasets.CIFAR10(root, train=False,
                                            transform=test_tf, download=need_download)

    pin = cfg.device == "cuda"
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin,
                              drop_last=True, persistent_workers=cfg.num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=cfg.val_batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin,
                             persistent_workers=cfg.num_workers > 0)
    return train_loader, test_loader


# ==============================================================================
# 6. QUANTIZATION DIAGNOSTICS (the whole point of running the experiment)
# ==============================================================================
def quant_layers(model):
    return [m for m in model.modules()
            if isinstance(m, _ComplexQuantBase) and m.quantize]


@torch.no_grad()
def phase_histogram(model):
    """Fraction of weights landing on each of the 4 corners {+1, -1, +i, -i}."""
    counts = torch.zeros(4, dtype=torch.float64)
    for layer in quant_layers(model):
        code = phase_code(layer.w_real, layer.w_imag).flatten().to(torch.int64)
        counts += torch.bincount(code, minlength=4).double().cpu()
    total = counts.sum().item()
    return (counts / max(total, 1.0)).tolist(), int(total)


@torch.no_grad()
def snapshot_codes(model):
    return [phase_code(l.w_real, l.w_imag).cpu() for l in quant_layers(model)]


@torch.no_grad()
def flip_rate(model, prev_codes):
    """Fraction of weights that changed corner since the last snapshot.

    This is a direct measure of STE stability: if it does not decay towards the
    end of training, the network never settles into a discrete configuration
    and is still rattling around the decision boundaries.
    """
    if prev_codes is None:
        return float("nan")
    cur = snapshot_codes(model)
    changed = sum((a != b).sum().item() for a, b in zip(cur, prev_codes))
    total = sum(a.numel() for a in cur)
    return changed / max(total, 1)


# ==============================================================================
# 7. TRAINING
# ==============================================================================
def build_optimizer(model, cfg=CFG):
    """
    IMPORTANT: weight decay is NOT applied to the latent w_real/w_imag.

    The projection is scale-invariant - only the signs and the ratio of |Re| to
    |Im| matter - so weight decay does not regularise the network at all. All
    it does is drag the weights toward zero, into the region where the corner
    decision is noisiest and weights start flickering between corners.
    """
    decay, no_decay = [], []
    latent_names = set()
    for m in model.modules():
        if isinstance(m, _ComplexQuantBase) and m.quantize:
            latent_names.add(id(m.w_real))
            latent_names.add(id(m.w_imag))

    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in latent_names or p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr)


def make_grad_scaler(cfg=CFG):
    """GradScaler that works with both the new (torch>=2.3) and the old API."""
    enabled = cfg.use_amp and cfg.device == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, criterion, optimizer, scaler, cfg, train=True):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    t0 = time.time()

    for step, (x, y) in enumerate(loader):
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        amp_on = cfg.use_amp and cfg.device == "cuda"
        with torch.autocast(device_type=cfg.device, enabled=amp_on):
            logits = model(x)
            loss = criterion(logits.float(), y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Clipping the latent weights: without it they drift to +-inf, the
            # corner decision freezes and the layer stops learning entirely.
            if cfg.weight_clip is not None:
                for layer in quant_layers(model):
                    layer.clip_latent_(cfg.weight_clip)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += bs

        if train and cfg.log_every and step % cfg.log_every == 0:
            print(f"    step {step:4d}/{len(loader)}  "
                  f"loss {total_loss / total_n:.4f}  "
                  f"acc {100 * total_correct / total_n:.2f}%")

    return total_loss / total_n, 100.0 * total_correct / total_n, time.time() - t0


def main(cfg=CFG):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("=" * 78)
    print(" CVQNN - weights at the 4th roots of unity {+1, -1, +i, -i}")
    print("=" * 78)
    print(f"device            : {cfg.device} "
          f"({torch.cuda.get_device_name(0) if cfg.device == 'cuda' else 'cpu'})")
    print(f"quantize          : {cfg.quantize} "
          f"(stem={cfg.quantize_stem}, head={cfg.quantize_head})")
    print(f"widths / blocks   : {cfg.widths} / {cfg.blocks}")
    print(f"epochs / bs / lr  : {cfg.epochs} / {cfg.batch_size} / {cfg.lr}")

    # Before downloading data or building the model: if the GPU is unusable
    # there is no reason to burn session time.
    check_gpu_compat(cfg)

    train_loader, test_loader = build_loaders(cfg)

    model = CVQResNet(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_quant = sum(l.w_real.numel() + l.w_imag.numel() for l in quant_layers(model)) // 2
    print(f"total parameters  : {n_params / 1e6:.2f}M")
    print(f"quantized weights : {n_quant / 1e6:.2f}M "
          f"(~{n_quant * 2 / 8 / 1e6:.2f} MB packed at 2 bits each)")
    print("-" * 78)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = make_grad_scaler(cfg)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "flip_rate": [], "lr": []}
    prev_codes = snapshot_codes(model) if cfg.quantize else None
    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        print(f"[epoch {epoch:3d}/{cfg.epochs}]  lr={optimizer.param_groups[0]['lr']:.2e}")
        tr_loss, tr_acc, tr_time = run_epoch(model, train_loader, criterion,
                                             optimizer, scaler, cfg, train=True)
        with torch.no_grad():
            va_loss, va_acc, _ = run_epoch(model, test_loader, criterion,
                                           optimizer, scaler, cfg, train=False)
        scheduler.step()

        fr = flip_rate(model, prev_codes) if cfg.quantize else float("nan")
        if cfg.quantize:
            prev_codes = snapshot_codes(model)

        hist["train_loss"].append(tr_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)
        hist["flip_rate"].append(fr)
        hist["lr"].append(optimizer.param_groups[0]["lr"])

        star = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(cfg.out_dir, "best.pt"))
            star = "  <- best"
        print(f"  train {tr_loss:.4f}/{tr_acc:5.2f}%   "
              f"val {va_loss:.4f}/{va_acc:5.2f}%   "
              f"flip {fr * 100:5.2f}%   {tr_time:.0f}s{star}")

    print("-" * 78)
    print(f"BEST val accuracy: {best_acc:.2f}%")

    if cfg.quantize:
        dist, total = phase_histogram(model)
        labels = ["+1", "-1", "+i", "-i"]
        print(f"Final weight distribution over corners ({total / 1e6:.2f}M weights):")
        for lab, p in zip(labels, dist):
            print(f"   {lab:>2} : {p * 100:5.2f}%")
    else:
        dist, labels = None, None

    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump({"history": hist, "best_acc": best_acc,
                   "phase_dist": dist}, f, indent=2)

    plot_results(hist, dist, labels, cfg)
    return model, hist


# ==============================================================================
# 8. PLOTS
# ==============================================================================
def plot_results(hist, dist, labels, cfg=CFG):
    n_panels = 4 if dist is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    epochs = range(1, len(hist["train_loss"]) + 1)

    axes[0].plot(epochs, hist["train_loss"], label="train")
    axes[0].plot(epochs, hist["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, hist["train_acc"], label="train")
    axes[1].plot(epochs, hist["val_acc"], label="val")
    axes[1].set_title("Accuracy, %"); axes[1].set_xlabel("epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, [f * 100 for f in hist["flip_rate"]], color="crimson")
    axes[2].set_title("Weight flip rate, % / epoch")
    axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3)

    if dist is not None:
        axes[3].bar(labels, [d * 100 for d in dist],
                    color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
        axes[3].axhline(25, ls="--", c="gray", lw=1)
        axes[3].set_title("Corner distribution, %")
        axes[3].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "curves.png"), dpi=140)
    plt.show()


# ==============================================================================
# 9. A/B EXPERIMENT: quantized network vs FP32 control
# ==============================================================================
def run_ab(cfg=CFG):
    """
    Train two architecturally IDENTICAL networks from the same seed:
      A) weights constrained to {+1,-1,+i,-i}   B) plain complex FP32 weights

    The meaningful research result is the DELTA between them, not the absolute
    accuracy: it answers "what does compressing a weight to 2 bits cost?".
    """
    base_out = cfg.out_dir

    class QuantCFG(cfg):
        quantize = True

    class FP32CFG(cfg):
        quantize = False

    # assigned from outside: a class body cannot see the enclosing function's locals
    QuantCFG.out_dir = os.path.join(base_out, "quant")
    FP32CFG.out_dir = os.path.join(base_out, "fp32")

    print("\n\n" + "#" * 78)
    print("#  RUN A: QUANTIZED NETWORK  {+1, -1, +i, -i}")
    print("#" * 78)
    _, hist_q = main(QuantCFG)

    print("\n\n" + "#" * 78)
    print("#  RUN B: CONTROL - complex FP32 network, same architecture")
    print("#" * 78)
    _, hist_f = main(FP32CFG)

    best_q = max(hist_q["val_acc"])
    best_f = max(hist_f["val_acc"])

    print("\n" + "=" * 78)
    print(" A/B RESULT")
    print("=" * 78)
    print(f"  quantized (2 bits/weight) : {best_q:.2f}%")
    print(f"  FP32 control (32 bits)    : {best_f:.2f}%")
    print(f"  COST OF QUANTIZATION      : {best_f - best_q:+.2f} pp "
          f"for a 16x weight compression")

    # comparison plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, len(hist_q["val_acc"]) + 1)
    ax[0].plot(ep, hist_q["val_acc"], label=f"quantized {{+-1,+-i}} ({best_q:.1f}%)")
    ax[0].plot(ep, hist_f["val_acc"], label=f"FP32 ({best_f:.1f}%)")
    ax[0].set_title("Val accuracy: quantized vs FP32")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, hist_q["val_loss"], label="quantized")
    ax[1].plot(ep, hist_f["val_loss"], label="FP32")
    ax[1].set_title("Val loss"); ax[1].set_xlabel("epoch")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "ab_comparison.png"), dpi=140)
    plt.show()

    with open(os.path.join(base_out, "ab_summary.json"), "w") as f:
        json.dump({"quant_best": best_q, "fp32_best": best_f,
                   "gap_pp": best_f - best_q}, f, indent=2)
    return hist_q, hist_f


if __name__ == "__main__":
    # The mode comes from an environment variable so that this file stays
    # self-contained: a Kaggle script kernel takes exactly one file and no argv.
    #   quant - quantized network only
    #   fp32  - FP32 control only
    #   both  - both runs plus the comparison (default, see CFG.mode)
    mode = os.environ.get("CVQNN_MODE", CFG.mode).lower()
    print(f"[mode] CVQNN_MODE={mode}")

    if mode == "both":
        run_ab(CFG)
    elif mode == "fp32":
        class _FP(CFG):
            quantize = False
        main(_FP)
    else:
        main(CFG)
