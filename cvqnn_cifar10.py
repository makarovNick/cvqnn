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
    # True  -> weights are quantized (see weight_mode below)
    # False -> plain complex FP32 network: the full-precision control arm
    quantize        = True

    # Which quantizer, when quantize is True:
    #   "phase4" - {+1,-1,+i,-i}, 2 bits/weight: the hypothesis under test
    #   "binary" - {+1,-1}, 1 bit/weight, purely real: the control that asks
    #              whether phase beats sign. In this mode the imaginary path is
    #              switched off entirely (see real_only in ComplexAmpNorm) -
    #              merely zeroing the imaginary weights is not enough, because
    #              the learnable beta_im would revive the imaginary channel and
    #              hand the baseline capacity a binary network must not have.
    weight_mode     = "phase4"

    # Widths for the binary arm. At 1 bit/weight it needs 2x the weights to
    # match the 2-bit budget, and parameter count scales with width^2, so the
    # channels grow by sqrt(2). main() prints the actual bit totals so the
    # match can be audited rather than taken on trust.
    binary_widths   = (68, 136, 272)

    # --- SCALING LADDER (mode = "scaling") ---
    # 32x compression means the saved memory can be spent on a bigger network.
    # The question is not only "does bigger help" but "bigger HOW": for low-bit
    # networks the literature generally finds width more valuable than depth,
    # because each layer's capacity is capped by its channel count and
    # quantization noise compounds with depth. So both arms below carry the
    # SAME parameter budget (2x the base run) and differ only in how it is
    # spent - which turns a vague "scale it up" into an actual comparison.
    scaling_arms = (
        {"name": "deeper", "widths": (48, 96, 192),  "blocks": (4, 4, 4)},
        # 70 rather than 68: doubling the blocks does not exactly double the
        # parameters (the stem and the downsample convs do not scale with it),
        # so the widths are nudged until both arms land within ~1% of each other
        {"name": "wider",  "widths": (70, 140, 280), "blocks": (2, 2, 2)},
    )

    # Best val accuracy of the recorded FP32 control (results/kaggle-t4-40ep),
    # quoted for context so a scaling run does not have to retrain it.
    fp32_reference       = 91.60
    fp32_reference_mb    = 12.49
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
    mode            = "vs_binary"

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


class BinarySignSTE(torch.autograd.Function):
    """Binary {+1,-1} quantizer, the control arm against which phase is judged.

    Same straight-through trick, same sign(0) = +1 convention as PhaseQuantSTE,
    so the only difference between the two arms is the size of the codebook.
    """

    @staticmethod
    def forward(ctx, w_real):
        ones = torch.ones_like(w_real)
        return torch.where(w_real >= 0, ones, -ones)

    @staticmethod
    def backward(ctx, g):
        return g


def binary_quantize(w_real):
    return BinarySignSTE.apply(w_real)


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
                 per_channel_scale=True, weight_mode="phase4"):
        super().__init__()
        self.quantize = quantize
        self.weight_mode = weight_mode if quantize else "fp32"
        self.fan_in = fan_in

        # Latent FP32 copies, initialised ~ N(0, 1/sqrt(fan_in)). The projection
        # itself only cares about signs and relative magnitudes, but this scale
        # sits comfortably inside the [-1, 1] clipping range.
        std = 1.0 / math.sqrt(fan_in)
        self.w_real = nn.Parameter(torch.randn(weight_shape) * std)

        if self.weight_mode == "binary":
            # No imaginary weights at all in the binary arm - registering them
            # as a zero buffer rather than a parameter keeps the diagnostics
            # working (phase_code then reports every weight on a real corner,
            # which is exactly right) without inflating the parameter count or
            # creating a tensor that never receives a gradient.
            self.register_buffer("w_imag", torch.zeros(weight_shape))
        else:
            self.w_imag = nn.Parameter(torch.randn(weight_shape) * std)

        # Learnable REAL scale: quantized weights have unit modulus, so output
        # variance grows with fan_in. The scale pulls the signal back into a
        # sane range (the analogue of alpha in XNOR-Net).
        n_scale = out_features if per_channel_scale else 1
        self.scale = nn.Parameter(torch.full((n_scale,), 1.0 / math.sqrt(fan_in)))

    def effective_weights(self):
        if not self.quantize:
            return self.w_real, self.w_imag
        if self.weight_mode == "binary":
            return binary_quantize(self.w_real), self.w_imag  # w_imag is a zero buffer
        return phase_quantize(self.w_real, self.w_imag)

    def bits_per_weight(self):
        return {"phase4": 2, "binary": 1}.get(self.weight_mode, 32)

    @torch.no_grad()
    def clip_latent_(self, limit):
        if self.quantize and limit is not None:
            self.w_real.clamp_(-limit, limit)
            if isinstance(self.w_imag, nn.Parameter):
                self.w_imag.clamp_(-limit, limit)


class ComplexQuantLinear(_ComplexQuantBase):
    def __init__(self, in_features, out_features, quantize=True, per_channel_scale=True,
                 weight_mode="phase4"):
        super().__init__(
            weight_shape=(out_features, in_features),
            out_features=out_features,
            fan_in=in_features,
            quantize=quantize,
            per_channel_scale=per_channel_scale,
            weight_mode=weight_mode,
        )

    def forward(self, x_re, x_im):
        w_re, w_im = self.effective_weights()
        out_re, out_im = complex_op(F.linear, x_re, x_im, w_re, w_im,
                                    cat_dim=0, chunk_dim=-1)
        s = self.scale.to(out_re.dtype)
        return out_re * s, out_im * s


class ComplexQuantConv2d(_ComplexQuantBase):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0,
                 quantize=True, per_channel_scale=True, weight_mode="phase4"):
        super().__init__(
            weight_shape=(out_ch, in_ch, kernel_size, kernel_size),
            out_features=out_ch,
            fan_in=in_ch * kernel_size * kernel_size,
            quantize=quantize,
            per_channel_scale=per_channel_scale,
            weight_mode=weight_mode,
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

    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True,
                 real_only=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        # real_only is what makes the binary control genuinely real-valued.
        # With real weights the imaginary channel would still be non-zero,
        # because beta_im injects a learnable bias into it - giving the
        # baseline a second, independent real network for free. Dropping
        # beta_im keeps Im identically zero from input to logits.
        self.real_only = real_only
        self.register_buffer("running_amp", torch.ones(num_features))
        if affine:
            # gamma is shared by Re and Im: a phase-preserving rescale
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta_re = nn.Parameter(torch.zeros(num_features))
            if not real_only:
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
            x_im = x_im * g
            if not self.real_only:
                x_im = x_im + self.beta_im.view(shape).to(x_im.dtype)
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
def is_real_only(cfg):
    """True when the network must stay purely real: the binary control arm."""
    return cfg.quantize and getattr(cfg, "weight_mode", "phase4") == "binary"


def effective_widths(cfg):
    """Widths for this arm - the binary control is widened to match bits."""
    if is_real_only(cfg):
        return cfg.binary_widths
    return cfg.widths


class ComplexBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, cfg=CFG):
        super().__init__()
        wm = getattr(cfg, "weight_mode", "phase4")
        ro = is_real_only(cfg)

        self.conv1 = ComplexQuantConv2d(in_ch, out_ch, 3, stride, 1,
                                        cfg.quantize, cfg.per_channel_scale, wm)
        self.norm1 = ComplexAmpNorm(out_ch, real_only=ro)
        self.act = ComplexSplitReLU()
        self.conv2 = ComplexQuantConv2d(out_ch, out_ch, 3, 1, 1,
                                        cfg.quantize, cfg.per_channel_scale, wm)
        self.norm2 = ComplexAmpNorm(out_ch, real_only=ro)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ComplexSequential(
                ComplexQuantConv2d(in_ch, out_ch, 1, stride, 0,
                                   cfg.quantize, cfg.per_channel_scale, wm),
                ComplexAmpNorm(out_ch, real_only=ro),
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
        w = effective_widths(cfg)
        b = cfg.blocks
        wm = getattr(cfg, "weight_mode", "phase4")
        ro = is_real_only(cfg)
        self.real_only = ro

        # --- Stem ---
        self.stem = ComplexSequential(
            ComplexQuantConv2d(3, w[0], 3, 1, 1,
                               cfg.quantize and cfg.quantize_stem,
                               cfg.per_channel_scale, wm),
            ComplexAmpNorm(w[0], real_only=ro),
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
                                       cfg.per_channel_scale, wm)
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
    print(f"total parameters  : {n_params / 1e6:.2f}M")

    # The bit budget is printed rather than assumed: the whole point of the
    # binary control is that it costs the SAME number of bits, and a claim
    # like that should be auditable from the log instead of trusted.
    qls = quant_layers(model)
    if qls:
        n_quant = sum(l.w_real.numel() for l in qls)
        total_bits = sum(l.w_real.numel() * l.bits_per_weight() for l in qls)
        print(f"quantized weights : {n_quant / 1e6:.2f}M "
              f"@ {qls[0].bits_per_weight()} bit ({cfg.weight_mode})")
        print(f"WEIGHT BIT BUDGET : {total_bits / 1e6:.2f} Mbit "
              f"(~{total_bits / 8 / 1e6:.2f} MB packed)")
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
    # A complex FP32 weight is w_real + w_imag = two float32 = 64 bits, not 32.
    # Getting this wrong understates the compression by a factor of two.
    print(f"  quantized (2 bits/complex weight) : {best_q:.2f}%")
    print(f"  FP32 control (64 bits)            : {best_f:.2f}%")
    print(f"  COST OF QUANTIZATION              : {best_f - best_q:+.2f} pp "
          f"for a 32x weight compression")

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


# ==============================================================================
# 10. THE DECISIVE CONTROL: phase (2 bits) vs sign (1 bit) at an equal bit budget
# ==============================================================================
def run_phase_vs_binary(cfg=CFG):
    """
    Does phase beat sign?

    Comparing a 2-bit complex network against a 1-bit real one of the same
    width would be a rigged fight - it simply has twice the storage. So the
    binary arm is widened by sqrt(2) (parameter count grows as width^2), which
    doubles its weight count and lands both arms on the same number of bits.

    That makes the question the honest one: given a fixed memory budget, is it
    better to spend it on a larger codebook per weight, or on more weights?
    """
    base_out = cfg.out_dir

    class PhaseCFG(cfg):
        quantize = True
        weight_mode = "phase4"

    class BinCFG(cfg):
        quantize = True
        weight_mode = "binary"

    PhaseCFG.out_dir = os.path.join(base_out, "phase4")
    BinCFG.out_dir = os.path.join(base_out, "binary")

    print("\n\n" + "#" * 78)
    print("#  ARM A: PHASE  {+1, -1, +i, -i}  - 2 bits/weight")
    print("#" * 78)
    _, hist_p = main(PhaseCFG)

    print("\n\n" + "#" * 78)
    print(f"#  ARM B: BINARY  {{+1, -1}}  - 1 bit/weight, widened to {cfg.binary_widths}")
    print("#" * 78)
    _, hist_b = main(BinCFG)

    best_p = max(hist_p["val_acc"])
    best_b = max(hist_b["val_acc"])

    print("\n" + "=" * 78)
    print(" PHASE vs SIGN, equal bit budget")
    print("=" * 78)
    print(f"  phase  {cfg.widths}, 2 bit/weight : {best_p:.2f}%")
    print(f"  binary {cfg.binary_widths}, 1 bit/weight : {best_b:.2f}%")
    print(f"  PHASE ADVANTAGE : {best_p - best_b:+.2f} pp")
    print("  (compare the WEIGHT BIT BUDGET lines above: if they differ by more")
    print("   than a few percent, the arms are not matched and this gap is void)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, len(hist_p["val_acc"]) + 1)
    ax[0].plot(ep, hist_p["val_acc"], label=f"phase 2-bit ({best_p:.1f}%)")
    ax[0].plot(ep, hist_b["val_acc"], label=f"binary 1-bit, wider ({best_b:.1f}%)")
    ax[0].set_title("Val accuracy at an equal bit budget")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, [f * 100 for f in hist_p["flip_rate"]], label="phase")
    ax[1].plot(ep, [f * 100 for f in hist_b["flip_rate"]], label="binary")
    ax[1].set_title("Weight flip rate, % / epoch")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "phase_vs_binary.png"), dpi=140)
    plt.show()

    with open(os.path.join(base_out, "phase_vs_binary.json"), "w") as f:
        json.dump({"phase_best": best_p, "binary_best": best_b,
                   "phase_advantage_pp": best_p - best_b,
                   "phase_widths": list(cfg.widths),
                   "binary_widths": list(cfg.binary_widths)}, f, indent=2)
    return hist_p, hist_b


# ==============================================================================
# 11. SCALING: spending the saved memory
# ==============================================================================
def deployed_bits(cfg):
    """Bits the weights of this configuration would occupy once deployed.

    Latent FP32 copies exist only during training; a shipped quantized model
    stores a corner index per weight. Reporting the checkpoint size instead
    would overstate a quantized model by 32x.
    """
    probe = CVQResNet(cfg)
    qls = quant_layers(probe)
    if qls:
        return sum(l.w_real.numel() * l.bits_per_weight() for l in qls)
    # complex FP32: w_real + w_imag, two float32 per complex weight
    n_complex = sum(m.w_real.numel() for m in probe.modules()
                    if isinstance(m, _ComplexQuantBase))
    return n_complex * 2 * 32


def run_scaling(cfg=CFG):
    """
    Quantization bought a 32x memory saving. This asks what to buy with it.

    Every arm carries the same parameter budget and differs only in how it is
    spent - depth versus width - so the comparison isolates the choice instead
    of confounding it with "one network is simply larger".

    The FP32 control is not retrained: it is quoted from the recorded run,
    which used the same seed, schedule and data pipeline. Its accuracy is a
    reference line, not a competitor measured under different conditions.
    """
    base_out = cfg.out_dir
    rows = []

    for arm in cfg.scaling_arms:
        class ArmCFG(cfg):
            quantize = True
            weight_mode = "phase4"

        ArmCFG.widths = tuple(arm["widths"])
        ArmCFG.blocks = tuple(arm["blocks"])
        ArmCFG.out_dir = os.path.join(base_out, arm["name"])

        bits = deployed_bits(ArmCFG)
        print("\n\n" + "#" * 78)
        print(f"#  ARM '{arm['name']}': widths={ArmCFG.widths} blocks={ArmCFG.blocks}")
        print(f"#  deployed weights: {bits / 8 / 1e6:.2f} MB "
              f"({cfg.fp32_reference_mb / (bits / 8 / 1e6):.1f}x smaller than FP32)")
        print("#" * 78)

        _, hist = main(ArmCFG)
        rows.append({
            "name": arm["name"],
            "widths": list(ArmCFG.widths),
            "blocks": list(ArmCFG.blocks),
            "mb": bits / 8 / 1e6,
            "best": max(hist["val_acc"]),
            "final_flip": hist["flip_rate"][-1],
            "hist": hist,
        })

    print("\n" + "=" * 78)
    print(" SCALING RESULT - equal parameter budget, different shape")
    print("=" * 78)
    print(f"{'arm':<10}{'widths':<18}{'blocks':<12}{'MB':>7}{'vs FP32':>9}"
          f"{'val acc':>9}{'gap':>8}")
    print("-" * 78)
    for r in rows:
        gap = r["best"] - cfg.fp32_reference
        print(f"{r['name']:<10}{str(tuple(r['widths'])):<18}"
              f"{str(tuple(r['blocks'])):<12}{r['mb']:>7.2f}"
              f"{cfg.fp32_reference_mb / r['mb']:>8.0f}x{r['best']:>9.2f}{gap:>+8.2f}")
    print("-" * 78)
    print(f"{'FP32 ref':<10}{'(48, 96, 192)':<18}{'(2, 2, 2)':<12}"
          f"{cfg.fp32_reference_mb:>7.2f}{'1x':>9}{cfg.fp32_reference:>9.2f}"
          f"{0.0:>+8.2f}")
    print("\nfinal flip rate per arm (near zero = quantization settled):")
    for r in rows:
        print(f"  {r['name']:<10}{r['final_flip']*100:6.3f}%")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in rows:
        ep = range(1, len(r["hist"]["val_acc"]) + 1)
        ax[0].plot(ep, r["hist"]["val_acc"],
                   label=f"{r['name']} {tuple(r['widths'])}x{tuple(r['blocks'])} "
                         f"({r['best']:.1f}%)")
        ax[1].plot(ep, [f * 100 for f in r["hist"]["flip_rate"]], label=r["name"])
    ax[0].axhline(cfg.fp32_reference, ls="--", c="gray",
                  label=f"FP32 reference ({cfg.fp32_reference:.1f}%)")
    ax[0].set_title("Val accuracy at an equal parameter budget")
    ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_title("Weight flip rate, % / epoch")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "scaling.png"), dpi=140)
    plt.show()

    with open(os.path.join(base_out, "scaling.json"), "w") as f:
        json.dump({"fp32_reference": cfg.fp32_reference,
                   "fp32_reference_mb": cfg.fp32_reference_mb,
                   "arms": [{k: v for k, v in r.items() if k != "hist"}
                            for r in rows]}, f, indent=2)
    return rows


if __name__ == "__main__":
    # The mode comes from an environment variable so that this file stays
    # self-contained: a Kaggle script kernel takes exactly one file and no argv.
    #   quant     - quantized network only
    #   fp32      - FP32 control only
    #   both      - quantized + FP32 control and their comparison
    #   vs_binary - phase vs sign at an equal bit budget
    #   scaling   - spend the saved memory: deeper vs wider (see CFG.mode)
    mode = os.environ.get("CVQNN_MODE", CFG.mode).lower()
    print(f"[mode] CVQNN_MODE={mode}")

    if mode == "both":
        run_ab(CFG)
    elif mode == "vs_binary":
        run_phase_vs_binary(CFG)
    elif mode == "scaling":
        run_scaling(CFG)
    elif mode == "fp32":
        class _FP(CFG):
            quantize = False
        main(_FP)
    else:
        main(CFG)
