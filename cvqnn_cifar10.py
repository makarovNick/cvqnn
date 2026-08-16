"""
================================================================================
 CVQNN — Complex-Valued Quantized Neural Network
 Веса жёстко зафиксированы в корнях 4-й степени из единицы: {+1, -1, +i, -i}
================================================================================

 Идея:
   Каждый вес сети — это комплексное число единичного модуля, лежащее в одной
   из 4 вершин "фазового квадрата". Т.е. на вес приходится ровно 2 бита
   информации (log2(4)), но, в отличие от бинарных сетей {+1,-1}, у нас есть
   ещё и фазовая степень свободы: умножение на i — это поворот на 90°,
   который ПЕРЕМЕШИВАЕТ вещественный и мнимый каналы сигнала.

   Обучение идёт по латентным FP32-копиям (w_real, w_imag), а в forward они
   проецируются на ближайшую вершину. Градиент проходит насквозь (STE).

 Архитектура: компактный комплекснозначный ResNet (CIFAR-style).
   ResNet выбран вместо ViT: с нуля на 50k картинок он стабильнее,
   не требует длинного warmup / сильных аугментаций, и переживает
   агрессивное квантование заметно лучше (skip-connection даёт
   градиенту чистый путь мимо квантованных слоёв).

 Запуск: просто вставить в Kaggle Notebook (Accelerator: GPU T4/P100) и Run All.
 Зависимости: torch, torchvision, numpy, matplotlib — всё есть на Kaggle из коробки.
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
# 0. КОНФИГ
# ==============================================================================
class CFG:
    # --- воспроизводимость / железо ---
    seed            = 1337
    device          = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers     = 2               # на Kaggle больше 2-4 смысла не имеет
    use_amp         = True            # fp16 autocast (критичные места считаются в fp32)

    # --- данные ---
    data_root       = "./data"
    batch_size      = 128
    val_batch_size  = 512

    # --- модель ---
    widths          = (48, 96, 192)   # каналы по стадиям
    blocks          = (2, 2, 2)       # residual-блоков на стадию
    num_classes     = 10

    # --- РЕЖИМ КВАНТОВАНИЯ (главный тумблер эксперимента) ---
    # True  -> веса проецируются в {+1,-1,+i,-i} (наша гипотеза)
    # False -> обычная комплексная FP32-сеть (контрольный baseline для абляции)
    quantize        = True
    quantize_stem   = True            # квантовать первый conv (классика BNN — оставлять FP)
    quantize_head   = True            # квантовать последний linear
    per_channel_scale = True          # scale как вектор на выходной канал (False -> один скаляр)
    weight_clip     = 1.0             # клиппинг латентных весов после шага; None -> выключить

    # --- обучение ---
    epochs          = 40
    lr              = 2e-3
    weight_decay    = 5e-2            # применяется ТОЛЬКО к не-квантованным параметрам
    label_smoothing = 0.1
    grad_clip       = 5.0
    log_every       = 100             # шагов между промежуточными логами

    # --- что запускать ---
    # "both" — квантованная сеть + FP32-контроль и сравнение между ними.
    #          Именно дельта между прогонами отвечает на вопрос ресерча,
    #          поэтому это значение по умолчанию.
    # "quant" / "fp32" — только один из прогонов.
    # Переопределяется переменной окружения CVQNN_MODE (на Kaggle её не задать,
    # поэтому основной способ настройки — правка этой строки).
    mode            = "both"

    # --- вывод ---
    out_dir         = "./cvqnn_out"


def check_gpu_compat(cfg=CFG):
    """
    Ранняя проверка, что сборка PyTorch содержит ядра под выданную карту.

    Иначе первая же CUDA-операция падает с cudaErrorNoKernelImageForDevice
    где-то в глубине сети, и по трейсбеку это выглядит как ошибка в нашем коде.
    Реальный случай: Kaggle выдаёт Tesla P100 (sm_60), а их предустановленный
    torch собран под sm_70+ — поддержку Pascal из сборок убрали.
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
            f"GPU {name} имеет compute capability {sm}, но установленный\n"
            f"PyTorch {torch.__version__} собран только под: {' '.join(arch_list)}.\n"
            f"Любая CUDA-операция упадёт с cudaErrorNoKernelImageForDevice.\n\n"
            f"Что делать: запросить другой ускоритель (на Kaggle — T4 вместо\n"
            f"P100: machine_shape в kernel-metadata.json), либо поставить сборку\n"
            f"torch под {sm}, либо считать на CPU (CFG.device = 'cpu').\n"
            f"{'!' * 70}"
        )
    print(f"[gpu] {name} ({sm}) — совместимость с torch {torch.__version__} подтверждена")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# 1. КВАНТОВАТОР: PhaseQuant + Straight-Through Estimator
# ==============================================================================
class PhaseQuantSTE(torch.autograd.Function):
    """
    Forward: комплексный вес W = w_real + i*w_imag проецируется на ближайшую
             вершину фазового квадрата {+1, -1, +i, -i}.

             Правило проекции — "кто больше по модулю, тот и выжил":
               |Re| >= |Im|  ->  q = sign(Re) * 1     (вещественная ось)
               |Re| <  |Im|  ->  q = sign(Im) * i     (мнимая ось)

             Геометрически это ровно проекция на ближайшую вершину: границы
             решения — диагонали Re = ±Im, что и есть биссектрисы между
             соседними вершинами квадрата.

    Backward: чистый STE — градиент по квантованному значению без изменений
              уходит на латентные FP32-копии. Функция проекции кусочно-
              постоянна, её настоящая производная равна нулю почти всюду,
              поэтому её подменяем тождественной.
    """

    @staticmethod
    def forward(ctx, w_real, w_imag):
        # sign с соглашением sign(0) = +1, чтобы не рождать "мёртвые" нулевые веса
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
        # Straight-Through: пропускаем как есть.
        return g_re, g_im


def phase_quantize(w_real, w_imag):
    return PhaseQuantSTE.apply(w_real, w_imag)


@torch.no_grad()
def phase_code(w_real, w_imag):
    """Целочисленный код вершины: 0:+1, 1:-1, 2:+i, 3:-i. Нужен для диагностики."""
    real_dominant = w_real.abs() >= w_imag.abs()
    z = torch.zeros_like(w_real, dtype=torch.int8)
    code = torch.where(
        real_dominant,
        torch.where(w_real >= 0, z + 0, z + 1),
        torch.where(w_imag >= 0, z + 2, z + 3),
    )
    return code


# ==============================================================================
# 2. КОМПЛЕКСНАЯ АЛГЕБРА ДЛЯ СЛОЁВ
# ==============================================================================
def complex_op(op, x_re, x_im, w_re, w_im, cat_dim, chunk_dim):
    """
    Комплексное применение линейного оператора (linear / conv2d):

        Out_real = X_real * W_real - X_imag * W_imag
        Out_imag = X_real * W_imag + X_imag * W_real

    Наивно это 4 вызова op(). Мы склеиваем [W_real; W_imag] по выходной
    размерности и делаем 2 вызова с удвоенным числом выходов — FLOPs те же,
    но вдвое меньше запусков ядер, что заметно на мелких свёртках.
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
    """Общая логика: латентные веса, квантование, обучаемый масштаб."""

    def __init__(self, weight_shape, out_features, fan_in, quantize=True,
                 per_channel_scale=True):
        super().__init__()
        self.quantize = quantize
        self.fan_in = fan_in

        # Латентные FP32-копии. Инициализация ~ N(0, 1/sqrt(fan_in)):
        # для самой проекции важны только знаки и относительные модули,
        # но такой масштаб хорошо согласуется с клиппингом в [-1, 1].
        std = 1.0 / math.sqrt(fan_in)
        self.w_real = nn.Parameter(torch.randn(weight_shape) * std)
        self.w_imag = nn.Parameter(torch.randn(weight_shape) * std)

        # Обучаемый ВЕЩЕСТВЕННЫЙ масштаб: квантованные веса имеют единичный
        # модуль, поэтому дисперсия выхода раздувается как fan_in.
        # scale возвращает сигнал в разумный диапазон (аналог alpha в XNOR-Net).
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
# 3. АКТИВАЦИЯ И НОРМАЛИЗАЦИЯ
# ==============================================================================
class ComplexSplitReLU(nn.Module):
    """ReLU покомпонентно: relu(Re) + i*relu(Im)."""

    def forward(self, x_re, x_im):
        return F.relu(x_re), F.relu(x_im)


class ComplexAmpNorm(nn.Module):
    """
    Амплитудная нормализация: обе компоненты делятся на ОДИН И ТОТ ЖЕ
    средний модуль сигнала (по батчу и пространству, отдельно на канал).

        amp   = sqrt(Re^2 + Im^2 + eps)
        m_c   = mean_{N,H,W} amp
        Re,Im = Re/m_c, Im/m_c

    Ключевое свойство: деление на общий модуль СОХРАНЯЕТ ФАЗУ сигнала —
    мы двигаем только длину вектора, не поворачивая его. Обычный BatchNorm,
    применённый к Re и Im по отдельности, фазу бы разрушил.

    Running-статистика ведётся как в BatchNorm, чтобы eval был детерминирован.
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5, affine=True):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.register_buffer("running_amp", torch.ones(num_features))
        if affine:
            # gamma — общий для Re и Im (фазосохраняющее растяжение)
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta_re = nn.Parameter(torch.zeros(num_features))
            self.beta_im = nn.Parameter(torch.zeros(num_features))

    def forward(self, x_re, x_im):
        dims = [0] + list(range(2, x_re.dim()))          # всё кроме канальной оси
        shape = [1, -1] + [1] * (x_re.dim() - 2)

        if self.training:
            # считаем статистику в fp32: под autocast fp16 sqrt легко даёт inf/0
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
    """nn.Sequential для модулей, работающих с парой (Re, Im)."""

    def forward(self, x_re, x_im):
        for module in self:
            x_re, x_im = module(x_re, x_im)
        return x_re, x_im


# ==============================================================================
# 4. МОДЕЛЬ: комплексный ResNet
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

        # Комплексный residual: складываем покомпонентно (= сложение в C)
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

        # --- Стадии ---
        stages = []
        in_ch = w[0]
        for si, (out_ch, n_blocks) in enumerate(zip(w, b)):
            for bi in range(n_blocks):
                stride = 2 if (bi == 0 and si > 0) else 1
                stages.append(ComplexBasicBlock(in_ch, out_ch, stride, cfg))
                in_ch = out_ch
        self.stages = ComplexSequential(*stages)

        # --- Голова ---
        self.head = ComplexQuantLinear(in_ch, cfg.num_classes,
                                       cfg.quantize and cfg.quantize_head,
                                       cfg.per_channel_scale)
        # Амплитуда неотрицательна, её динамический диапазон мал ->
        # обучаемая температура + сдвиг на класс дают softmax'у нормальный размах.
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.logit_bias = nn.Parameter(torch.zeros(cfg.num_classes))

    def forward(self, x):
        # ВХОД: вещественная часть — нормализованные пиксели, мнимая — нули.
        # Фаза "рождается" уже внутри сети, при умножении на веса ±i.
        x_re = x
        x_im = torch.zeros_like(x)

        x_re, x_im = self.stem(x_re, x_im)
        x_re, x_im = self.stages(x_re, x_im)

        # Комплексный global average pooling
        x_re = x_re.mean(dim=(2, 3))
        x_im = x_im.mean(dim=(2, 3))

        out_re, out_im = self.head(x_re, x_im)

        # ВЫХОД: комплексный вектор -> вещественные логиты через амплитуду |z|
        mag = torch.sqrt(out_re.float() ** 2 + out_im.float() ** 2 + 1e-8)
        return self.logit_scale * mag + self.logit_bias


# ==============================================================================
# 5. ДАННЫЕ (CIFAR-10, с оффлайн-фолбэком для Kaggle без интернета)
# ==============================================================================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def resolve_cifar_root(default_root):
    """Ищем уже скачанный CIFAR-10 (Kaggle-датасеты), иначе качаем сами."""
    candidates = [default_root]
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for name in os.listdir(kaggle_input):
            candidates.append(os.path.join(kaggle_input, name))
    for root in candidates:
        if os.path.isdir(os.path.join(root, "cifar-10-batches-py")):
            print(f"[data] найден локальный CIFAR-10: {root}")
            return root, False
    print("[data] локальная копия не найдена -> скачиваем (нужен Internet: ON)")
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
# 6. ДИАГНОСТИКА КВАНТОВАНИЯ (то, ради чего эксперимент и ставится)
# ==============================================================================
def quant_layers(model):
    return [m for m in model.modules()
            if isinstance(m, _ComplexQuantBase) and m.quantize]


@torch.no_grad()
def phase_histogram(model):
    """Доли весов, севших в каждую из 4 вершин: {+1, -1, +i, -i}."""
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
    """Доля весов, сменивших вершину с прошлого замера — мера стабильности STE."""
    if prev_codes is None:
        return float("nan")
    cur = snapshot_codes(model)
    changed = sum((a != b).sum().item() for a, b in zip(cur, prev_codes))
    total = sum(a.numel() for a in cur)
    return changed / max(total, 1)


# ==============================================================================
# 7. ОБУЧЕНИЕ
# ==============================================================================
def build_optimizer(model, cfg=CFG):
    """
    ВАЖНО: weight decay НЕ применяется к латентным w_real/w_imag.
    Проекция масштабно-инвариантна (важны только знаки и |Re| vs |Im|),
    поэтому WD не регуляризует сеть, а лишь стягивает веса к нулю —
    к области, где решение о вершине максимально шумное и веса начинают
    беспорядочно "мигать" между вершинами.
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
    """GradScaler с поддержкой и нового (torch>=2.3), и старого API."""
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

            # Клиппинг латентных весов: без него они уходят на ±inf,
            # решение о вершине "замерзает" и слой перестаёт обучаться.
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
    print(" CVQNN — веса в корнях 4-й степени из единицы {+1, -1, +i, -i}")
    print("=" * 78)
    print(f"device            : {cfg.device} "
          f"({torch.cuda.get_device_name(0) if cfg.device == 'cuda' else 'cpu'})")
    print(f"quantize          : {cfg.quantize} "
          f"(stem={cfg.quantize_stem}, head={cfg.quantize_head})")
    print(f"widths / blocks   : {cfg.widths} / {cfg.blocks}")
    print(f"epochs / bs / lr  : {cfg.epochs} / {cfg.batch_size} / {cfg.lr}")

    # до скачивания данных и построения модели: если карта несовместима,
    # незачем тратить время сессии
    check_gpu_compat(cfg)

    train_loader, test_loader = build_loaders(cfg)

    model = CVQResNet(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_quant = sum(l.w_real.numel() + l.w_imag.numel() for l in quant_layers(model)) // 2
    print(f"параметров всего  : {n_params / 1e6:.2f}M")
    print(f"квантованных весов: {n_quant / 1e6:.2f}M "
          f"(~{n_quant * 2 / 8 / 1e6:.2f} MB при упаковке по 2 бита)")
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
    print(f"ЛУЧШАЯ val accuracy: {best_acc:.2f}%")

    if cfg.quantize:
        dist, total = phase_histogram(model)
        labels = ["+1", "-1", "+i", "-i"]
        print("Финальное распределение весов по вершинам "
              f"({total / 1e6:.2f}M весов):")
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
# 8. ГРАФИКИ
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
        axes[3].set_title("Распределение по вершинам, %")
        axes[3].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.out_dir, "curves.png"), dpi=140)
    plt.show()


# ==============================================================================
# 9. A/B-ЭКСПЕРИМЕНТ: квантованная сеть против FP32-контроля
# ==============================================================================
def run_ab(cfg=CFG):
    """
    Гоняет две ОДИНАКОВЫЕ по архитектуре сети с одним сидом:
      A) веса зажаты в {+1,-1,+i,-i}   B) обычные комплексные FP32-веса
    Осмысленный результат ресерча — это ДЕЛЬТА между ними, а не абсолютная
    точность: она отвечает на вопрос "сколько стоит сжатие веса до 2 бит".
    """
    base_out = cfg.out_dir

    class QuantCFG(cfg):
        quantize = True

    class FP32CFG(cfg):
        quantize = False

    # присваиваем снаружи: тело класса не видит локальные переменные функции
    QuantCFG.out_dir = os.path.join(base_out, "quant")
    FP32CFG.out_dir = os.path.join(base_out, "fp32")

    print("\n\n" + "#" * 78)
    print("#  ПРОГОН A: КВАНТОВАННАЯ СЕТЬ  {+1, -1, +i, -i}")
    print("#" * 78)
    _, hist_q = main(QuantCFG)

    print("\n\n" + "#" * 78)
    print("#  ПРОГОН B: КОНТРОЛЬ — комплексная FP32-сеть той же архитектуры")
    print("#" * 78)
    _, hist_f = main(FP32CFG)

    best_q = max(hist_q["val_acc"])
    best_f = max(hist_f["val_acc"])

    print("\n" + "=" * 78)
    print(" ИТОГ A/B")
    print("=" * 78)
    print(f"  квантованная (2 бита/вес) : {best_q:.2f}%")
    print(f"  FP32-контроль (32 бита)   : {best_f:.2f}%")
    print(f"  ЦЕНА КВАНТОВАНИЯ          : {best_f - best_q:+.2f} п.п. "
          f"при сжатии весов в 16 раз")

    # график сравнения
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ep = range(1, len(hist_q["val_acc"]) + 1)
    ax[0].plot(ep, hist_q["val_acc"], label=f"квант {{±1,±i}} ({best_q:.1f}%)")
    ax[0].plot(ep, hist_f["val_acc"], label=f"FP32 ({best_f:.1f}%)")
    ax[0].set_title("Val accuracy: квантование vs FP32")
    ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, hist_q["val_loss"], label="квант")
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
    # Режим задаётся переменной окружения, чтобы файл оставался
    # самодостаточным (Kaggle-скрипт принимает ровно один файл, без argv).
    #   quant (по умолчанию) — только квантованная сеть
    #   fp32                 — только FP32-контроль
    #   both                 — оба прогона + сравнение
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
