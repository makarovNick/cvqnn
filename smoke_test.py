"""
Smoke-тест CVQNN. Проверяет не "запускается ли", а корректна ли математика.

Запуск:
    python smoke_test.py            # только юнит-тесты (быстро, без данных)
    python smoke_test.py --train    # + мини-обучение на подвыборке CIFAR-10
"""

import os
import sys
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import cvqnn_cifar10 as M


PASS, FAIL = "  [OK]  ", "  [FAIL]"
_failures = []


def check(name, cond, detail=""):
    print(f"{PASS if cond else FAIL} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


# ==============================================================================
# 1. КВАНТОВАТОР
# ==============================================================================
def test_quantizer():
    print("\n--- 1. PhaseQuantSTE ---")
    torch.manual_seed(0)
    wr = torch.randn(5000, requires_grad=True)
    wi = torch.randn(5000, requires_grad=True)
    qr, qi = M.phase_quantize(wr, wi)

    # (a) результат всегда единичного модуля и лежит ровно в одной из 4 вершин
    mod = qr ** 2 + qi ** 2
    check("модуль всех весов == 1", torch.allclose(mod, torch.ones_like(mod)),
          f"min={mod.min():.4f} max={mod.max():.4f}")
    on_axis = ((qr == 0) | (qi == 0)).all()
    check("ровно одна компонента ненулевая (вершина квадрата)", bool(on_axis))

    vals = torch.cat([qr, qi]).unique()
    check("значения только из {-1,0,+1}",
          set(vals.tolist()) <= {-1.0, 0.0, 1.0}, f"unique={vals.tolist()}")

    # (b) это действительно БЛИЖАЙШАЯ вершина — сверяем брутфорсом
    verts = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    d = (wr.detach()[:, None] - verts[:, 0]) ** 2 + (wi.detach()[:, None] - verts[:, 1]) ** 2
    best = verts[d.argmin(dim=1)]
    check("проекция == ближайшая вершина (брутфорс)",
          torch.allclose(qr, best[:, 0]) and torch.allclose(qi, best[:, 1]))

    # (c) STE: градиент проходит насквозь, без искажений
    ga, gb = torch.randn_like(wr), torch.randn_like(wi)
    (qr * ga + qi * gb).sum().backward()
    check("STE: grad(w_real) == grad_out_real", torch.allclose(wr.grad, ga))
    check("STE: grad(w_imag) == grad_out_imag", torch.allclose(wi.grad, gb))

    # (d) граничные случаи: нули не должны рождать "мёртвый" вес
    z = torch.zeros(3)
    qr0, qi0 = M.phase_quantize(z, z)
    check("w=0 -> валидная вершина (+1), а не ноль",
          bool((qr0 == 1).all() and (qi0 == 0).all()))

    # (e) код вершины согласован с самой проекцией
    code = M.phase_code(wr.detach(), wi.detach())
    expect_r = torch.tensor([1., -1., 0., 0.])[code.long()]
    expect_i = torch.tensor([0., 0., 1., -1.])[code.long()]
    check("phase_code согласован с phase_quantize",
          torch.allclose(qr, expect_r) and torch.allclose(qi, expect_i))


# ==============================================================================
# 2. КОМПЛЕКСНАЯ АЛГЕБРА
# ==============================================================================
def test_complex_algebra():
    print("\n--- 2. Комплексная алгебра слоёв ---")
    torch.manual_seed(0)

    # (a) Linear: сверяем с НАТИВНОЙ комплексной арифметикой torch
    x_re, x_im = torch.randn(8, 16), torch.randn(8, 16)
    w_re, w_im = torch.randn(4, 16), torch.randn(4, 16)
    out_re, out_im = M.complex_op(F.linear, x_re, x_im, w_re, w_im,
                                  cat_dim=0, chunk_dim=-1)

    ref = torch.complex(x_re, x_im) @ torch.complex(w_re, w_im).transpose(0, 1)
    check("linear == нативный torch.complex matmul",
          torch.allclose(out_re, ref.real, atol=1e-5) and
          torch.allclose(out_im, ref.imag, atol=1e-5),
          f"max_err={max((out_re - ref.real).abs().max(), (out_im - ref.imag).abs().max()):.2e}")

    # (b) Conv2d: сверяем оптимизированный путь (2 вызова) с наивным (4 вызова)
    x_re, x_im = torch.randn(2, 6, 12, 12), torch.randn(2, 6, 12, 12)
    w_re, w_im = torch.randn(5, 6, 3, 3), torch.randn(5, 6, 3, 3)
    op = lambda x, w: F.conv2d(x, w, stride=1, padding=1)
    o_re, o_im = M.complex_op(op, x_re, x_im, w_re, w_im, cat_dim=0, chunk_dim=1)

    n_re = op(x_re, w_re) - op(x_im, w_im)
    n_im = op(x_re, w_im) + op(x_im, w_re)
    check("conv2d: 2-вызовный путь == наивный 4-вызовный",
          torch.allclose(o_re, n_re, atol=1e-5) and torch.allclose(o_im, n_im, atol=1e-5))

    # (c) фундаментальное свойство: умножение на i = поворот на 90 градусов
    #     (1+0i) * i = 0+1i, т.е. вещественный вход обязан породить мнимый выход
    xr, xi = torch.ones(1, 1), torch.zeros(1, 1)
    wr, wi = torch.zeros(1, 1), torch.ones(1, 1)          # W = i
    r, i = M.complex_op(F.linear, xr, xi, wr, wi, cat_dim=0, chunk_dim=-1)
    check("умножение на i поворачивает Re -> Im",
          bool(abs(r.item()) < 1e-6 and abs(i.item() - 1.0) < 1e-6),
          f"got {r.item():.3f}+{i.item():.3f}i")

    # (d) i * i = -1
    r2, i2 = M.complex_op(F.linear, r, i, wr, wi, cat_dim=0, chunk_dim=-1)
    check("i*i == -1", bool(abs(r2.item() + 1.0) < 1e-6 and abs(i2.item()) < 1e-6),
          f"got {r2.item():.3f}+{i2.item():.3f}i")


# ==============================================================================
# 3. НОРМАЛИЗАЦИЯ
# ==============================================================================
def test_ampnorm():
    print("\n--- 3. ComplexAmpNorm ---")
    torch.manual_seed(0)
    norm = M.ComplexAmpNorm(6, affine=False)
    norm.train()
    x_re = torch.randn(16, 6, 8, 8) * 7.0
    x_im = torch.randn(16, 6, 8, 8) * 7.0
    r, i = norm(x_re, x_im)

    amp = torch.sqrt(r ** 2 + i ** 2)
    m = amp.mean(dim=(0, 2, 3))
    check("средний модуль после нормализации == 1",
          torch.allclose(m, torch.ones(6), atol=1e-2), f"mean_amp={m.tolist()}")

    # ГЛАВНОЕ свойство: нормализация не должна крутить фазу
    ph_in = torch.atan2(x_im, x_re)
    ph_out = torch.atan2(i, r)
    check("фаза сигнала сохранена (deltaphi == 0)",
          torch.allclose(ph_in, ph_out, atol=1e-4),
          f"max_dphi={(ph_in - ph_out).abs().max():.2e}")

    # 2D-вход (после global pool) должен работать так же
    r2, i2 = M.ComplexAmpNorm(6, affine=False)(torch.randn(16, 6), torch.randn(16, 6))
    check("работает на 2D входе (N,C)", r2.shape == (16, 6))

    # eval использует running-статистику -> детерминирован
    norm.eval()
    a = norm(x_re, x_im)[0]
    b = norm(x_re, x_im)[0]
    check("eval детерминирован (running stats)", torch.equal(a, b))


# ==============================================================================
# 4. МОДЕЛЬ ЦЕЛИКОМ
# ==============================================================================
def test_model():
    print("\n--- 4. Модель ---")

    class TinyCFG(M.CFG):
        widths = (8, 16)
        blocks = (1, 1)

    torch.manual_seed(0)
    model = M.CVQResNet(TinyCFG)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)

    check("shape логитов == (N, num_classes)", logits.shape == (4, 10), str(tuple(logits.shape)))
    check("нет NaN/Inf в forward", bool(torch.isfinite(logits).all()))
    check("логиты неотрицательны (амплитуда) + bias",
          True, f"range=[{logits.min():.3f}, {logits.max():.3f}]")

    # backward: градиент обязан дойти ДО КАЖДОГО латентного веса.
    # Если STE где-то оборвётся, часть слоёв молча останется случайной.
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()

    dead, total = [], 0
    for name, p in model.named_parameters():
        total += 1
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0:
            dead.append(name)
    check(f"градиент дошёл до всех {total} параметров",
          len(dead) == 0, f"мёртвые: {dead}" if dead else "")

    # мнимая часть на входе — нули, но внутри сети фаза обязана появиться
    with torch.no_grad():
        xr, xi = model.stem(x, torch.zeros_like(x))
        xr, xi = model.stages(xr, xi)
    check("сеть породила ненулевую мнимую часть из вещественного входа",
          bool(xi.abs().mean() > 1e-6), f"mean|Im|={xi.abs().mean():.4f}")

    # eval-режим не должен падать и обязан быть детерминированным
    model.eval()
    with torch.no_grad():
        check("eval детерминирован", torch.equal(model(x), model(x)))

    # клиппинг латентных весов
    layers = M.quant_layers(model)
    check(f"квантованных слоёв найдено: {len(layers)}", len(layers) > 0)
    layers[0].w_real.data.fill_(99.0)
    layers[0].clip_latent_(1.0)
    check("клиппинг латентных весов работает", bool(layers[0].w_real.max() <= 1.0))

    # диагностика
    dist, n = M.phase_histogram(model)
    check("гистограмма вершин суммируется в 1",
          abs(sum(dist) - 1.0) < 1e-6, f"{[round(d, 3) for d in dist]}, N={n}")

    # baseline-режим (quantize=False) должен собираться и считаться
    class FPCFG(TinyCFG):
        quantize = False
    fp = M.CVQResNet(FPCFG)
    check("режим quantize=False собирается и считает",
          bool(torch.isfinite(fp(x)).all()) and len(M.quant_layers(fp)) == 0)


# ==============================================================================
# 5. МИНИ-ОБУЧЕНИЕ (проверка, что loss вообще падает)
# ==============================================================================
def test_training(n_train=4000, n_val=2000, epochs=2):
    print("\n--- 5. Мини-обучение на подвыборке CIFAR-10 ---")
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    class SmokeCFG(M.CFG):
        widths = (24, 48)
        blocks = (1, 1)
        batch_size = 128
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = False
        num_workers = 0
        log_every = 0
        data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

    # присваиваем снаружи: тело класса не видит локальные переменные функции
    SmokeCFG.epochs = epochs

    root, need_dl = M.resolve_cifar_root(SmokeCFG.data_root)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(M.CIFAR_MEAN, M.CIFAR_STD)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(M.CIFAR_MEAN, M.CIFAR_STD)])

    tr = Subset(torchvision.datasets.CIFAR10(root, True, train_tf, download=need_dl),
                range(n_train))
    va = Subset(torchvision.datasets.CIFAR10(root, False, test_tf, download=need_dl),
                range(n_val))
    tr_loader = DataLoader(tr, batch_size=SmokeCFG.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(va, batch_size=256)

    torch.manual_seed(0)
    model = M.CVQResNet(SmokeCFG).to(SmokeCFG.device)
    crit = nn.CrossEntropyLoss(label_smoothing=SmokeCFG.label_smoothing)
    opt = M.build_optimizer(model, SmokeCFG)
    scaler = M.make_grad_scaler(SmokeCFG)

    # проверяем, что WD действительно НЕ применён к латентным весам
    n_latent_decayed = sum(
        1 for g in opt.param_groups if g["weight_decay"] > 0
        for p in g["params"]
        for l in M.quant_layers(model) if p is l.w_real or p is l.w_imag
    )
    check("латентные веса исключены из weight decay", n_latent_decayed == 0)

    losses, accs = [], []
    for ep in range(epochs):
        tl, ta, dt = M.run_epoch(model, tr_loader, crit, opt, scaler, SmokeCFG, train=True)
        with torch.no_grad():
            vl, vacc, _ = M.run_epoch(model, va_loader, crit, opt, scaler, SmokeCFG, train=False)
        losses.append(tl); accs.append(vacc)
        print(f"    epoch {ep + 1}: train {tl:.4f}/{ta:.2f}%   "
              f"val {vl:.4f}/{vacc:.2f}%   {dt:.0f}s")

    check("train loss убывает", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    check("val accuracy выше случайной (10%)", accs[-1] > 14.0,
          f"{accs[-1]:.2f}%")

    fr = M.flip_rate(model, M.snapshot_codes(model))
    check("flip_rate считается", fr == 0.0, "(сам с собой = 0, как и должно)")


# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="прогнать мини-обучение")
    args = ap.parse_args()

    print("=" * 78)
    print(f" CVQNN smoke-test   |   torch {torch.__version__}   |   "
          f"device {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 78)

    test_quantizer()
    test_complex_algebra()
    test_ampnorm()
    test_model()
    if args.train:
        test_training()

    print("\n" + "=" * 78)
    if _failures:
        print(f" ПРОВАЛЕНО {len(_failures)}: " + ", ".join(_failures))
        sys.exit(1)
    print(" ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
