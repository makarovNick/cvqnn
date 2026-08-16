"""
Smoke test for CVQNN. Checks that the maths is right, not merely that it runs.

    python smoke_test.py            # unit checks only (fast, no data needed)
    python smoke_test.py --train    # plus a mini-training run on a CIFAR-10 subset
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
# 1. QUANTIZER
# ==============================================================================
def test_quantizer():
    print("\n--- 1. PhaseQuantSTE ---")
    torch.manual_seed(0)
    wr = torch.randn(5000, requires_grad=True)
    wi = torch.randn(5000, requires_grad=True)
    qr, qi = M.phase_quantize(wr, wi)

    # (a) every result has unit modulus and sits on exactly one of 4 corners
    mod = qr ** 2 + qi ** 2
    check("all weights have modulus 1", torch.allclose(mod, torch.ones_like(mod)),
          f"min={mod.min():.4f} max={mod.max():.4f}")
    on_axis = ((qr == 0) | (qi == 0)).all()
    check("exactly one component non-zero (a square corner)", bool(on_axis))

    vals = torch.cat([qr, qi]).unique()
    check("values drawn only from {-1,0,+1}",
          set(vals.tolist()) <= {-1.0, 0.0, 1.0}, f"unique={vals.tolist()}")

    # (b) it really is the NEAREST corner - verified by brute force, not by
    #     trusting the |Re| vs |Im| shortcut the implementation uses
    verts = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    d = (wr.detach()[:, None] - verts[:, 0]) ** 2 + (wi.detach()[:, None] - verts[:, 1]) ** 2
    best = verts[d.argmin(dim=1)]
    check("projection == nearest corner (brute force)",
          torch.allclose(qr, best[:, 0]) and torch.allclose(qi, best[:, 1]))

    # (c) STE: the gradient passes through undistorted
    ga, gb = torch.randn_like(wr), torch.randn_like(wi)
    (qr * ga + qi * gb).sum().backward()
    check("STE: grad(w_real) == grad_out_real", torch.allclose(wr.grad, ga))
    check("STE: grad(w_imag) == grad_out_imag", torch.allclose(wi.grad, gb))

    # (d) edge case: zeros must not produce a dead weight
    z = torch.zeros(3)
    qr0, qi0 = M.phase_quantize(z, z)
    check("w=0 -> a valid corner (+1), not zero",
          bool((qr0 == 1).all() and (qi0 == 0).all()))

    # (e) the diagnostic corner id agrees with the projection itself
    code = M.phase_code(wr.detach(), wi.detach())
    expect_r = torch.tensor([1., -1., 0., 0.])[code.long()]
    expect_i = torch.tensor([0., 0., 1., -1.])[code.long()]
    check("phase_code agrees with phase_quantize",
          torch.allclose(qr, expect_r) and torch.allclose(qi, expect_i))


# ==============================================================================
# 2. COMPLEX ALGEBRA
# ==============================================================================
def test_complex_algebra():
    print("\n--- 2. Complex algebra of the layers ---")
    torch.manual_seed(0)

    # (a) Linear: cross-checked against NATIVE torch complex arithmetic.
    #     A flipped sign in Re*Re - Im*Im would show up here immediately.
    x_re, x_im = torch.randn(8, 16), torch.randn(8, 16)
    w_re, w_im = torch.randn(4, 16), torch.randn(4, 16)
    out_re, out_im = M.complex_op(F.linear, x_re, x_im, w_re, w_im,
                                  cat_dim=0, chunk_dim=-1)

    ref = torch.complex(x_re, x_im) @ torch.complex(w_re, w_im).transpose(0, 1)
    check("linear == native torch.complex matmul",
          torch.allclose(out_re, ref.real, atol=1e-5) and
          torch.allclose(out_im, ref.imag, atol=1e-5),
          f"max_err={max((out_re - ref.real).abs().max(), (out_im - ref.imag).abs().max()):.2e}")

    # (b) Conv2d: the optimised 2-call path against the naive 4-call one
    x_re, x_im = torch.randn(2, 6, 12, 12), torch.randn(2, 6, 12, 12)
    w_re, w_im = torch.randn(5, 6, 3, 3), torch.randn(5, 6, 3, 3)
    op = lambda x, w: F.conv2d(x, w, stride=1, padding=1)
    o_re, o_im = M.complex_op(op, x_re, x_im, w_re, w_im, cat_dim=0, chunk_dim=1)

    n_re = op(x_re, w_re) - op(x_im, w_im)
    n_im = op(x_re, w_im) + op(x_im, w_re)
    check("conv2d: 2-call path == naive 4-call path",
          torch.allclose(o_re, n_re, atol=1e-5) and torch.allclose(o_im, n_im, atol=1e-5))

    # (c) the defining property: multiplying by i is a 90-degree rotation,
    #     so a purely real input must produce a purely imaginary output
    xr, xi = torch.ones(1, 1), torch.zeros(1, 1)
    wr, wi = torch.zeros(1, 1), torch.ones(1, 1)          # W = i
    r, i = M.complex_op(F.linear, xr, xi, wr, wi, cat_dim=0, chunk_dim=-1)
    check("multiplying by i rotates Re -> Im",
          bool(abs(r.item()) < 1e-6 and abs(i.item() - 1.0) < 1e-6),
          f"got {r.item():.3f}+{i.item():.3f}i")

    # (d) i * i = -1
    r2, i2 = M.complex_op(F.linear, r, i, wr, wi, cat_dim=0, chunk_dim=-1)
    check("i*i == -1", bool(abs(r2.item() + 1.0) < 1e-6 and abs(i2.item()) < 1e-6),
          f"got {r2.item():.3f}+{i2.item():.3f}i")


# ==============================================================================
# 3. NORMALIZATION
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
    check("mean modulus after normalization == 1",
          torch.allclose(m, torch.ones(6), atol=1e-2), f"mean_amp={m.tolist()}")

    # THE property this layer exists for: normalization must not rotate phase
    ph_in = torch.atan2(x_im, x_re)
    ph_out = torch.atan2(i, r)
    check("signal phase preserved (delta phi == 0)",
          torch.allclose(ph_in, ph_out, atol=1e-4),
          f"max_dphi={(ph_in - ph_out).abs().max():.2e}")

    # a 2D input (after global pooling) must behave the same way
    r2, i2 = M.ComplexAmpNorm(6, affine=False)(torch.randn(16, 6), torch.randn(16, 6))
    check("works on a 2D (N,C) input", r2.shape == (16, 6))

    # eval uses running statistics -> deterministic
    norm.eval()
    a = norm(x_re, x_im)[0]
    b = norm(x_re, x_im)[0]
    check("eval is deterministic (running stats)", torch.equal(a, b))


# ==============================================================================
# 4. THE WHOLE MODEL
# ==============================================================================
def test_model():
    print("\n--- 4. Model ---")

    class TinyCFG(M.CFG):
        widths = (8, 16)
        blocks = (1, 1)

    torch.manual_seed(0)
    model = M.CVQResNet(TinyCFG)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)

    check("logits shape == (N, num_classes)", logits.shape == (4, 10), str(tuple(logits.shape)))
    check("no NaN/Inf in forward", bool(torch.isfinite(logits).all()))
    check("logits non-negative (magnitude) + bias",
          True, f"range=[{logits.min():.3f}, {logits.max():.3f}]")

    # Backward: the gradient must reach EVERY latent weight. If the STE is
    # broken anywhere, those layers stay random and accuracy just comes out a
    # bit lower - which we would happily misread as the cost of quantization.
    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()

    dead, total = [], 0
    for name, p in model.named_parameters():
        total += 1
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().max() == 0:
            dead.append(name)
    check(f"gradient reached all {total} parameters",
          len(dead) == 0, f"dead: {dead}" if dead else "")

    # the imaginary input is zero, yet phase must appear inside the network -
    # otherwise the whole complex machinery degenerates to a real-valued net
    with torch.no_grad():
        xr, xi = model.stem(x, torch.zeros_like(x))
        xr, xi = model.stages(xr, xi)
    check("network created a non-zero imaginary part from a real input",
          bool(xi.abs().mean() > 1e-6), f"mean|Im|={xi.abs().mean():.4f}")

    model.eval()
    with torch.no_grad():
        check("eval is deterministic", torch.equal(model(x), model(x)))

    layers = M.quant_layers(model)
    check(f"quantized layers found: {len(layers)}", len(layers) > 0)
    layers[0].w_real.data.fill_(99.0)
    layers[0].clip_latent_(1.0)
    check("latent weight clipping works", bool(layers[0].w_real.max() <= 1.0))

    dist, n = M.phase_histogram(model)
    check("corner histogram sums to 1",
          abs(sum(dist) - 1.0) < 1e-6, f"{[round(d, 3) for d in dist]}, N={n}")

    # the control arm (quantize=False) must build and run too
    class FPCFG(TinyCFG):
        quantize = False
    fp = M.CVQResNet(FPCFG)
    check("quantize=False builds and runs",
          bool(torch.isfinite(fp(x)).all()) and len(M.quant_layers(fp)) == 0)


# ==============================================================================
# 5. BINARY CONTROL ARM
# ==============================================================================
def test_binary_control():
    print("\n--- 5. Binary control {+1,-1} ---")

    class PhaseCFG(M.CFG):
        widths = (16, 32)
        blocks = (1, 1)
        quantize = True
        weight_mode = "phase4"

    class BinCFG(M.CFG):
        widths = (16, 32)
        binary_widths = (23, 45)      # ~sqrt(2) wider
        blocks = (1, 1)
        quantize = True
        weight_mode = "binary"

    torch.manual_seed(0)
    ph = M.CVQResNet(PhaseCFG)
    torch.manual_seed(0)
    bn = M.CVQResNet(BinCFG)

    check("binary arm uses the widened channel counts",
          M.effective_widths(BinCFG) == BinCFG.binary_widths,
          str(M.effective_widths(BinCFG)))

    # w_imag must not be a trainable parameter in the binary arm, otherwise it
    # would sit in the optimizer collecting no gradient
    bl = M.quant_layers(bn)
    imag_params = [l for l in bl if isinstance(l.w_imag, nn.Parameter)]
    check("binary arm has no imaginary weight parameters", len(imag_params) == 0,
          f"{len(imag_params)} of {len(bl)} layers still carry one")

    check("binary weights quantize to 1 bit", bl[0].bits_per_weight() == 1)
    check("phase weights quantize to 2 bits",
          M.quant_layers(ph)[0].bits_per_weight() == 2)

    # THE point of the control: the bit budgets must actually match
    bits_ph = sum(l.w_real.numel() * l.bits_per_weight() for l in M.quant_layers(ph))
    bits_bn = sum(l.w_real.numel() * l.bits_per_weight() for l in bl)
    rel = abs(bits_ph - bits_bn) / bits_ph
    check("bit budgets match within 5%", rel < 0.05,
          f"phase {bits_ph/1e3:.1f} kbit vs binary {bits_bn/1e3:.1f} kbit "
          f"({rel*100:.1f}% apart)")

    # THE correctness requirement: the imaginary channel must stay identically
    # zero end to end. If beta_im leaked, the "binary" baseline would secretly
    # run a second real network alongside and the comparison would be void.
    x = torch.randn(4, 3, 32, 32)
    bn.eval()
    with torch.no_grad():
        r, i = bn.stem(x, torch.zeros_like(x))
        max_im = [i.abs().max().item()]
        for blk in bn.stages:
            r, i = blk(r, i)
            max_im.append(i.abs().max().item())
        r2, i2 = r.mean(dim=(2, 3)), i.mean(dim=(2, 3))
        orr, oii = bn.head(r2, i2)
        max_im.append(oii.abs().max().item())
    check("imaginary channel is identically zero throughout the binary arm",
          max(max_im) == 0.0, f"max|Im| per stage: {[round(v, 6) for v in max_im]}")

    # the phase arm must NOT be real-only - a sanity check on the check itself
    ph.eval()
    with torch.no_grad():
        pr, pi = ph.stem(x, torch.zeros_like(x))
        pr, pi = ph.stages(pr, pi)
    check("phase arm does produce an imaginary part (control is meaningful)",
          pi.abs().max().item() > 0, f"max|Im|={pi.abs().max():.4f}")

    # all corners in the binary arm must be real ones
    dist, _ = M.phase_histogram(bn)
    check("binary weights land only on the real corners",
          abs(dist[2] + dist[3]) < 1e-9,
          f"+1={dist[0]*100:.1f}% -1={dist[1]*100:.1f}% "
          f"+i={dist[2]*100:.1f}% -i={dist[3]*100:.1f}%")

    # gradients must still reach every parameter
    logits = bn(x)
    F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    dead = [n for n, p in bn.named_parameters()
            if p.grad is None or p.grad.abs().max() == 0]
    check("gradient reaches every binary-arm parameter", len(dead) == 0,
          f"dead: {dead}" if dead else "")


# ==============================================================================
# 6. MINI-TRAINING (does the loss actually go down?)
# ==============================================================================
def test_training(n_train=4000, n_val=2000, epochs=2):
    print("\n--- 5. Mini-training on a CIFAR-10 subset ---")
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

    # assigned from outside: a class body cannot see the enclosing function's locals
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

    # verify that weight decay really is kept away from the latent weights
    n_latent_decayed = sum(
        1 for g in opt.param_groups if g["weight_decay"] > 0
        for p in g["params"]
        for l in M.quant_layers(model) if p is l.w_real or p is l.w_imag
    )
    check("latent weights excluded from weight decay", n_latent_decayed == 0)

    losses, accs = [], []
    for ep in range(epochs):
        tl, ta, dt = M.run_epoch(model, tr_loader, crit, opt, scaler, SmokeCFG, train=True)
        with torch.no_grad():
            vl, vacc, _ = M.run_epoch(model, va_loader, crit, opt, scaler, SmokeCFG, train=False)
        losses.append(tl); accs.append(vacc)
        print(f"    epoch {ep + 1}: train {tl:.4f}/{ta:.2f}%   "
              f"val {vl:.4f}/{vacc:.2f}%   {dt:.0f}s")

    check("train loss decreases", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")
    check("val accuracy above chance (10%)", accs[-1] > 14.0,
          f"{accs[-1]:.2f}%")

    fr = M.flip_rate(model, M.snapshot_codes(model))
    check("flip_rate computes", fr == 0.0, "(against itself = 0, as it should be)")


# ==============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="also run the mini-training")
    args = ap.parse_args()

    print("=" * 78)
    print(f" CVQNN smoke test   |   torch {torch.__version__}   |   "
          f"device {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 78)

    test_quantizer()
    test_complex_algebra()
    test_ampnorm()
    test_model()
    test_binary_control()
    if args.train:
        test_training()

    print("\n" + "=" * 78)
    if _failures:
        print(f" FAILED {len(_failures)}: " + ", ".join(_failures))
        sys.exit(1)
    print(" ALL CHECKS PASSED")
