"""
Run the notebook's visualization logic on CPU against a small real model.

Why: a notebook cannot be "compiled". A syntax check says nothing about whether
shapes line up, whether a hook returns a tuple, or whether plotly survives the
array it is handed. Here the same code runs against an actual model, only a
tiny one, so shape bugs surface in seconds rather than twelve minutes into a
Colab training run.

    python test_viz_logic.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cvqnn_cifar10 as M

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA

OK, FAIL = "  [OK]  ", "  [FAIL]"
fails = []


def check(name, fn):
    try:
        info = fn()
        print(f"{OK} {name}" + (f"   {info}" if info else ""))
    except Exception as e:
        print(f"{FAIL} {name}   {type(e).__name__}: {e}")
        fails.append(name)


# ------------------------------------------------------------------ setup
class TinyCFG(M.CFG):
    widths = (8, 16, 24)
    blocks = (1, 1, 1)
    quantize = True


torch.manual_seed(0)
device = "cpu"
TinyCFG.device = device
model = M.CVQResNet(TinyCFG).to(device).eval()
layers = M.quant_layers(model)

VERTEX = ["+1", "-1", "+i", "-i"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
rng = np.random.default_rng(0)

print("=" * 74)
print(f" visualization logic check | layers: {len(layers)} | plotly ok")
print("=" * 74)


# ---------------------------------------------- 4. latent weights in 3D
def t_latent_3d():
    fig = go.Figure()
    for vi, (vname, vcol) in enumerate(zip(VERTEX, COLORS)):
        xs, ys, zs = [], [], []
        for li, layer in enumerate(layers):
            wr = layer.w_real.detach().flatten().cpu().numpy()
            wi = layer.w_imag.detach().flatten().cpu().numpy()
            code_ = M.phase_code(layer.w_real, layer.w_imag).flatten().cpu().numpy()
            scale = max(np.abs(wr).max(), np.abs(wi).max()) + 1e-9
            sel = np.where(code_ == vi)[0]
            xs.append(wr[sel] / scale)
            ys.append(wi[sel] / scale)
            zs.append(np.full(len(sel), li, dtype=float))
        fig.add_trace(go.Scatter3d(x=np.concatenate(xs), y=np.concatenate(ys),
                                   z=np.concatenate(zs), mode="markers", name=vname))
    # this is where type errors surface: plotly serialises lazily
    fig.to_json()
    n = sum(len(t.x) for t in fig.data)
    return f"points: {n}, traces: {len(fig.data)}"


# ------------------------------------------------------------- 5. margin
def t_margin():
    medians = []
    for layer in layers:
        wr = layer.w_real.detach().flatten().cpu().numpy()
        wi = layer.w_imag.detach().flatten().cpu().numpy()
        margin = np.abs(np.abs(wr) - np.abs(wi)) / (np.abs(wr) + np.abs(wi) + 1e-12)
        assert np.all((margin >= 0) & (margin <= 1)), "margin escaped [0,1]"
        medians.append(np.median(margin))
    go.Figure([go.Violin(y=[0.1, 0.2, 0.3])]).to_json()
    return f"per-layer medians: {np.round(medians, 2).tolist()}"


# -------------------------------------------------- 6. corner distribution
def t_vertex_dist():
    per_layer = np.zeros((len(layers), 4))
    for li, layer in enumerate(layers):
        c = M.phase_code(layer.w_real, layer.w_imag).flatten().cpu().numpy()
        per_layer[li] = np.bincount(c, minlength=4) / c.size
    assert np.allclose(per_layer.sum(axis=1), 1.0), "shares do not sum to 1"
    total = per_layer.mean(axis=0)
    return f"network: {', '.join(f'{v}={p*100:.0f}%' for v, p in zip(VERTEX, total))}"


# ------------------------------------------------ 7. hooks and trajectories
acts = []


def t_hooks():
    hooks = []

    def _hook(name):
        def fn(mod, inp, out):
            acts.append((name, out[0].detach().float().cpu(),
                         out[1].detach().float().cpu()))
        return fn

    for name, mod in model.named_modules():
        if isinstance(mod, M.ComplexAmpNorm):
            hooks.append(mod.register_forward_hook(_hook(name)))

    x_one = torch.randn(1, 3, 32, 32)
    acts.clear()
    with torch.no_grad():
        _ = model(x_one)
    for h in hooks:
        h.remove()
    assert acts, "hooks did not fire"
    return f"layers captured: {len(acts)}"


def t_trajectories():
    traj_re, traj_im = [], []
    for nm, re_, im_ in acts:
        r = re_[0].mean(dim=(1, 2)).numpy() if re_.dim() == 4 else re_[0].numpy()
        i = im_[0].mean(dim=(1, 2)).numpy() if im_.dim() == 4 else im_[0].numpy()
        traj_re.append(r)
        traj_im.append(i)

    groups, cur = [], [0]
    for li in range(1, len(traj_re)):
        if len(traj_re[li]) == len(traj_re[li - 1]):
            cur.append(li)
        else:
            groups.append(cur)
            cur = [li]
    groups.append(cur)
    groups = [g for g in groups if len(g) > 1]

    # the point of the grouping: width must be constant inside a group,
    # otherwise a channel index would denote different features
    for g in groups:
        widths = {len(traj_re[l]) for l in g}
        assert len(widths) == 1, f"group {g} has mixed widths: {widths}"

    fig = go.Figure()
    for gi, g in enumerate(groups):
        n_ch = len(traj_re[g[0]])
        amp = np.sqrt(traj_re[g[-1]] ** 2 + traj_im[g[-1]] ** 2)
        for ci in np.argsort(-amp)[:min(8, n_ch)]:
            fig.add_trace(go.Scatter3d(
                x=[traj_re[l][ci] for l in g], y=[traj_im[l][ci] for l in g],
                z=[float(l) for l in g], mode="lines"))
    fig.to_json()
    return f"groups: {len(groups)} {[(g[0], g[-1], len(traj_re[g[0]])) for g in groups]}"


# ------------------------------------------------------ 8. domain colouring
def t_domain_coloring():
    idx = min(3, len(acts) - 1)
    name, re_t, im_t = acts[idx]
    assert re_t.dim() == 4, f"expected a feature map, got {re_t.shape}"
    re_np, im_np = re_t[0].numpy(), im_t[0].numpy()

    def domain_rgb(re, im):
        amp = np.sqrt(re ** 2 + im ** 2)
        ang = np.arctan2(im, re)
        h = (ang % (2 * np.pi)) / (2 * np.pi)
        v = amp / (amp.max() + 1e-9)
        hsv = np.stack([h, np.ones_like(h), v], axis=-1)
        return (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)

    energy = (re_np ** 2 + im_np ** 2).sum(axis=(1, 2))
    top = np.argsort(-energy)[:min(8, len(energy))]

    fig = make_subplots(rows=2, cols=4)
    for k, c in enumerate(top):
        rgb = domain_rgb(re_np[c], im_np[c])
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"bad RGB shape: {rgb.shape}"
        fig.add_trace(go.Image(z=rgb), row=k // 4 + 1, col=k % 4 + 1)
    fig.to_json()
    return f"layer {name}, channels shown: {len(top)}"


# --------------------------------------------------------- 9. complex logits
def t_complex_logits():
    xb = torch.randn(32, 3, 32, 32)
    yb = torch.randint(0, 10, (32,))
    with torch.no_grad():
        r, i = model.stem(xb, torch.zeros_like(xb))
        r, i = model.stages(r, i)
        r, i = r.mean(dim=(2, 3)), i.mean(dim=(2, 3))
        orr, oii = model.head(r, i)
    assert orr.shape == (32, 10), f"logit shape {orr.shape}"
    lab = yb.numpy()
    ang = np.arctan2(oii.numpy()[np.arange(len(lab)), lab],
                     orr.numpy()[np.arange(len(lab)), lab])
    assert np.isfinite(ang).all()
    return f"logits {tuple(orr.shape)}, phase computes"


# -------------------------------------------------------------- 10. PCA 3D
def t_pca():
    xb = torch.randn(64, 3, 32, 32)
    with torch.no_grad():
        r, i = model.stem(xb, torch.zeros_like(xb))
        r, i = model.stages(r, i)
        X = np.concatenate([r.mean(dim=(2, 3)).numpy(), i.mean(dim=(2, 3)).numpy()],
                           axis=1)
    p3 = PCA(n_components=3).fit(X)
    Z = p3.transform(X)
    assert Z.shape == (64, 3)
    go.Figure([go.Scatter3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2], mode="markers")]).to_json()
    return f"features: {X.shape[1]}, explained: {p3.explained_variance_ratio_.sum()*100:.0f}%"


# ------------------------------------- 7a. density surface over the plane
def t_density_surface():
    BINS = 31
    surfaces = []
    for li, layer in enumerate(layers):
        wr = layer.w_real.detach().flatten().cpu().numpy()
        wi = layer.w_imag.detach().flatten().cpu().numpy()
        scale = max(np.abs(wr).max(), np.abs(wi).max()) + 1e-9
        H, xe, ye = np.histogram2d(wr / scale, wi / scale, bins=BINS,
                                   range=[[-1, 1], [-1, 1]])
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        Z = H.T
        # Surface expects z[row=y][col=x]; histogram2d returns [x, y]
        assert Z.shape == (len(yc), len(xc)), f"transposed wrong: {Z.shape}"
        assert Z.sum() == wr.size, "histogram lost weights"
        surfaces.append(go.Surface(x=xc, y=yc, z=Z, visible=(li == 0)))
    fig = go.Figure(data=surfaces)
    buttons = [dict(label=f"layer {li}", method="update",
                    args=[{"visible": [i == li for i in range(len(layers))]}])
               for li in range(len(layers))]
    fig.update_layout(updatemenus=[dict(buttons=buttons)])
    fig.to_json()
    return f"{len(surfaces)} surfaces, {BINS}x{BINS} bins"


# ----------------------------------------------- 7b. codebook rings in 3D
def t_codebook_rings():
    CORNER_XY = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    theta = np.linspace(0, 2 * np.pi, 121)
    fig = go.Figure()
    for li in range(len(layers)):
        fig.add_trace(go.Scatter3d(x=np.cos(theta), y=np.sin(theta),
                                   z=np.full_like(theta, li), mode="lines"))
    total_share = np.zeros(4)
    for vi in range(4):
        sizes = []
        for layer in layers:
            c = M.phase_code(layer.w_real, layer.w_imag).flatten().cpu().numpy()
            share = (c == vi).mean()
            total_share[vi] += share / len(layers)
            sizes.append(6 + 46 * np.sqrt(share))
        assert all(s > 0 for s in sizes), "non-positive marker size"
        fig.add_trace(go.Scatter3d(
            x=[CORNER_XY[vi][0]] * len(layers), y=[CORNER_XY[vi][1]] * len(layers),
            z=list(range(len(layers))), mode="markers",
            marker=dict(size=sizes)))
    fig.to_json()
    assert abs(total_share.sum() - 1.0) < 1e-6, "shares do not sum to 1"
    return f"rings: {len(layers)}, markers: 4 per ring"


# ------------------------------------------ 7c. init vs trained comparison
def t_init_vs_trained():
    class Fresh(TinyCFG):
        pass

    torch.manual_seed(TinyCFG.seed)
    fresh = M.CVQResNet(Fresh)

    def corner_shares(ls):
        out = np.zeros(4)
        n = 0
        for l in ls:
            c = M.phase_code(l.w_real, l.w_imag).flatten().cpu().numpy()
            out += np.bincount(c, minlength=4)
            n += c.size
        return out / n

    a = corner_shares(M.quant_layers(fresh))
    b = corner_shares(layers)
    assert abs(a.sum() - 1) < 1e-9 and abs(b.sum() - 1) < 1e-9
    drift = np.abs(b - a).sum() / 2
    go.Figure([go.Bar(x=["+1", "-1", "+i", "-i"], y=a * 100),
               go.Bar(x=["+1", "-1", "+i", "-i"], y=b * 100)]).to_json()
    return f"total variation distance: {drift*100:.2f}%"


for nm, fn in [
    ("latent weights in the phase square (3D)", t_latent_3d),
    ("margin to the decision boundary", t_margin),
    ("corner distribution", t_vertex_dist),
    ("density surface over the complex plane", t_density_surface),
    ("trained codebook as rings in 3D", t_codebook_rings),
    ("initialization vs trained corner shares", t_init_vs_trained),
    ("hooks on the normalization layers", t_hooks),
    ("trajectories within stages", t_trajectories),
    ("domain colouring of feature maps", t_domain_coloring),
    ("complex logits", t_complex_logits),
    ("PCA of features in 3D", t_pca),
]:
    check(nm, fn)

print("=" * 74)
if fails:
    print(f" FAILED {len(fails)}: {', '.join(fails)}")
    sys.exit(1)
print(" ALL VISUALIZATIONS RAN")
