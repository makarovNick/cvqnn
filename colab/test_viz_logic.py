"""
Прогон логики визуализаций из Colab-ноутбука на CPU с маленькой моделью.

Смысл: ноутбук нельзя «скомпилировать». Проверка синтаксиса ничего не говорит
о том, сойдутся ли размерности, вернёт ли хук кортеж и переживёт ли plotly
переданный массив. Здесь тот же код выполняется на настоящей модели, только
крошечной — все ошибки формы вылезают за секунды, а не на 12-й минуте
обучения в Colab.

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


# ---------------------------------------------------------------- подготовка
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
print(f" проверка логики визуализаций | слоёв: {len(layers)} | plotly ok")
print("=" * 74)


# ------------------------------------------------- 4. латентные веса в 3D
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
    # именно здесь ловятся ошибки типов: plotly сериализует всё лениво
    fig.to_json()
    n = sum(len(t.x) for t in fig.data)
    return f"точек: {n}, трейсов: {len(fig.data)}"


# ------------------------------------------------------------ 5. margin
def t_margin():
    medians = []
    for layer in layers:
        wr = layer.w_real.detach().flatten().cpu().numpy()
        wi = layer.w_imag.detach().flatten().cpu().numpy()
        margin = np.abs(np.abs(wr) - np.abs(wi)) / (np.abs(wr) + np.abs(wi) + 1e-12)
        assert np.all((margin >= 0) & (margin <= 1)), "margin вышел за [0,1]"
        medians.append(np.median(margin))
    go.Figure([go.Violin(y=[0.1, 0.2, 0.3])]).to_json()
    return f"медианы по слоям: {np.round(medians, 2).tolist()}"


# ------------------------------------------- 6. распределение по вершинам
def t_vertex_dist():
    per_layer = np.zeros((len(layers), 4))
    for li, layer in enumerate(layers):
        c = M.phase_code(layer.w_real, layer.w_imag).flatten().cpu().numpy()
        per_layer[li] = np.bincount(c, minlength=4) / c.size
    assert np.allclose(per_layer.sum(axis=1), 1.0), "доли не суммируются в 1"
    total = per_layer.mean(axis=0)
    return f"по сети: {', '.join(f'{v}={p*100:.0f}%' for v, p in zip(VERTEX, total))}"


# ------------------------------------------------- 7. хуки и траектории
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
    assert acts, "хуки не сработали"
    return f"снято слоёв: {len(acts)}"


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

    # главная проверка: внутри группы ширина обязана совпадать,
    # иначе индекс канала обозначал бы разные признаки
    for g in groups:
        widths = {len(traj_re[l]) for l in g}
        assert len(widths) == 1, f"в группе {g} разная ширина: {widths}"

    fig = go.Figure()
    for gi, g in enumerate(groups):
        n_ch = len(traj_re[g[0]])
        amp = np.sqrt(traj_re[g[-1]] ** 2 + traj_im[g[-1]] ** 2)
        for ci in np.argsort(-amp)[:min(8, n_ch)]:
            fig.add_trace(go.Scatter3d(
                x=[traj_re[l][ci] for l in g], y=[traj_im[l][ci] for l in g],
                z=[float(l) for l in g], mode="lines"))
    fig.to_json()
    return f"групп: {len(groups)} {[(g[0], g[-1], len(traj_re[g[0]])) for g in groups]}"


# ----------------------------------------------- 8. доменная раскраска
def t_domain_coloring():
    idx = min(3, len(acts) - 1)
    name, re_t, im_t = acts[idx]
    assert re_t.dim() == 4, f"ожидалась карта признаков, получено {re_t.shape}"
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
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"плохая форма RGB: {rgb.shape}"
        fig.add_trace(go.Image(z=rgb), row=k // 4 + 1, col=k % 4 + 1)
    fig.to_json()
    return f"слой {name}, каналов показано: {len(top)}"


# ---------------------------------------------- 9. комплексные логиты
def t_complex_logits():
    xb = torch.randn(32, 3, 32, 32)
    yb = torch.randint(0, 10, (32,))
    with torch.no_grad():
        r, i = model.stem(xb, torch.zeros_like(xb))
        r, i = model.stages(r, i)
        r, i = r.mean(dim=(2, 3)), i.mean(dim=(2, 3))
        orr, oii = model.head(r, i)
    assert orr.shape == (32, 10), f"форма логитов {orr.shape}"
    lab = yb.numpy()
    ang = np.arctan2(oii.numpy()[np.arange(len(lab)), lab],
                     orr.numpy()[np.arange(len(lab)), lab])
    assert np.isfinite(ang).all()
    return f"логиты {tuple(orr.shape)}, фаза считается"


# ------------------------------------------------------- 10. PCA в 3D
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
    return f"признаков: {X.shape[1]}, объяснено: {p3.explained_variance_ratio_.sum()*100:.0f}%"


for nm, fn in [
    ("латентные веса в фазовом квадрате (3D)", t_latent_3d),
    ("запас до границы решения", t_margin),
    ("распределение по вершинам", t_vertex_dist),
    ("хуки на нормализациях", t_hooks),
    ("траектории внутри стадий", t_trajectories),
    ("доменная раскраска карт признаков", t_domain_coloring),
    ("комплексные логиты", t_complex_logits),
    ("PCA признаков в 3D", t_pca),
]:
    check(nm, fn)

print("=" * 74)
if fails:
    print(f" ПРОВАЛЕНО {len(fails)}: {', '.join(fails)}")
    sys.exit(1)
print(" ВСЕ ВИЗУАЛИЗАЦИИ ОТРАБОТАЛИ")
