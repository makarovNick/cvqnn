# CVQNN — Complex-Valued Quantized Neural Network

**Every weight is pinned to one of the 4th roots of unity: `{+1, −1, +i, −i}`.**

Two bits per weight instead of thirty-two. The question this repository exists
to answer is narrow and quantitative: *what does that compression actually
cost in accuracy, and does the phase degree of freedom buy anything over plain
binary weights?*

---

## Result so far

CIFAR-10, 40 epochs, identical architecture and seed, Kaggle T4.
Artefacts in [`results/kaggle-t4-40ep/`](results/kaggle-t4-40ep).

| arm | bits per complex weight | deployed size | best val acc |
|---|---|---|---|
| quantized `{+1, −1, +i, −i}` | 2 | 0.39 MB | **89.15%** |
| complex FP32 control | 64 | 12.49 MB | 91.60% |
| | | | **gap: 2.45 pp for 32× compression** |

The ratio is 32×, not 16×: both arms hold the same 1.56M *complex* weights, but
a complex FP32 weight is `w_real + w_imag` — two float32, 64 bits — against 2
bits for a corner index. (The FP32 checkpoint on disk is 12.55 MB against 12.49
MB predicted, which confirms the accounting.)

Two diagnostics matter as much as the accuracy:

**The phase degree of freedom was not abandoned.** Final corner distribution is
`+1 = 22.2%`, `−1 = 28.4%`, `+i = 24.6%`, `−i = 24.8%` — the imaginary corners
hold 49.4% of all weights. The network did not collapse into a binary one.

**Quantization converged.** The flip rate — the fraction of weights changing
corner per epoch — decayed `20.4% → 9.5% → 5.4% → 0.05%` over 40 epochs. By the
end the configuration is genuinely discrete and stable, so the reported
accuracy belongs to a network that still exists after the last step rather than
to a momentary state.

### Does phase beat sign? Yes, by 1.66 pp

The control that settles it: a real-valued binary `{+1, −1}` network widened by
√2 so that both arms occupy the **same number of bits**. A same-width binary
network would simply have had half the storage, which is not a comparison.
Artefacts in [`results/phase-vs-binary-40ep/`](results/phase-vs-binary-40ep).

| arm | shape | bits/weight | best val acc |
|---|---|---|---|
| phase `{+1,−1,+i,−i}` | (48, 96, 192) | 2 | **88.60%** |
| binary `{+1,−1}`, widened | (68, 136, 272) | 1 | 86.94% |
| | | | **phase advantage: +1.66 pp** |

At a fixed memory budget it is better to spend it on a larger codebook per
weight than on more weights. The binary arm confirmed it was genuinely real:
0.0% of its weights landed on an imaginary corner.

Two things worth noting beyond the headline. Binary's flip rate collapses to
~5% after one epoch while phase sits at ~19% and decays slowly — with one
decision boundary instead of two diagonals, binary weights have far less to
argue about. And binary ends with a smaller train–val gap (+3.47 pp against
+6.34), i.e. it underfits rather than overfits, so a longer schedule would
likely favour it somewhat.

**How solid is 1.66 pp?** The phase arm scored 89.15% in the first experiment
and 88.60% here on an identical configuration and seed — a 0.55 pp spread from
GPU non-determinism alone (`cudnn.benchmark`, non-deterministic float
reductions). So the advantage is roughly three times the observed noise, from a
single seed per arm. Suggestive, and consistent with the flip-rate and corner
evidence, but two or three seeds would be needed to call it settled.

### What this does *not* yet show

The 49.4% imaginary share, taken alone, proves less than it appears to. Weight
initialization draws `w_real` and `w_imag` i.i.d., so by symmetry roughly half
the weights start on an imaginary corner anyway: a balanced histogram is the
*default state*, not a finding. It shows the phase corners were not abandoned,
not that they were earned — which is why the binary control above exists, and
why the notebook plots initialization against the trained state rather than the
trained state alone.

Still open: every number here comes from a **single seed per arm**, and both
experiments share one architecture family, one dataset and one schedule. The
measured 0.55 pp run-to-run spread bounds how much weight any single gap can
carry.

---

## The idea

A binary network stores a sign. This one stores a **corner of the phase
square** — still 2 bits (log₂ 4), but with an extra property: multiplying by
`i` is a 90° rotation that **mixes the real and imaginary channels** of the
signal. Information that a binary network can only scale, this one can also
rotate.

Training runs on latent FP32 pairs `(w_real, w_imag)`. The forward pass
collapses each pair onto the nearest corner; the gradient passes straight
through to the latent copy (Straight-Through Estimator).

The projection rule is *larger magnitude wins*:

```
|Re| ≥ |Im|  →  q = sign(Re) · 1     (real axis)
|Re| <  |Im|  →  q = sign(Im) · i     (imaginary axis)
```

This is genuinely the nearest corner, not a heuristic: the decision boundaries
are the diagonals `Re = ±Im`, which are exactly the bisectors between adjacent
corners. `smoke_test.py` verifies this by brute force rather than trusting the
shortcut.

---

## Layout

| path | what it is |
|---|---|
| `cvqnn_cifar10.py` | everything: layers, model, training loop, A/B experiment, plots |
| `smoke_test.py` | maths checks + mini-training — run this before spending GPU time |
| `colab/cvqnn_visualization.ipynb` | interactive 3D visualization of a trained network |
| `build_colab_notebook.py` | generator for that notebook (do not edit the `.ipynb` by hand) |
| `colab/validate_notebook.py` | parses every cell and checks cross-cell names |
| `colab/test_viz_logic.py` | runs the visualization code against a real model |
| `kaggle/` | script-kernel metadata, push and status-polling helpers |

---

## Implementation decisions that matter

These are not stylistic. Quantized networks are unusually sensitive to each of
them, and without them the network simply does not learn.

**Weight decay is not applied to the latent weights.** The projection is
scale-invariant — only signs and the ratio `|Re| : |Im|` matter — so weight
decay regularizes nothing. All it does is drag weights toward zero, into the
region where the corner decision is noisiest and weights start flickering.

**Latent weights are clipped to `[−1, 1]` after every step.** Without it they
drift toward ±∞, the corner decision freezes, and the layer stops learning
while still looking healthy in the loss curve.

**Normalization divides `Re` and `Im` by a shared modulus.** This preserves
phase: it changes the length of the vector without rotating it. A plain
BatchNorm applied to `Re` and `Im` separately would destroy the very quantity
the experiment is about.

**Complex multiplication uses 2 kernel launches, not 4.** Weights are
concatenated as `[W_re; W_im]` along the output axis. Identical FLOPs, half the
launches — which is what dominates on small convolutions.

**The magnitude output gets a learnable temperature.** Logits are `|z| ≥ 0`
with a narrow dynamic range; without a temperature and per-class shift the
softmax stays nearly flat.

---

## Diagnostics

Beyond loss and accuracy, two quantities specific to this experiment are
recorded every epoch:

**Corner distribution.** If the imaginary corners `+i`/`−i` sit empty, the
network has degenerated into an ordinary binary one — the phase hypothesis
failed, regardless of what the accuracy says.

**Flip rate** — the fraction of weights that changed corner during the epoch.
If it does not decay toward the end of training, the network never settled into
a discrete configuration; it is still rattling around the decision boundaries,
and the reported accuracy belongs to a configuration that no longer exists.

The Colab notebook adds a third: the **margin to the decision boundary**,
`||Re| − |Im|| / (|Re| + |Im|)`, per layer. A median near zero identifies
exactly which layers failed to converge.

---

## Experiment protocol

Absolute accuracy on its own says nothing. The result is the **delta** between
two runs that share a seed and an architecture and differ in one bit of config:

- `quantize = True` — weights in `{±1, ±i}`, 2 bits each;
- `quantize = False` — plain complex FP32 weights, 32 bits each.

`CFG.mode = "both"` runs the pair and prints the gap. That number — accuracy
points lost per 16× weight compression — is the actual output of this project.

---

## Running it

**Locally, before spending GPU time:**

```bash
python smoke_test.py --train
```

This checks the projection against brute force, cross-checks the complex
algebra against native `torch.complex` arithmetic, verifies that normalization
preserves phase, and confirms the gradient reaches every latent weight — a
broken STE would otherwise show up only as slightly worse accuracy, which is
easy to misread as the cost of quantization.

**On Kaggle GPU:**

```bash
kaggle kernels push -p kaggle/
```

**Visualization in Colab:** open `colab/cvqnn_visualization.ipynb`, set a T4
runtime, run all.

---

## Environment notes

**Kaggle assigns a Tesla P100 by default, and it does not work.** The P100 is
`sm_60`; the PyTorch preinstalled on Kaggle is built for `sm_70`+, since Pascal
support was dropped from the wheels. Every CUDA op fails with
`cudaErrorNoKernelImageForDevice`, and the traceback points deep into model
code rather than at the environment.

Two mitigations are in place. `kernel-metadata.json` sets
`"machine_shape": "NvidiaTeslaT4"` — note that `"Gpu"` means P100, and the
accelerator enum is not documented in the SDK. And `check_gpu_compat()` runs
before the data download, so an incompatible GPU costs seconds instead of a
wasted session.
