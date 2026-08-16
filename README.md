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

| arm | bits/weight | best val acc |
|---|---|---|
| quantized `{+1, −1, +i, −i}` | 2 | **89.15%** |
| complex FP32 control | 32 | 91.60% |
| | | **gap: 2.45 pp for 16× compression** |

Two diagnostics matter as much as the accuracy:

**The phase degree of freedom was not abandoned.** Final corner distribution is
`+1 = 22.2%`, `−1 = 28.4%`, `+i = 24.6%`, `−i = 24.8%` — the imaginary corners
hold 49.4% of all weights. The network did not collapse into a binary one.

**Quantization converged.** The flip rate — the fraction of weights changing
corner per epoch — decayed `20.4% → 9.5% → 5.4% → 0.05%` over 40 epochs. By the
end the configuration is genuinely discrete and stable, so the reported
accuracy belongs to a network that still exists after the last step rather than
to a momentary state.

### What this does *not* yet show

The comparison above prices quantization against *full precision*. It does not
show that **phase beats sign** — the control that would settle that is a
real-valued binary `{+1, −1}` network of matched capacity, which has not been
run yet.

The 49.4% imaginary share is suggestive but not sufficient on its own: weight
initialization draws `w_real` and `w_imag` i.i.d., so by symmetry roughly half
the weights start on an imaginary corner anyway. The balance proves the phase
corners were not *abandoned*; it does not prove they were *earned*. The flip
rate does show heavy reorganisation early on, so the final assignment is not
merely frozen initialization noise — but the binary baseline is what turns this
into an answer.

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
