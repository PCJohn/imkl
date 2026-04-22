import sys
import os
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("Agg")

from imkl import IMKL

# Benchmark config
IMG_SIZES = [128, 512, 1024]
IMG_COUNTS = [1, 5, 10, 20]
REF_SIZE = 512
N_WARMUP = 10
N_RUNS = 20

# Hasher config
MKL_CONFIG = """
shared_preproc:
  edges: [false]
  log_polar: [false]

kernels:
  - class: ColorHash
    params:
      img_size: 64
  - class: PerceptualHash
    params:
      hash_size: 16
      highfreq_factor: 4
      thresh: mean
  - class: PixelHash
    params:
      hash_size: 32
      thresh: mean
  - class: WaveletHash
    params:
      hash_size: 16
      scale: 4
      thresh: mean
  - class: HDiffHash
    params:
      hash_size: 32
  - class: VDiffHash
    params:
      hash_size: 32
  - class: HOGHash
    params:
      img_size: 64
      thresh: mean
  - class: CornerCountHash
    params:
      img_size: 64
      hash_dim: 32
  - class: LineCountHash
    params:
      img_size: 64
      hash_dim: 32

fit_params:
  policy: cka
  topk: 0
  normalize: true
"""

m = IMKL(MKL_CONFIG)
hash_names = [type(hf).__name__ for hf in m.hash_funcs]
num_hash_range = list(range(1, m.num_hash + 1))
n_hashes = m.num_hash


def rand_img(h, w):
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def bench(fn):
    for _ in range(N_WARMUP):
        fn()
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.std(times))


# Per-hash latency
hash_lat = defaultdict(dict)
for size in IMG_SIZES:
    img = rand_img(size, size)
    for k, name in enumerate(hash_names):
        m.hash_funcs, saved = m.hash_funcs[k : k + 1], m.hash_funcs
        hash_lat[name][size] = bench(lambda i=img: m.hash([i]))
        m.hash_funcs = saved
# Latency of hash() vs num_hashes
lat2 = {n: {} for n in IMG_COUNTS}
for n_img in IMG_COUNTS:
    imgs = [rand_img(REF_SIZE, REF_SIZE) for _ in range(n_img)]
    for k in num_hash_range:
        m.hash_funcs, saved = m.hash_funcs[:k], m.hash_funcs
        lat2[n_img][k] = bench(lambda i=imgs: m.hash(i))
        m.hash_funcs = saved
# Latency of hash() vs image size
lat3 = {n: {} for n in IMG_COUNTS}
for n_img in IMG_COUNTS:
    for size in IMG_SIZES:
        imgs = [rand_img(size, size) for _ in range(n_img)]
        lat3[n_img][size] = bench(lambda i=imgs: m.hash(i))

PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
IMG_COLORS = {n: PALETTE[i] for i, n in enumerate(IMG_COUNTS)}
MARKERS = ["o", "s", "^"]
fig = plt.figure(figsize=(20, 14))
fig.suptitle("MKL Hash Function Benchmark", fontsize=16, fontweight="bold", y=0.98)
# Plot 1: per-hash latency
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(hash_names))
width = 0.25
for i, size in enumerate(IMG_SIZES):
    meds = [hash_lat[n][size][0] for n in hash_names]
    errs = [hash_lat[n][size][1] for n in hash_names]
    ax1.bar(
        x + i * width,
        meds,
        width,
        label=f"{size}px",
        color=PALETTE[i],
        alpha=0.85,
        yerr=errs,
        capsize=3,
    )
ax1.set_title("1. Per-hash latency vs image size", fontweight="bold")
ax1.set_xlabel("Hash function")
ax1.set_ylabel("Latency (ms)")
ax1.set_xticks(x + width)
ax1.set_xticklabels(hash_names, rotation=35, ha="right", fontsize=8)
ax1.legend(title="Image size")
ax1.grid(axis="y", alpha=0.3)
ax1.set_yscale("log")
# Plot 2: latency vs num_hashes
ax2 = fig.add_subplot(2, 2, 2)
for n_img in IMG_COUNTS:
    meds = [lat2[n_img][k][0] for k in num_hash_range]
    errs = [lat2[n_img][k][1] for k in num_hash_range]
    ax2.errorbar(
        num_hash_range,
        meds,
        yerr=errs,
        label=f"{n_img} img{'s' if n_img > 1 else ''}",
        color=IMG_COLORS[n_img],
        marker="o",
        linewidth=1.8,
        capsize=3,
        markersize=5,
    )
ax2.set_title(
    f"2. hash() latency vs # hash functions\n(image size {REF_SIZE}×{REF_SIZE})",
    fontweight="bold",
)
ax2.set_xlabel("Number of hash functions")
ax2.set_ylabel("Latency (ms)")
ax2.set_xticks(num_hash_range)
ax2.set_xticklabels([str(k) for k in num_hash_range])
ax2.legend(title="# images")
ax2.grid(alpha=0.3)
# Plot 3: latency vs num_images
ax3 = fig.add_subplot(2, 2, 3)
for i, size in enumerate(IMG_SIZES):
    meds = [lat3[n][size][0] for n in IMG_COUNTS]
    errs = [lat3[n][size][1] for n in IMG_COUNTS]
    ax3.errorbar(
        IMG_COUNTS,
        meds,
        yerr=errs,
        label=f"{size}×{size}px",
        color=PALETTE[i],
        marker=MARKERS[i],
        linewidth=1.8,
        capsize=3,
        markersize=6,
    )
ax3.set_title(
    f"3. hash() latency vs # images\n({n_hashes} hash functions)", fontweight="bold"
)
ax3.set_xlabel("Number of images")
ax3.set_ylabel("Latency (ms)")
ax3.set_xticks(IMG_COUNTS)
ax3.legend(title="Image size")
ax3.grid(alpha=0.3)
# Plot 4: latency vs image size
ax4 = fig.add_subplot(2, 2, 4)
for n_img in IMG_COUNTS:
    meds = [lat3[n_img][size][0] for size in IMG_SIZES]
    errs = [lat3[n_img][size][1] for size in IMG_SIZES]
    ax4.errorbar(
        IMG_SIZES,
        meds,
        yerr=errs,
        label=f"{n_img} img{'s' if n_img > 1 else ''}",
        color=IMG_COLORS[n_img],
        marker="o",
        linewidth=1.8,
        capsize=3,
        markersize=6,
    )
ax4.set_title(
    f"4. hash() latency vs image size\n({n_hashes} hash functions)", fontweight="bold"
)
ax4.set_xlabel("Image size (px)")
ax4.set_ylabel("Latency (ms)")
ax4.set_xticks(IMG_SIZES)
ax4.set_xticklabels([f"{s}×{s}" for s in IMG_SIZES])
ax4.legend(title="# images")
ax4.grid(alpha=0.3)
# Package and save plot
plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "hash_benchmark.png"), dpi=150, bbox_inches="tight")
