import sys
import os
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hashes import (
    ColorHash,
    PerceptualHash,
    PixelHash,
    WaveletHash,
    HDiffHash,
    VDiffHash,
    HOGHash,
    CornerCountHash,
    LineCountHash,
)
from utils import MemoizedImage

# config
IMG_SIZES = [128, 512, 1024]
IMG_COUNTS = [1, 5, 10, 20]
REF_SIZE = 512
N_WARMUP = 3
N_RUNS = 15


def make_hashes():
    return {
        "ColorHash": ColorHash(img_size=64),
        "PerceptualHash": PerceptualHash(
            hash_size=16, highfreq_factor=4, thresh="mean"
        ),
        "PixelHash": PixelHash(hash_size=32, thresh="mean"),
        "WaveletHash": WaveletHash(hash_size=16, scale=4, thresh="mean"),
        "HDiffHash": HDiffHash(hash_size=32),
        "VDiffHash": VDiffHash(hash_size=32),
        "HOGHash": HOGHash(img_size=64, thresh="mean"),
        "CornerCountHash": CornerCountHash(img_size=64, hash_dim=32),
        "LineCountHash": LineCountHash(img_size=64, hash_dim=32),
    }


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


def run_hashes(hashes_dict, imgs):
    memos = [MemoizedImage(img) for img in imgs]
    return {
        i: np.array([hf.feat(m) for m in memos], dtype=np.uint8)
        for i, hf in enumerate(hashes_dict.values())
    }


# per-hash latency
hash_lat = defaultdict(dict)
hash_names = list(make_hashes().keys())
for size in IMG_SIZES:
    img = rand_img(size, size)
    hashes = make_hashes()
    for name, hf in hashes.items():
        memo = MemoizedImage(img)
        hash_lat[name][size] = bench(lambda hf=hf, memo=memo: hf.feat(memo))

# hash() latency vs num_hashes
all_hashes = make_hashes()
all_hash_names = list(all_hashes.keys())
num_hash_range = list(range(1, len(all_hash_names) + 1))
lat2 = {n: {} for n in IMG_COUNTS}
for n_img in IMG_COUNTS:
    imgs = [rand_img(REF_SIZE, REF_SIZE) for _ in range(n_img)]
    for k in num_hash_range:
        subset = {n: all_hashes[n] for n in all_hash_names[:k]}
        lat2[n_img][k] = bench(lambda s=subset, i=imgs: run_hashes(s, i))

# hash() latency vs image size
all_hashes_fixed = make_hashes()
n_hashes = len(all_hashes_fixed)
lat3 = {n: {} for n in IMG_COUNTS}

for n_img in IMG_COUNTS:
    for size in IMG_SIZES:
        imgs = [rand_img(size, size) for _ in range(n_img)]
        lat3[n_img][size] = bench(lambda h=all_hashes_fixed, i=imgs: run_hashes(h, i))

# plotting
PALETTE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
IMG_COLORS = {n: PALETTE[i] for i, n in enumerate(IMG_COUNTS)}
MARKERS = ["o", "s", "^"]
fig = plt.figure(figsize=(20, 14))
fig.suptitle("MKL Hash Function Benchmark", fontsize=16, fontweight="bold", y=0.98)

# plot 1 — per-hash latency
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
# plot 2 — latency vs num_hashes
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
# plot 3 — latency vs num_images
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
# plot 4 — latency vs image size
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
plt.tight_layout()
out_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(out_dir, exist_ok=True)

plt.savefig(os.path.join(out_dir, "hash_benchmark.png"), dpi=150, bbox_inches="tight")
