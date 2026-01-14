import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType


def gabor_kernel(theta, frequency, sigma=4.0, size=11):
    """Create a real-valued 2D Gabor kernel."""
    # Create grid
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    # Rotation
    x_rot = xx * np.cos(theta) + yy * np.sin(theta)
    y_rot = -xx * np.sin(theta) + yy * np.cos(theta)
    # Gabor formula
    gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * frequency * x_rot)
    kernel = gaussian * sinusoid
    return kernel


def make_gabor_bank(num_scales=5, num_orients=8, filter_size=11, sigmas=[2, 4, 8]):
    bank = []
    # 1. Adaptive Frequency Spacing (Octave-based)
    # Start at a high frequency (e.g., 0.5) and decrease by half for each scale.
    # This ensures the kernel captures different levels of detail (coarse to fine).
    f_max = 0.4  # Cycles per pixel (Nyquist limit is 0.5)
    freqs = np.array([f_max / (np.sqrt(2) ** i) for i in range(num_scales)])
    for f in freqs:
        for k in range(num_orients):
            # heuristic sigma = 1/f if not provided already
            current_sigma = [1.0 / f] if not sigmas else sigmas
            for s in current_sigma:
                theta = k * np.pi / num_orients
                kern = gabor_kernel(theta, f, sigma=s, size=filter_size)
                bank.append(kern.astype(np.float32))
    return bank


class GaborFeatureNet(nn.Module):
    def __init__(self, bank, pool_size=(1, 1), energy="square"):
        super().__init__()
        num_filters = len(bank)
        # Conv with fixed weights
        filter_size = bank[0].shape[0]
        pad_size = filter_size // 2
        self.conv = nn.Conv2d(
            1, num_filters, kernel_size=filter_size, padding=pad_size, bias=False
        )
        with torch.no_grad():
            weight = np.stack(bank)[:, None, :, :]
            self.conv.weight.copy_(torch.from_numpy(weight))
        self.conv.weight.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.energy = energy

    def forward(self, x):
        x = self.conv(x)
        if self.energy == "square":
            x = x**2
        elif self.energy == "abs":
            x = x.abs()
        return self.pool(torch.log1p(x**2)).flatten(1)


if __name__ == "__main__":
    IMG_SIZE = 32
    FILTER_SIZE = 7  # good for 32 x 32: 7 or 11
    NUM_SCALE = 4
    NUM_ORIENTS = 8
    SIGMAS = None  # [2, 4, 8]
    POOL = (2, 2)  # (2, 2) or (4, 4)
    ENERGY = "square"  # "square" or "abs"
    ONNX_PATH = f"C:\\Users\\prith\\code\\imkl\\models\\gabor_filter-{FILTER_SIZE}_pool-{POOL[0]}_energy-{ENERGY}.onnx"

    bank = make_gabor_bank(
        num_scales=NUM_SCALE,
        num_orients=NUM_ORIENTS,
        filter_size=FILTER_SIZE,
        sigmas=SIGMAS,
    )
    model = GaborFeatureNet(bank, POOL, ENERGY)
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["features"],
        opset_version=17,
    )
    print(f"Saved Gabor filter bank model to: {ONNX_PATH}")

    # Use onnxsim to simplify the onnx compute graph
    print("Simplifying ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx_model, _ = simplify(onnx_model)
    onnx.save(onnx_model, ONNX_PATH)
    print(f"Done. Simplified model saved to {ONNX_PATH}")
