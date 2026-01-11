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
    freqs = np.linspace(0.1, 0.4, num_scales)  # frequencies in cycles/pixel
    for f in freqs:
        for k in range(num_orients):
            for s in sigmas:
                theta = k * np.pi / num_orients
                kern = gabor_kernel(theta, f, sigma=s, size=filter_size)
                bank.append(kern.astype(np.float32))
    return bank


class GaborFeatureNet(nn.Module):
    def __init__(self, bank):
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
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x**2).flatten(1)


if __name__ == "__main__":
    IMG_SIZE = 32
    FILTER_SIZE = 7
    NUM_SCALE = 5
    NUM_ORIENTS = 8
    SIGMAS = [2, 4, 8]
    bank = make_gabor_bank(
        num_scales=NUM_SCALE,
        num_orients=NUM_ORIENTS,
        filter_size=FILTER_SIZE,
        sigmas=SIGMAS,
    )
    ONNX_PATH = f"C:\\Users\\prith\\code\\imkl\\gabor_filter-{FILTER_SIZE}.onnx"
    model = GaborFeatureNet(bank)
    dummy = torch.randn(1, 1, IMG_SIZE, IMG_SIZE)
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["features"],
        opset_version=12,
    )
    print(f"Saved Gabor filter bank model to: {ONNX_PATH}")

    # Use onnxsim to simplify the onnx compute graph
    print("Simplifying ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx_model, _ = simplify(onnx_model)
    onnx.save(onnx_model, ONNX_PATH)
    print(f"Done. Simplified model saved to {ONNX_PATH}")
