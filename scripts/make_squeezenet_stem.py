import torch
import torch.nn as nn
from torchvision import models
import onnx
from onnxsim import simplify
from onnxruntime.quantization import quantize_dynamic, QuantType


class SqueezeNetFeatureExtractor(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    # config
    N = 1
    INPUT_SHAPE = (1, 3, 32, 32)
    ONNX_PATH = f"squeezenet1_1_{N}-layer_input-{INPUT_SHAPE[-1]}.onnx"
    QUANT_ONNX_PATH = f"squeezenet1_1_{N}-layer_input-{INPUT_SHAPE[-1]}_int.onnx"
    OPSET = 12
    quantize = False

    # load pretrained model
    model = models.squeezenet1_1(pretrained=True)
    model.eval()
    features = model.features[:N]

    feat_model = SqueezeNetFeatureExtractor(features).eval()
    # Export to ONNX with opset 12
    dummy_input = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        feat_model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={
            "input": {0: "batch"},
            "features": {0: "batch"},
        },
    )
    print(f"ONNX model exported: {ONNX_PATH}")

    # Use onnxsim to simplify the onnx compute graph
    print("Simplifying ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx_model, _ = simplify(onnx_model)
    onnx.save(onnx_model, ONNX_PATH)
    print(f"Done. Simplified model saved to {ONNX_PATH}")

    # Dynamic INT8 quantization (weights only)
    if quantize:
        quantize_dynamic(
            model_input=ONNX_PATH,
            model_output=QUANT_ONNX_PATH,
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized ONNX model saved: {QUANT_ONNX_PATH}")
