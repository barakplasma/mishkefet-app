"""Export Mishkefet-v1 for the apps.

The ONNX graph is the backbone only: line image (1, 1, 64, W) -> features
(1, T, 512), T = W // 4, taken after the final LayerNorm. The CTC head (one
Linear 512 -> 198) is written separately to head.bin and applied by the apps
themselves, so the general model and a user's fine-tuned head share one code
path and fine-tuning never touches the ONNX file.

    python export_onnx.py            # writes $BUILD/{backbone.fp32.onnx, backbone.fp16.onnx, head.bin, charset.json}

head.bin layout (little endian): magic "MKH1", u32 classes, u32 dim,
float32 weight[classes][dim], float32 bias[classes].
"""

import json
import struct

import numpy as np
import torch
import torch.nn as nn

from common import BUILD, load_torch_model


class Backbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, image):
        m = self.m
        x = m.proj(m.cnn(image))
        x = m.pos(x)
        x = m.encoder(x)
        return m.norm(x)


def write_head(model, path):
    w = model.head.weight.detach().float().numpy()
    b = model.head.bias.detach().float().numpy()
    with open(path, "wb") as fh:
        fh.write(b"MKH1" + struct.pack("<II", *w.shape))
        fh.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(b, dtype="<f4").tobytes())


def export(model, path):
    torch.backends.mha.set_fastpath_enabled(False)
    net = Backbone(model).eval()
    dummy = torch.rand(1, 1, 64, 400)
    width = torch.export.Dim("width", min=16, max=2560)
    prog = torch.onnx.export(
        net, (dummy,), input_names=["image"], output_names=["features"],
        dynamic_shapes={"image": {3: width}}, opset_version=20, dynamo=True, optimize=True,
    )
    prog.save(str(path), external_data=False)


def to_fp16(src, dst):
    import onnx
    from onnxconverter_common import float16

    m = onnx.load(str(src))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(dst))


def check(model, path, widths=(16, 57, 400, 1203, 2560)):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    net = Backbone(model).eval()
    worst = 0.0
    for w in widths:
        x = torch.rand(1, 1, 64, w)
        with torch.no_grad():
            ref = net(x).numpy()
        out = sess.run(None, {"image": x.numpy()})[0]
        assert out.shape == ref.shape == (1, w // 4, 512), (out.shape, ref.shape)
        worst = max(worst, float(np.abs(out - ref).max()))
    return worst


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    model, chars = load_torch_model()
    (BUILD / "charset.json").write_text(json.dumps(chars, ensure_ascii=False))
    write_head(model, BUILD / "head.bin")
    fp32 = BUILD / "backbone.fp32.onnx"
    export(model, fp32)
    print("fp32 max |diff| vs torch:", check(model, fp32))
    fp16 = BUILD / "backbone.fp16.onnx"
    to_fp16(fp32, fp16)
    print("fp16 max |diff| vs torch:", check(model, fp16))
    for p in (fp32, fp16, BUILD / "head.bin"):
        print(p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
