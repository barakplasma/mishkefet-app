"""Export the English model (HTR-VT, trained on IAM) in the app's format. Run by the Dockerfile.

    python export.py <HTR-VT checkout> <best_CER.pth> <out dir>

Writes onnx/backbone.fp16.onnx (image (1, 1, 64, 512) -> features (1, 128, 768)),
onnx/head.bin (MKH1, Linear 768 -> 80), onnx/charset.json and manifest.json
(path, bytes, sha256 per file; the app verifies downloads against it).

HTR-VT ends with a parameter-free LayerNorm over the whole logit matrix. It is
skipped: it never changes the per-frame argmax, and the app decodes greedily
for this model (no English language model).
"""
import hashlib, json, re, shutil, struct, sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

htrvt, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(htrvt))
from model import HTR_VT  # noqa: E402

charset = json.loads((Path(__file__).parent / 'charset.json').read_text())
assert charset[0] == '' and len(charset) == 80

model = HTR_VT.create_model(nb_cls=len(charset), img_size=[64, 512])
# weights_only: the checkpoint comes from a third party; refuse anything but tensors.
state = torch.load(ckpt, map_location='cpu', weights_only=True)['state_dict_ema']
model.load_state_dict(OrderedDict((re.sub('module.', '', k), v) for k, v in state.items()), strict=True)
model.eval()


class Backbone(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        m = self.m
        x = m.patch_embed(m.layer_norm(image))
        b, c = x.shape[:2]
        x = x.view(b, c, -1).permute(0, 2, 1) + m.pos_embed
        for blk in m.blocks:
            x = blk(x)
        return m.norm(x)


net = Backbone(model).eval()
x0 = torch.rand(1, 1, 64, 512)
with torch.no_grad():  # backbone + head must reproduce the full model
    full = model(x0)
    split = F.layer_norm(model.head(net(x0)), full.shape[1:])
assert torch.allclose(full, split, atol=1e-5), float((full - split).abs().max())

onnx_dir = out / 'onnx'
onnx_dir.mkdir(parents=True, exist_ok=True)
fp32 = out / 'backbone.fp32.onnx'
torch.onnx.export(net, (x0,), str(fp32), input_names=['image'], output_names=['features'], opset_version=17, dynamo=False)

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnxconverter_common import float16  # noqa: E402

onnx.save(float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True), onnx_dir / 'backbone.fp16.onnx')
fp32.unlink()

w = model.head.weight.detach().numpy()
b = model.head.bias.detach().numpy()
with open(onnx_dir / 'head.bin', 'wb') as fh:
    fh.write(b'MKH1' + struct.pack('<II', *w.shape))
    fh.write(np.ascontiguousarray(w, dtype='<f4').tobytes())
    fh.write(np.ascontiguousarray(b, dtype='<f4').tobytes())
shutil.copy(Path(__file__).parent / 'charset.json', onnx_dir / 'charset.json')

# The fp16 graph must agree with PyTorch on which class wins each frame.
feats = ort.InferenceSession(str(onnx_dir / 'backbone.fp16.onnx'), providers=['CPUExecutionProvider']).run(None, {'image': x0.numpy()})[0][0]
with torch.no_grad():
    ref = model.head(net(x0))[0].numpy()
agree = float(((feats @ w.T + b).argmax(-1) == ref.argmax(-1)).mean())
assert agree >= 0.95, f'fp16 argmax agreement {agree:.3f}'

manifest = []
for key, name in (('backbone', 'backbone.fp16.onnx'), ('head', 'head.bin'), ('charset', 'charset.json')):
    data = (onnx_dir / name).read_bytes()
    manifest.append({'key': key, 'path': f'onnx/{name}', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(f'fp16 argmax agreement {agree:.3f}; wrote', ', '.join(f"{m['path']} ({m['bytes']} B)" for m in manifest))
