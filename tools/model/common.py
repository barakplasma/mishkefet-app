"""Shared paths and model loading for the tools in this folder.

The reference source is the Hugging Face snapshot of Mishkefet-v1 (its
`hebocr/` package is the code the shipped weights were evaluated with), cached
by fetch-model.sh / `python -m huggingface_hub` under ~/.cache/mishkefet/hf.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "spec"
HF_DIR = Path(os.environ.get("MISHKEFET_SRC", Path.home() / ".cache/mishkefet/hf"))
BUILD = Path(os.environ.get("MISHKEFET_BUILD", Path.home() / ".cache/mishkefet/build"))

HF_REPO = "itayinbar/Mishkefet-v1"
HF_SOURCE_REVISION = "23b3e237ffbf8ef73e87ef7c64dcf080e41e416b"


def hebocr():
    """Import the reference package from the snapshot, not from site-packages."""
    if str(HF_DIR) not in sys.path:
        sys.path.insert(0, str(HF_DIR))
    import hebocr  # noqa: F401

    return hebocr


def load_torch_model():
    """(model in eval mode, charset list) from mishkefet-v1.pt."""
    import torch

    hebocr()
    from hebocr.models.htr_vt import build_model

    state = torch.load(HF_DIR / "mishkefet-v1.pt", map_location="cpu", weights_only=False)
    chars = state["charset"]
    model = build_model(len(chars), state.get("config", {}).get("size", "base"), mask_ratio=0.0)
    model.load_state_dict(state["model"])
    model.eval()
    return model, chars
