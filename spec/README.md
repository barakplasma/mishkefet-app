# Mishkefet spec

The contract the web app (`web/src/engine/`) and the iPhone app (`ios/MishkefetCore/`)
both implement. The source of truth is `tools/model/reference.py` (plus
`adapter.py` and `strokes.py`); these documents describe it in prose, and
`golden/` holds the numbers an implementation must reproduce. When a document and
the reference disagree, the reference is right.

| document | covers | reference |
|---|---|---|
| [model.md](model.md) | artifacts, backbone I/O, head, log-softmax | `Model`, `read_head` |
| [preprocess.md](preprocess.md) | line crop to model input, TTA views | `preprocess`, `tta_view` |
| [decoder.md](decoder.md) | greedy, prefix beam with LMs, confidence, TTA selection | `beam_search`, `Reader` |
| [lm-format.md](lm-format.md) | `charlm.bin`, `words.bin` | `CharLM`, `WordLM` |
| [orientation.md](orientation.md) | paper crop, quarter turns | `find_paper`, `orient_page` |
| [segmenter.md](segmenter.md) | page to line boxes (deskew up to 44 degrees) | `segment_page` |
| [straighten.md](straighten.md) | per-line tilt and curve | `straighten_line` |
| [strokes.md](strokes.md) | training canvas strokes to a line image | `strokes.render` |
| [adapter.md](adapter.md) | on-device fine-tuning (CTC, head, Adam) | `adapter.py` |
| [training.md](training.md) | the training screen, word choice, rounds, corrected lines | web `lib/training.ts`, `lib/words.ts` |

`charset.json` is the 198 output classes (index 0 is the CTC blank, 38 is space).
`adapt.json` holds the fine-tuning numbers, `train-words.json` the word pool.

`fixtures/` are Pinkas lines and pages (CC-BY-4.0, https://zenodo.org/records/3569694),
early modern Hebrew. They test fidelity, not accuracy.

## Tolerances

| what | tolerance |
|---|---|
| preprocess | every value within 3/255 of `golden/preprocess/*.u8` (blur rounding) |
| decoder on golden log-probs | identical text (it is deterministic double arithmetic) |
| LM values | within 1e-4 |
| CTC loss / gradient | within 1e-4 relative |
| adapter run | losses within 1e-3 relative, weights within 1e-4 |
| full line read (backbone on the platform runtime) | mean CER against `lines.json` `read` at most 0.02 |
| segmentation | same angle; at least 90% of golden boxes matched at IoU >= 0.9 |
| stroke render | same width; pixel sum within 0.5% |

Regenerate with `tools/model/golden.py` after changing any of the reference code.
