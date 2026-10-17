# Model artifacts

Mishkefet-v1 (HF `itayinbar/Mishkefet-v1`, CC-BY-NC-SA-4.0) is a CNN + 8-layer
ViT encoder with a CTC head over 198 characters. The apps ship it split in two.

| file | size | what |
|---|---|---|
| `backbone.fp16.onnx` | 63 MB | image `(1, 1, 64, W)` float32 to `features (1, W // 4, 512)` float32; fp16 weights, fp32 I/O |
| `head.bin` | 406 KB | the CTC head, applied by the app |
| `charlm.bin` | 8.4 MB | character 6-gram LM ([lm-format.md](lm-format.md)) |
| `words.bin` | 2.7 MB | word unigram prior |
| `charset.json` | 1.5 KB | the 198 classes |

fp16 vs fp32 on the 225 ivrit.ai benchmark lines: median CER 0.1875 both, mean
0.2635 both (hebocr itself: 0.1875 / 0.2631). fp16 ships.

`W` may be anything from 16 to 2560. There is no padding mask in the model, so
run one line per call at its true width (no batching with padding).

## head.bin

```
"MKH1"  u32 classes (198)  u32 dim (512)
f32 weight[classes][dim]   f32 bias[classes]
```

A fine-tuned head is stored in the same format plus an optional adapter
(see [adapter.md](adapter.md)).

## Log-probabilities

```
logits[t][k] = sum_d features[t][d] * weight[k][d] + bias[k]
logprobs[t]  = logits[t] - max - log(sum(exp(logits[t] - max)))
```

in float32 (the reference computes in float32 numpy).
