# On-device fine-tuning ("teach Mishkefet your handwriting")

The backbone is frozen. Only the CTC head (`W` 198x512, `b` 198) is trained, on
backbone features cached per training view. `adapter.py` is the reference;
`golden/adapter.json` and `golden/ctc.json` are the vectors.

## Why head only
Simulated on the ivrit.ai benchmark's 10 writers (8 lines, about 40 words, per
writer; `tools/model/adapt_sim.py`), greedy CER:

| variant | that writer | other writers |
|---|---|---|
| head, L2 0.1, lr 1e-3, 100 steps | -17.5% | +8.0% |
| head, L2 1.0, 200 steps | -13.2% | +3.1% |
| head + rank-16 residual adapter | -16.0% | +17.0% |

A residual adapter forgets more and learns no more, so the apps train the head
only. "Other writers" pays a little, which is why every scan can be re-read with
the general model.

## Shipped configuration (`adapt.json`)
```
steps 100, Adam lr 1e-3, beta1 0.9, beta2 0.999, eps 1e-8, l2 0.1 (toward the general head)
views per sample: plain render + AUG augmented renders (iOS AUG 4, web AUG 1)
max training samples: the latest 240 non-held-out samples
held out: a sample is held out when hash(sample id) % 8 == 0 (FNV-1a 32 of the UTF-8 id)
accept the new head only if held-out character accuracy does not drop
```

## CTC (per sample)
Logits `z = F W^T + b` (T x C). Log-softmax in double. Extended label
`l' = [blank, l1, blank, l2, ..., blank]`, `S = 2L + 1`. A skip from `s-2` is allowed
when `l'[s]` is not blank and `l'[s] != l'[s-2]`.

```
alpha[0][0] = lp[0][blank]; alpha[0][1] = lp[0][l1]
alpha[t][s] = lse(alpha[t-1][s], alpha[t-1][s-1], skip ? alpha[t-1][s-2]) + lp[t][l'[s]]
beta[T-1][S-1] = beta[T-1][S-2] = 0                       (excludes frame t's emission)
beta[t][s] = lse over s' in {s, s+1, s+2 if skip(s+2)} of beta[t+1][s'] + lp[t+1][l'[s']]
logZ = lse(alpha[T-1][S-1], alpha[T-1][S-2])
loss = -logZ
dz[t][k] = exp(lp[t][k]) - sum over s with l'[s] = k of exp(alpha[t][s] + beta[t][s] - logZ)
```
If `logZ` is -inf (label too long for T), the sample contributes nothing (zero_infinity).
Use -1e30 for log zero, as the reference does.

## Objective and update
```
loss = mean over samples with finite logZ of ctc(sample) / len(label)
     + l2 * (|W - W0|^2 + |b - b0|^2)
gW = mean_i (dz_i / len_i)^T F_i + 2 l2 (W - W0)
gb = mean_i colsum(dz_i / len_i) + 2 l2 (b - b0)
Adam (bias corrected), full batch, `steps` steps, parameters stored as float32.
```

## Head file with a fine-tuned head
Same as `head.bin` ("MKH1"); the apps keep each accepted round as its own file
(`head-r<round>.bin`) and remember which one is current.

## Measuring a round
Character accuracy on the held-out samples = `1 - sum(edit distance) / sum(len(target))`,
reading each held-out sample's plain render with beam search (TTA 1), once with
the previous head and once with the new one.
