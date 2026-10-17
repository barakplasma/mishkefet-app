# Decoding

Input: `logprobs (T, 198)` float32 for one view. Classes: 0 = blank, 38 = space.
All score arithmetic is in **double**. Output strings are in logical (reading)
order already; never reverse them.

## Greedy
Argmax per frame (ties to the lower index), collapse repeats, drop blanks.

## Confidence
`p_t = exp(max_k logprobs[t][k])`.
- all-frame: mean of `p_t` over all T (0 if T = 0). Stored with each line.
- emitting: mean over frames whose argmax is not blank and differs from the
  previous frame's argmax (frame 0 counts if not blank); 0 if none. Used only
  to choose between TTA views.

## Prefix beam search
Constants: beam 12, top_k 8, lm_weight 0.4, length_bonus 0.6, word_weight 0.2.
`lse(a, b)`: if a = -inf return b; if b = -inf return a; else `hi + log1p(exp(lo - hi))`.

```
beams = ordered map { [] : (pb 0, pnb -inf) }
for t in 0..T-1:
  cands = indices of the 8 largest logprobs[t], largest first, ties to lower index
  next = empty ordered map (insertion order matters)
  for (prefix, (pb, pnb)) in beams, in order:
    ptot = lse(pb, pnb)
    for c in cands:
      p = logprobs[t][c]
      if c == blank:
        e = next.getOrInsert(prefix, (-inf, -inf));  e.pb = lse(e.pb, ptot + p);  continue
      if prefix not empty and last(prefix) == c:
        e = next.getOrInsert(prefix, (-inf, -inf));  e.pnb = lse(e.pnb, pnb + p)
        src = pb
      else:
        src = ptot
      if src == -inf: continue
      bonus = 0.4 * charlm.logprob(prefix, c) + 0.6
      if c == space and lastWord(prefix) not empty:
        bonus += 0.2 * words.bonus(lastWord(prefix))
      e = next.getOrInsert(prefix + [c], (-inf, -inf));  e.pnb = lse(e.pnb, src + p + bonus)
  beams = the first 12 of next, stably sorted by lse(pb, pnb) descending
          (equal scores keep insertion order); if empty, { [] : (0, -inf) }
final score = lse(pb, pnb) + 0.2 * words.bonus(lastWord(prefix)) when lastWord is not empty
answer = the first beam (in beam order) with the maximum final score
```

`lastWord(prefix)` is everything after the last space (the whole prefix if there
is no space), punctuation included. Note `getOrInsert` on `prefix` for the
repeat case happens **before** the extension is inserted.

## Reading a line (TTA 3)
For each scale in `1.0, 0.82, 1.22`: view, preprocess, model, beam. Score =
emitting confidence, or -inf if the text is blank after trimming whitespace.
Keep the first view with the strictly highest score. The line's stored
confidence is the all-frame confidence of the scale-1.0 view.

With TTA 1 (training check, web on WASM), just the scale-1.0 view.

## Page
Segment ([segmenter.md](segmenter.md)), read every crop in reading order. Lines
whose text is blank are dropped from the transcript; nothing else is filtered.
