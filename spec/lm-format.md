# Language model binaries

Both are written by `tools/model/convert_lm.py` from the pickles in the HF repo.
Characters are charset indices (one byte). Log-probs are quantized:
`lp = -q / scale` with `q` a u16; `q = 65535` means "absent".

## charlm.bin (stupid-backoff character 6-gram)

```
offset 0   "MKL1"
       4   u32 order (6)
       8   f32 log_backoff   (log 0.4)
      12   f32 log_floor     (log(1 / total chars))
      16   f32 scale         (3000)
      20   u32 reserved
      24   u32 count[order]  nodes per level 1..6
           u32 child[level][count[level] + 1]   for levels 1..5
           u16 q[level][count[level]]           for levels 1..6
           u8  char[level][count[level]]        for levels 1..6
```

A trie in level order. Level-1 nodes are the root's children (all of them).
The children of node `i` at level `n` are nodes `child[n][i] .. child[n][i+1]-1`
at level `n+1`. Siblings are sorted by `char`, so finding a child is a binary
search. A node at level `n` is an n-gram; its `q` is
`log(count(ngram) / count(ngram minus its last char))` (unigrams: over total).

```
logprob(context, c):
  context = last 5 entries of context
  penalty = 0
  for L = len(context) down to 0:
    node = find(context[len-L:] + [c])        # walk from the root
    if node exists and q != 65535: return penalty - q / scale
    penalty += log_backoff
  return penalty + log_floor
```

Read `log_backoff`, `log_floor`, `scale` as float32 and widen to double.

## words.bin (word unigram prior)

```
offset 0   "MKW1"
       4   u32 n
       8   f32 oov_logprob      (-14)
      12   f32 prefix_discount  (-2.5)
      16   f32 scale            (3000)
      20   u32 reserved
      24   u32 offset[n + 1]    into the byte blob
           u16 q[n]             log(count / total)
           u8  blob[]           words as charset indices, sorted bytewise
```

```
logprob(word):
  empty -> 0
  found(word) -> -q/scale
  len > 2 and word[0] in {ב ה ו כ ל מ ש} and found(word[1:]) -> -q/scale + prefix_discount
  else -> oov_logprob
bonus(word) = max(0, logprob(word) - oov_logprob)
```

Lookup is a binary search with bytewise (unsigned, then shorter-first) comparison.
