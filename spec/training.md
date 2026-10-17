# The training flow (both apps)

Behavioral, not bit-exact: the two apps should feel the same, not draw the same
random words. Numbers come from `adapt.json`.

## A round
- 40 written words make a round (`round_size`). The home card shows progress.
- A word is shown; the user writes it on the pad. When the pen or finger lifts,
  a 1200 ms timer starts (`inactivity_ms`) and an orange bar runs along the
  pad; touching down again cancels it. When it fires, the strokes are rendered
  ([strokes.md](strokes.md), plain params, `n = word length`) and read with beam
  search, TTA 1, using the current fine-tuned head if there is one.
- Correct when the trimmed reading equals the word exactly. Strokes turn green
  or red, a badge springs in, a haptic plays, the reading is shown, the card
  fades, the next word appears. Every attempt is saved, right or wrong
  ("גם הטעויות נשמרות ללימוד").
- Skip moves on without saving. Clear wipes the pad.

## Choosing words (`train-words.json`, 3000 frequent words of 2 to 7 letters)
- Never repeat a word from the user's last 120 attempts.
- First round: until every one of the 27 letters (22 plus 5 final forms) has
  appeared, prefer common words that add uncovered letters.
- After that, weight each word by `(mean letter error rate)^2 * 1/sqrt(1 + rank/50)`,
  halved for words outside 3 to 6 letters. Letter error rate is
  `(misreads + 1) / (seen + 4)`, from a Levenshtein alignment of each attempt's
  reading against its word (substituted or deleted target letters count).
- Words from the least confident third of the user's scan lines (2 to 7 Hebrew
  letters) join the pool at double weight. Confidence is never shown to the user.

## Refit
See [adapter.md](adapter.md). Held out: `fnv1a32(sample id) % 8 == 0` (about 5
words a round), never trained on. Training uses all other samples (latest 240),
each as the plain render plus `aug_views` augmented renders (augment seeds
`(fnv1a32(id) & 0xFFFFFF) * 16 + k`). Features are cached per sample and seed.

The refit screen shows real work: the grid of this round's samples, then a ring
for feature extraction (first 40%) and training steps (last 60%), then the
held-out test, then character accuracy before and after. The new head is kept
only if accuracy did not drop; otherwise the screen says the previous version
stays.

## Corrected scan lines
When the user edits a line of a scan's reading, the line crop (padding 0.12 h
vertical, 0.05 h horizontal) and the corrected text become a training sample
(one view, never held out). They are real paper ink, which canvas strokes are not.

## Models on a scan
Each scan keeps one reading per model (`base`, or `r<round>` for a fine-tuned
head). New scans read with the fine-tuned head when one exists and the user has
not switched the toggle to the general model. Re-reading after a refit and
keeping the new reading marks the scan "נקרא מחדש".
