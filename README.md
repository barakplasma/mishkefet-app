# משקפת · Mishkefet

Hebrew handwriting recognition that runs entirely on your phone, and learns your
handwriting. Nothing is uploaded.

- **Web** (iPhone Safari, Android Chrome, desktop): https://itayinbarr.github.io/mishkefet-app/
- **iPhone app**: coming to the App Store

Photograph a line or a whole page (or pick images and PDFs), and Mishkefet reads
it on the device and keeps every scan and its readings. Write a few words on the
screen and it fine-tunes itself to your hand: every 40 words it refits, measures
itself on words it never trained on, and keeps the new version only if it got
better. Any scan can be read with the general model or with your fine-tuned one.

Powered by [Mishkefet-v1](https://huggingface.co/itayinbar/Mishkefet-v1), a
30M-parameter CNN + ViT CTC recognizer for modern Hebrew cursive (best page CER on
the ivrit.ai handwriting leaderboard; training code in
[heb-ocr](https://github.com/itayinbarr/heb-ocr)).

## Layout

| path | what |
|---|---|
| `web/` | Vite + React web app, deployed to GitHub Pages |
| `ios/` | SwiftUI iPhone app (`xcodegen`), with the `MishkefetCore` engine package |
| `spec/` | What both apps implement: written specs, test vectors, golden readings |
| `tools/model/` | Reference pipeline (`reference.py`), ONNX export, LM conversion, fine-tuning reference and simulation, pinned model fetch |

## How it works

A page is binarized, deskewed, cleaned of ruled lines and cut into lines
(`spec/segmenter.md`). Each line is contrast-normalized and scaled to 64 px, read
at three horizontal scales by the fp16 ONNX backbone, and decoded with a CTC prefix
beam search fused with a character 6-gram and a word prior (`spec/decoder.md`).

The CTC head (one 512 to 198 layer) runs outside ONNX, so fine-tuning is a small,
hand-written CTC + Adam loop over cached backbone features (`spec/adapter.md`).
Simulated on ten real writers, one round cuts that writer's character error rate by
about 17%.

Both apps implement the same steps and are tested against
`tools/model/reference.py`, which is itself checked against the original `hebocr`
code (same benchmark CER).

## License

Code: MIT. Model, data and fonts: see `NOTICE`. The model weights are
CC-BY-NC-SA-4.0, so Mishkefet is and stays free and non-commercial.
