# Offline, self-contained image of the Mishkefet web app.
# Upstream downloads the model from Hugging Face at runtime; this image bakes the
# pinned model files in and points the app at them, so it runs with no network.

FROM --platform=$BUILDPLATFORM node:22-alpine AS build
RUN apk add --no-cache curl
WORKDIR /src
COPY tools/model/fetch-model.sh tools/model/
COPY spec spec
COPY web web

# Model files at the revision upstream pins, verified against its SHA-256 list.
RUN rev=$(sed -n 's/^REV="\(.*\)"/\1/p' tools/model/fetch-model.sh) \
 && mkdir -p /model/$rev/onnx && cd /model/$rev/onnx \
 && sed -n '/^[0-9a-f]\{64\} /p' /src/tools/model/fetch-model.sh > SHA256SUMS \
 && for n in $(awk '{print $2}' SHA256SUMS); do \
      curl -fsSL --retry 4 -o "$n" "https://huggingface.co/itayinbar/Mishkefet-v1/resolve/$rev/onnx/$n"; done \
 && sed -i 's/ /  /' SHA256SUMS && sha256sum -c SHA256SUMS && rm SHA256SUMS

# Serve the model from our own origin. Fails the build if upstream changes this line.
RUN f=web/src/engine/constants.ts \
 && grep -q 'return `https://huggingface.co/${MODEL_REPO}/resolve/${revision}/${path}`' "$f" \
 && sed -i 's#return `https://huggingface.co/${MODEL_REPO}/resolve/${revision}/${path}`#return `/mishkefet-app/hf/${revision}/${path}`#' "$f"

WORKDIR /src/web
ENV ONNXRUNTIME_NODE_INSTALL=skip
RUN npm ci --no-audit --no-fund && npm run build

# English model: HTR-VT trained on IAM, exported to the app's format (tools/model-en).
# Code pinned by git commit, weights by SHA-256; the output is the same for every platform.
# torchvision comes from the CPU index too: timm imports it, and the PyPI build lacks the CPU torch ops.
FROM --platform=$BUILDPLATFORM python:3.13-slim AS model-en
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2 \
 && pip install --no-cache-dir timm==1.0.30 numpy==2.5.3 onnx==1.23.2 onnxconverter-common==1.16.0 onnxruntime==1.29.0
ADD https://github.com/YutingLi0606/HTR-VT.git#649ce7feca3ba55787498b0e1f4c48a38d403e07 /htrvt
ADD --checksum=sha256:eee631c8fd6411c821eb014d0517cf3949bde3f8df6a7b79c30eb878b6a7b193 \
    https://huggingface.co/Mir0da/HTR-VT-english/resolve/e2f2137900735f0d611d931a47879ab25fe6cf7a/best_CER.pth /ckpt/best_CER.pth
COPY tools/model-en /src
RUN python /src/export.py /htrvt /ckpt/best_CER.pth /out

# Ferron (official image): distroless, non-root (nobody), automatic HTTPS built in.
FROM ghcr.io/ferronweb/ferron:2
COPY docker/ferron.kdl /etc/ferron.kdl
COPY --from=build /src/web/dist /var/www/ferron/mishkefet-app
COPY --from=build /model /var/www/ferron/mishkefet-app/hf
COPY --from=model-en /out /var/www/ferron/mishkefet-app/hf-en
EXPOSE 8080
