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
 && sed -n '/^[0-9a-f]\{64\} /p' /src/tools/model/fetch-model.sh | grep -v charset.json > SHA256SUMS \
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

FROM nginx:1.29-alpine
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/web/dist /usr/share/nginx/html/mishkefet-app
COPY --from=build /model /usr/share/nginx/html/mishkefet-app/hf
EXPOSE 8080
