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

# 11notes/caddy: distroless (no shell), non-root 1000:1000, multi-arch, public on GHCR.
FROM ghcr.io/11notes/caddy:2.11.7
COPY docker/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /src/web/dist /srv/mishkefet-app
COPY --from=build /model /srv/mishkefet-app/hf
EXPOSE 8080
# Base image's healthcheck probes :3000; ours listens on :8080.
HEALTHCHECK CMD ["/usr/local/bin/localhealth", "http://127.0.0.1:8080/mishkefet-app/", "-I"]
CMD ["run", "--config", "/etc/caddy/Caddyfile", "--adapter", "caddyfile"]
