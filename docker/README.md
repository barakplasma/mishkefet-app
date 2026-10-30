# Offline Docker image

The whole app plus the Mishkefet-v1 model in one image. No network needed at runtime.

```sh
docker run --rm -p 8080:8080 ghcr.io/barakplasma/mishkefet-app:latest
# open http://localhost:8080
```

Images: `linux/amd64` and `linux/arm64`, built by `.github/workflows/docker.yml` on every push to `main`.

## HTTPS is required off localhost

The browser only allows the service worker, `crypto.subtle` (model hash check) and the camera
in a secure context. `http://localhost` counts; `http://192.168.x.x` does not. On another
host, put it behind TLS (e.g. a k3s Ingress with cert-manager, or Caddy).

## How it works

- Build stage downloads the model at the revision pinned in `tools/model/fetch-model.sh`
  and checks the SHA-256s listed there.
- One line in `web/src/engine/constants.ts` is rewritten so the app loads the model from
  `/mishkefet-app/hf/<revision>/onnx/...` on the same origin instead of huggingface.co. The build fails
  if upstream changes that line.
- nginx serves the static build on port 8080.

## License

The model weights are CC-BY-NC-SA-4.0 (non-commercial, share-alike). See `NOTICE`.
