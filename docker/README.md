# Offline Docker image

The whole app plus the Mishkefet-v1 model in one image. No network needed at runtime.
Images: `linux/amd64` and `linux/arm64`, built by `.github/workflows/docker.yml` on every push to `main`.

## Plain HTTP (localhost, or behind a TLS ingress such as k3s + cert-manager)

```sh
docker run --rm -p 8080:8080 ghcr.io/barakplasma/mishkefet-app:latest
# open http://localhost:8080
```

## Automatic HTTPS (Let's Encrypt, built into Ferron)

Point a DNS name at the host, then mount a config that names it. Ferron gets and renews
the certificate itself (TLS-ALPN-01 on port 443) and redirects HTTP to HTTPS.

`ferron.kdl`:

```kdl
globals {
  listen_ip "0.0.0.0"
}

mishkefet.example.com {
  root "/var/www/ferron"
  status 302 url="/" location="/mishkefet-app/"
}
```

```sh
docker run -d --restart=always -p 80:80 -p 443:443 \
  -v ./ferron.kdl:/etc/ferron.kdl:ro \
  -v ferron-acme:/var/cache/ferron-acme \
  ghcr.io/barakplasma/mishkefet-app:latest
```

Keep the `ferron-acme` volume, or every restart requests a new certificate and you hit
Let's Encrypt rate limits.

## Why HTTPS matters

The browser only allows the service worker, `crypto.subtle` (model hash check) and the camera
in a secure context. `http://localhost` counts; `http://192.168.x.x` does not.

## How it works

- Build stage downloads the model at the revision pinned in `tools/model/fetch-model.sh`
  and checks the SHA-256s listed there.
- One line in `web/src/engine/constants.ts` is rewritten so the app loads the model from
  `/mishkefet-app/hf/<revision>/onnx/...` on the same origin instead of huggingface.co. The
  build fails if upstream changes that line.
- The final image is the official [`ghcr.io/ferronweb/ferron:2`](https://github.com/ferronweb/ferron)
  (distroless, non-root `nobody`), with `docker/ferron.kdl` serving the build on port 8080.

## License

The model weights are CC-BY-NC-SA-4.0 (non-commercial, share-alike). See `NOTICE`.
