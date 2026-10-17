"""Reference for on-device writer adaptation: CTC loss and gradient, the head
(+ optional residual adapter), and its trainer. spec/adapter.md describes the
same thing for the apps; this file is the yardstick.

The backbone is frozen and its features are cached, so the trainable part is

    h      = f + B · gelu(A · f)          (adapter, rank r; skipped when r = 0)
    logits = W · h + b                    (the CTC head, 198 x 512)
    loss   = mean over samples of CTC(logits) / len(label)
             + l2 · (|W - W0|^2 + |b - b0|^2) + l2_adapter · (|A|^2 + |B|^2)

B starts at zero, so step 0 reproduces the general model exactly.
"""

import math
from dataclasses import dataclass, field

import numpy as np

NEG = -1e30


def log_softmax(z):
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def lse2(a, b):
    m = np.maximum(a, b)
    return np.where(m <= NEG / 2, NEG, m + np.log(np.exp(a - m) + np.exp(b - m)))


def ctc_loss_grad(logits: np.ndarray, label: list[int], blank=0):
    """CTC negative log-likelihood and its gradient w.r.t. the logits.

    logits (T, C) are pre-softmax. Returns (loss, dlogits). Infeasible
    alignments (label longer than T allows) return (0, zeros), like PyTorch's
    zero_infinity=True."""
    T, C = logits.shape
    lp = log_softmax(logits.astype(np.float64))
    ext = [blank]
    for c in label:
        ext += [c, blank]
    S = len(ext)
    ext = np.array(ext)
    # a skip from s-2 to s is allowed when ext[s] is a label that differs from ext[s-2]
    skip = np.zeros(S, bool)
    skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])

    alpha = np.full((T, S), NEG)
    alpha[0, 0] = lp[0, ext[0]]
    if S > 1:
        alpha[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = alpha[t - 1]
        a = prev.copy()
        a[1:] = lse2(a[1:], prev[:-1])
        a[2:] = np.where(skip[2:], lse2(a[2:], prev[:-2]), a[2:])
        alpha[t] = a + lp[t, ext]

    # beta[t, s]: log prob of emitting the rest after being in state s at t,
    # excluding frame t's own emission
    beta = np.full((T, S), NEG)
    beta[T - 1, S - 1] = 0.0
    if S > 1:
        beta[T - 1, S - 2] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = beta[t + 1] + lp[t + 1, ext]
        b = nxt.copy()
        b[:-1] = lse2(b[:-1], nxt[1:])
        sk = np.zeros(S, bool)
        sk[:-2] = skip[2:]
        b[:-2] = np.where(sk[:-2], lse2(b[:-2], nxt[2:]), b[:-2])
        beta[t] = b

    logz = alpha[T - 1, S - 1]
    if S > 1:
        logz = float(lse2(np.array(logz), np.array(alpha[T - 1, S - 2])))
    if logz <= NEG / 2:
        return 0.0, np.zeros_like(logits, dtype=np.float32)

    post = np.exp(alpha + beta - logz)  # (T, S) state occupancy
    onehot = np.zeros((S, C))
    onehot[np.arange(S), ext] = 1.0
    occ = post @ onehot
    grad = np.exp(lp) - occ
    return float(-logz), grad.astype(np.float32)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    t = np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3))
    dt = (1 - t * t) * 0.7978845608028654 * (1 + 3 * 0.044715 * x * x)
    return 0.5 * (1 + t) + 0.5 * x * dt


@dataclass
class Config:
    rank: int = 0
    steps: int = 150
    lr: float = 1e-3
    l2: float = 1e-2
    l2_adapter: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 0


@dataclass
class Head:
    W: np.ndarray
    b: np.ndarray
    A: np.ndarray | None = None
    B: np.ndarray | None = None

    def hidden(self, f):
        if self.A is None:
            return f, None
        pre = f @ self.A.T
        return f + gelu(pre) @ self.B.T, pre

    def logits(self, f):
        h, _ = self.hidden(f)
        return h @ self.W.T + self.b

    def logprobs(self, f):
        return log_softmax(self.logits(f))


def init_head(W0, b0, cfg: Config) -> Head:
    head = Head(W0.copy(), b0.copy())
    if cfg.rank:
        rng = np.random.default_rng(cfg.seed)
        d = W0.shape[1]
        head.A = (rng.standard_normal((cfg.rank, d)) / math.sqrt(d)).astype(np.float32)
        head.B = np.zeros((d, cfg.rank), np.float32)
    return head


def loss_and_grads(head: Head, W0, b0, samples, cfg: Config):
    """samples: list of (features (T, D), label). Returns loss and grads dict."""
    g = {"W": np.zeros_like(head.W), "b": np.zeros_like(head.b)}
    if head.A is not None:
        g["A"], g["B"] = np.zeros_like(head.A), np.zeros_like(head.B)
    total, n = 0.0, 0
    for f, label in samples:
        h, pre = head.hidden(f)
        z = h @ head.W.T + head.b
        loss, dz = ctc_loss_grad(z, label)
        if loss == 0.0 and not dz.any():
            continue
        scale = 1.0 / max(1, len(label))
        total += loss * scale
        n += 1
        dz *= scale
        g["W"] += dz.T @ h
        g["b"] += dz.sum(axis=0)
        if head.A is not None:
            dh = dz @ head.W
            act = gelu(pre)
            g["B"] += dh.T @ act
            dpre = (dh @ head.B) * gelu_grad(pre)
            g["A"] += dpre.T @ f
    n = max(n, 1)
    for k in g:
        g[k] /= n
    total /= n
    total += cfg.l2 * (float(((head.W - W0) ** 2).sum()) + float(((head.b - b0) ** 2).sum()))
    g["W"] += 2 * cfg.l2 * (head.W - W0)
    g["b"] += 2 * cfg.l2 * (head.b - b0)
    if head.A is not None:
        total += cfg.l2_adapter * (float((head.A ** 2).sum()) + float((head.B ** 2).sum()))
        g["A"] += 2 * cfg.l2_adapter * head.A
        g["B"] += 2 * cfg.l2_adapter * head.B
    return total, g


def train(W0, b0, samples, cfg: Config, on_step=None) -> Head:
    """Full-batch Adam over the cached features."""
    head = init_head(W0, b0, cfg)
    names = ["W", "b"] + (["A", "B"] if head.A is not None else [])
    m = {k: np.zeros_like(getattr(head, k)) for k in names}
    v = {k: np.zeros_like(getattr(head, k)) for k in names}
    for step in range(1, cfg.steps + 1):
        loss, g = loss_and_grads(head, W0, b0, samples, cfg)
        for k in names:
            m[k] = cfg.beta1 * m[k] + (1 - cfg.beta1) * g[k]
            v[k] = cfg.beta2 * v[k] + (1 - cfg.beta2) * g[k] ** 2
            mh = m[k] / (1 - cfg.beta1 ** step)
            vh = v[k] / (1 - cfg.beta2 ** step)
            setattr(head, k, (getattr(head, k) - cfg.lr * mh / (np.sqrt(vh) + cfg.eps)).astype(np.float32))
        if on_step:
            on_step(step, loss)
    return head
