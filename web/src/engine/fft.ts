/**
 * Radix-2 complex FFT in double, used to correlate image rows with large
 * Gaussian kernels (a page background blur has 300+ taps). The result equals
 * direct summation to ~1e-13, far below the uint8 rounding that follows.
 */

export interface FFT {
  n: number
  /** In place; `inverse` includes the 1/n scaling. */
  transform(re: Float64Array, im: Float64Array, inverse: boolean): void
}

export function createFFT(n: number): FFT {
  if (n < 1 || (n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`)
  const bits = Math.log2(n)
  const rev = new Uint32Array(n)
  for (let i = 0; i < n; i++) {
    let r = 0
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b)
    rev[i] = r
  }
  const cos = new Float64Array(n / 2)
  const sin = new Float64Array(n / 2)
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n)
    sin[i] = Math.sin((2 * Math.PI * i) / n)
  }
  return {
    n,
    transform(re, im, inverse) {
      for (let i = 0; i < n; i++) {
        const j = rev[i]
        if (j > i) {
          let t = re[i]
          re[i] = re[j]
          re[j] = t
          t = im[i]
          im[i] = im[j]
          im[j] = t
        }
      }
      const sign = inverse ? 1 : -1
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1
        const step = n / size
        for (let start = 0; start < n; start += size) {
          for (let k = 0; k < half; k++) {
            const wr = cos[k * step]
            const wi = sign * sin[k * step]
            const a = start + k
            const b = a + half
            const xr = re[b] * wr - im[b] * wi
            const xi = re[b] * wi + im[b] * wr
            re[b] = re[a] - xr
            im[b] = im[a] - xi
            re[a] += xr
            im[a] += xi
          }
        }
      }
      if (inverse) {
        for (let i = 0; i < n; i++) {
          re[i] /= n
          im[i] /= n
        }
      }
    },
  }
}
