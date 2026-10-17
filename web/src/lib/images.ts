/**
 * Getting pictures into the app: files (JPEG, PNG, HEIC where the browser can
 * decode it, PDF pages), camera frames, thumbnails, and line crops.
 */

import type { LineBox } from '../engine/api.ts'

/** Pages are read at up to this long side; the segmenter works at 2000 anyway. */
const MAX_SIDE = 3200
const THUMB = 160

export const ACCEPT = 'image/*,.heic,.heif,application/pdf,.pdf'

export interface Picture {
  blob: Blob
  width: number
  height: number
  name: string
}

/** Honors EXIF orientation (createImageBitmap does by default, the explicit option is for older Safari). */
async function bitmap(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob, { imageOrientation: 'from-image' } as ImageBitmapOptions)
}

function canvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h)
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  return c
}

async function toBlob(c: OffscreenCanvas | HTMLCanvasElement, type = 'image/jpeg', quality = 0.9): Promise<Blob> {
  if ('convertToBlob' in c) return c.convertToBlob({ type, quality })
  return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('encode failed'))), type, quality))
}

function ctx2d(c: OffscreenCanvas | HTMLCanvasElement) {
  return c.getContext('2d') as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D
}

/** Draws onto white (transparent PNGs read as paper, not black) at most MAX_SIDE. */
async function normalize(source: CanvasImageSource, w: number, h: number, name: string): Promise<Picture> {
  const s = Math.min(1, MAX_SIDE / Math.max(w, h))
  const W = Math.max(1, Math.round(w * s))
  const H = Math.max(1, Math.round(h * s))
  const c = canvas(W, H)
  const g = ctx2d(c)
  g.fillStyle = '#fff'
  g.fillRect(0, 0, W, H)
  g.imageSmoothingQuality = 'high'
  g.drawImage(source, 0, 0, W, H)
  return { blob: await toBlob(c), width: W, height: H, name }
}

/** One file can be several pictures (a PDF's pages). */
export async function picturesFromFile(file: File): Promise<Picture[]> {
  const name = file.name.replace(/\.[^.]+$/, '')
  if (file.type === 'application/pdf' || /\.pdf$/i.test(file.name)) return pdfPages(file, name)
  const bmp = await bitmap(file)
  try {
    return [await normalize(bmp, bmp.width, bmp.height, name)]
  } finally {
    bmp.close()
  }
}

async function pdfPages(file: File, name: string): Promise<Picture[]> {
  const pdfjs = await import('pdfjs-dist')
  const workerUrl = (await import('pdfjs-dist/build/pdf.worker.min.mjs?url')).default
  pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
  const doc = await pdfjs.getDocument({ data: new Uint8Array(await file.arrayBuffer()) }).promise
  const out: Picture[] = []
  for (let p = 1; p <= Math.min(doc.numPages, 20); p++) {
    const page = await doc.getPage(p)
    const base = page.getViewport({ scale: 1 })
    const scale = Math.min(MAX_SIDE / Math.max(base.width, base.height), 4)
    const vp = page.getViewport({ scale })
    const c = document.createElement('canvas')
    c.width = Math.round(vp.width)
    c.height = Math.round(vp.height)
    const g = c.getContext('2d')!
    g.fillStyle = '#fff'
    g.fillRect(0, 0, c.width, c.height)
    await page.render({ canvasContext: g, viewport: vp, canvas: c }).promise
    out.push(await normalize(c, c.width, c.height, doc.numPages > 1 ? `${name} ${p}` : name))
  }
  return out
}

/** A camera frame, cropped to the guide rectangle (fractions of the frame). */
export async function pictureFromVideo(video: HTMLVideoElement, crop: { x: number; y: number; w: number; h: number }): Promise<Picture> {
  const vw = video.videoWidth
  const vh = video.videoHeight
  const sx = Math.round(crop.x * vw)
  const sy = Math.round(crop.y * vh)
  const sw = Math.round(crop.w * vw)
  const sh = Math.round(crop.h * vh)
  const c = canvas(sw, sh)
  ctx2d(c).drawImage(video, sx, sy, sw, sh, 0, 0, sw, sh)
  return normalize(c, sw, sh, '')
}

export async function imageData(blob: Blob): Promise<ImageData> {
  const bmp = await bitmap(blob)
  const c = canvas(bmp.width, bmp.height)
  const g = ctx2d(c)
  g.drawImage(bmp, 0, 0)
  bmp.close()
  return g.getImageData(0, 0, c.width, c.height)
}

export async function thumbnail(blob: Blob): Promise<Blob> {
  const bmp = await bitmap(blob)
  const s = THUMB / Math.max(bmp.width, bmp.height)
  const c = canvas(Math.max(1, Math.round(bmp.width * s)), Math.max(1, Math.round(bmp.height * s)))
  const g = ctx2d(c)
  g.imageSmoothingQuality = 'high'
  g.drawImage(bmp, 0, 0, c.width, c.height)
  bmp.close()
  return toBlob(c, 'image/jpeg', 0.8)
}

/** A line's pixels for a training sample, with the same padding the segmenter uses. */
export async function cropLine(blob: Blob, box: LineBox): Promise<ImageData> {
  const bmp = await bitmap(blob)
  const py = Math.trunc(box.h * 0.12)
  const px = Math.trunc(box.h * 0.05)
  const x0 = Math.max(0, box.x - px)
  const y0 = Math.max(0, box.y - py)
  const x1 = Math.min(bmp.width, box.x + box.w + px)
  const y1 = Math.min(bmp.height, box.y + box.h + py)
  const c = canvas(Math.max(1, x1 - x0), Math.max(1, y1 - y0))
  const g = ctx2d(c)
  g.drawImage(bmp, x0, y0, x1 - x0, y1 - y0, 0, 0, x1 - x0, y1 - y0)
  bmp.close()
  return g.getImageData(0, 0, c.width, c.height)
}

export function grayToDataUrl(gray: Uint8Array, width: number, height: number): string {
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  const g = c.getContext('2d')!
  const img = g.createImageData(width, height)
  for (let i = 0; i < gray.length; i++) {
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = gray[i]
    img.data[i * 4 + 3] = 255
  }
  g.putImageData(img, 0, 0)
  return c.toDataURL('image/png')
}
