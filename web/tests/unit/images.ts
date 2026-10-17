/**
 * Fixture image decoding for Node tests (the app decodes with the browser).
 *
 * JPEGs go through MozJPEG's decoder (libjpeg-turbo: islow IDCT, fancy
 * upsampling), the same decoder family PIL uses for the golden vectors; a
 * different JPEG decoder is off by a level or two on half the pixels, which
 * is enough to move 0.3% of a binarized page.
 */

import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { PNG } from 'pngjs'
import { rgbaToGray, type Gray } from '../../src/engine/image.ts'

export interface Rgba {
  data: Uint8Array
  width: number
  height: number
}

let jpegReady: Promise<(b: ArrayBuffer) => Promise<{ data: Uint8ClampedArray; width: number; height: number }>> | null = null

function jpegDecoder() {
  jpegReady ??= (async () => {
    const g = globalThis as { ImageData?: unknown }
    g.ImageData ??= class {
      data: Uint8ClampedArray
      width: number
      height: number
      constructor(data: Uint8ClampedArray, width: number, height: number) {
        this.data = data
        this.width = width
        this.height = height
      }
    }
    const mod = await import('@jsquash/jpeg/decode.js')
    const wasmPath = createRequire(import.meta.url).resolve('@jsquash/jpeg/codec/dec/mozjpeg_dec.wasm')
    await mod.init(new WebAssembly.Module(readFileSync(wasmPath)) as never)
    return (b: ArrayBuffer) => mod.default(b) as Promise<{ data: Uint8ClampedArray; width: number; height: number }>
  })()
  return jpegReady
}

export async function loadRgba(path: string): Promise<Rgba> {
  const buf = readFileSync(path)
  if (/\.jpe?g$/i.test(path)) {
    const decode = await jpegDecoder()
    const img = await decode(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) as ArrayBuffer)
    return { data: new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.length), width: img.width, height: img.height }
  }
  const png = PNG.sync.read(buf)
  return { data: new Uint8Array(png.data.buffer, png.data.byteOffset, png.data.length), width: png.width, height: png.height }
}

export async function loadGray(path: string): Promise<Gray> {
  const { data, width, height } = await loadRgba(path)
  return rgbaToGray(data, width, height)
}

/** First channel of a PNG (the golden masks and renders are 8-bit gray). */
export function loadPngChannel(path: string): Gray {
  const png = PNG.sync.read(readFileSync(path))
  const data = new Uint8Array(png.width * png.height)
  for (let i = 0; i < data.length; i++) data[i] = png.data[i * 4]
  return { data, width: png.width, height: png.height }
}
