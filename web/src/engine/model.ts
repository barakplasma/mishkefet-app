/**
 * The frozen backbone: (1, 1, 64, W) image to (W // 4, 512) features.
 *
 * DOM-free and runtime-agnostic: the ONNX Runtime module (onnxruntime-web in
 * the worker, onnxruntime-node in the golden script) and its session are
 * injected, so the browser and the Node harness run the same code. There is no
 * padding mask in the model, so every line runs alone at its true width.
 */

import type { InferenceSession, Tensor } from 'onnxruntime-common'
import { LINE_HEIGHT } from './constants.ts'

export interface OrtModule {
  Tensor: typeof Tensor
}

export interface Features {
  /** frames x dim, row-major (dim 512 for Hebrew, 768 for English). */
  values: Float32Array
  frames: number
}

export interface Backbone {
  features(image: Float32Array, width: number): Promise<Features>
}

export function createBackbone(ort: OrtModule, session: InferenceSession): Backbone {
  const input = session.inputNames[0] ?? 'image'
  const output = session.outputNames[0] ?? 'features'
  return {
    async features(image, width) {
      if (image.length !== LINE_HEIGHT * width) throw new Error(`image: expected 64 x ${width}`)
      const x = new ort.Tensor('float32', image, [1, 1, LINE_HEIGHT, width])
      const res = await session.run({ [input]: x })
      const t = res[output]
      // Own the memory: runtimes may reuse output buffers between runs.
      const values = new Float32Array((await t.getData()) as Float32Array)
      const frames = Number(t.dims[1])
      const dim = Number(t.dims[2])
      t.dispose?.()
      if (values.length !== frames * dim) throw new Error(`features: got ${values.length} values for ${frames} x ${dim}`)
      return { values, frames }
    },
  }
}
