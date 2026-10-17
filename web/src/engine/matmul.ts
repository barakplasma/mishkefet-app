/**
 * Float32 matrix products for the trainer, run by ONNX Runtime when one is
 * available. The trainer's two products (logits and the weight gradient) are
 * about 0.5 GMAC each per step for a 40-sample round; plain JS loops manage
 * ~2.5 GMAC/s, ORT's SIMD WASM kernels several times that and WebGPU far more.
 *
 * The model is a single MatMul node with symbolic shapes, written here as
 * protobuf bytes so no file has to be shipped.
 */

import type { InferenceSession } from 'onnxruntime-common'
import type { OrtModule } from './model.ts'

/** C (m x n) = A (m x k) B (k x n), row-major. */
export type MatMul = (a: Float32Array, b: Float32Array, m: number, k: number, n: number) => Promise<Float32Array>

function varint(v: number): number[] {
  const out: number[] = []
  while (v > 127) {
    out.push((v & 127) | 128)
    v = Math.floor(v / 128)
  }
  out.push(v)
  return out
}

function field(no: number, payload: number[] | string): number[] {
  const bytes = typeof payload === 'string' ? [...new TextEncoder().encode(payload)] : payload
  return [...varint((no << 3) | 2), ...varint(bytes.length), ...bytes]
}

function intField(no: number, v: number): number[] {
  return [...varint(no << 3), ...varint(v)]
}

/** ValueInfoProto: float tensor with symbolic 2-D shape. */
function valueInfo(name: string, d0: string, d1: string): number[] {
  const dim = (p: string) => field(1, field(2, p))
  const shape = field(2, [...dim(d0), ...dim(d1)])
  const tensor = field(1, [...intField(1, 1), ...shape]) // elem_type FLOAT
  return [...field(1, name), ...field(2, tensor)]
}

/** ONNX ModelProto bytes for `c = MatMul(a, b)`. */
export function matmulModel(): Uint8Array {
  const node = [...field(1, 'a'), ...field(1, 'b'), ...field(2, 'c'), ...field(3, 'mm'), ...field(4, 'MatMul')]
  const graph = [
    ...field(1, node),
    ...field(2, 'matmul'),
    ...field(11, valueInfo('a', 'm', 'k')),
    ...field(11, valueInfo('b', 'k', 'n')),
    ...field(12, valueInfo('c', 'm', 'n')),
  ]
  const opset = [...field(1, ''), ...intField(2, 13)]
  return Uint8Array.from([...intField(1, 7), ...field(7, graph), ...field(8, opset)])
}

export function createOrtMatMul(ort: OrtModule, session: InferenceSession): MatMul {
  return async (a, b, m, k, n) => {
    const res = await session.run({
      a: new ort.Tensor('float32', a, [m, k]),
      b: new ort.Tensor('float32', b, [k, n]),
    })
    const t = res.c
    const out = new Float32Array((await t.getData()) as Float32Array)
    t.dispose?.()
    if (out.length !== m * n) throw new Error(`MatMul returned ${out.length} values, expected ${m * n}`)
    return out
  }
}
