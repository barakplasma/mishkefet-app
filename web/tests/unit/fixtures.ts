/** Loading the shared spec/ vectors and model artifacts for tests. */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseCharset, type Charset } from '../../src/engine/charset.ts'
import type { LogProbs } from '../../src/engine/decoder.ts'
import { parseCharLM, parseWordLM, type CharLM, type WordLM } from '../../src/engine/lm.ts'
import { MODEL_DIR, SPEC } from './paths.ts'

export function golden<T = any>(name: string): T {
  return JSON.parse(readFileSync(join(SPEC, 'golden', name), 'utf8'))
}

export function stem(file: string): string {
  return file.split('/').pop()!.replace(/\.[^.]+$/, '')
}

export function charset(): Charset {
  return parseCharset(readFileSync(join(SPEC, 'charset.json'), 'utf8'))
}

export function goldenLogProbs(file: string): LogProbs {
  const b = readFileSync(join(SPEC, 'golden/logprobs', `${stem(file)}.f32`))
  const data = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length))
  return { data, frames: data.length / 198, classes: 198 }
}

let lms: { lm: CharLM; words: WordLM } | null = null
export function languageModels(): { lm: CharLM; words: WordLM } {
  lms ??= {
    lm: parseCharLM(readFileSync(join(MODEL_DIR, 'charlm.bin'))),
    words: parseWordLM(readFileSync(join(MODEL_DIR, 'words.bin')), charset()),
  }
  return lms
}
