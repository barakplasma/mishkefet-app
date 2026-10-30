/**
 * Golden end-to-end check (opt-in: `npm run test:golden`).
 *
 * Runs the shared engine (src/engine/*, the same modules the browser worker
 * uses) with the real fp16 backbone and compares against spec/golden/ from
 * tools/model/golden.py:
 *   - every fixture line read with TTA 3: mean CER against lines.json "read" <= 0.02
 *     (also reported: log-prob drift against golden/logprobs, beam TTA 1, stroke reads)
 *   - each page read in full (paper crop, quarter turns, wide deskew, straightening):
 *     same paper (2 px), turns and angle; mean line CER against pages.json "read" <= 0.05
 *     with the same line count, or <= 0.08 and +-3 lines for turned-tilted and screenshot
 *   - trainer timing for a 40-sample round (40 x 2 views x 60 frames, 100 steps)
 *
 *   MK_MODEL_DIR=~/.cache/mishkefet/models/16380452 npm run test:golden
 *   MK_GOLDEN_RUNTIME=web npm run test:golden    # onnxruntime-web (WASM, 1 thread) in Node
 *   MK_SKIP_TRAIN=1 npm run test:golden           # skip the trainer timing
 *   MK_SKIP_PAGES=1 npm run test:golden           # skip the (slow on WASM) page reads
 */

import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'
import type { InferenceSession } from 'onnxruntime-common'
import type { HeadData, TrainView } from '../src/engine/api.ts'
import { labelsToText, parseCharset } from '../src/engine/charset.ts'
import { MODEL_FILES } from '../src/engine/constants.ts'
import { beamSearch, greedy } from '../src/engine/decoder.ts'
import { applyHead, parseHead } from '../src/engine/head.ts'
import { parseCharLM, parseWordLM } from '../src/engine/lm.ts'
import { createOrtMatMul, matmulModel } from '../src/engine/matmul.ts'
import { cer } from '../src/engine/metrics.ts'
import { createBackbone, type OrtModule } from '../src/engine/model.ts'
import { preprocess } from '../src/engine/preprocess.ts'
import { grayFeatures, readLine, readPage, renderWord, type Engine } from '../src/engine/reader.ts'
import { mulberry32 } from '../src/engine/strokes.ts'
import { createTrainer } from '../src/engine/trainer.ts'
import { loadGray, loadRgba } from '../tests/unit/images.ts'

const ROOT = resolve(import.meta.dirname, '..', '..')
const SPEC = join(ROOT, 'spec')
const MODEL_DIR = process.env.MK_MODEL_DIR ?? join(homedir(), '.cache/mishkefet/models/16380452')
const RUNTIME = process.env.MK_GOLDEN_RUNTIME ?? 'node'

type Ort = OrtModule & { InferenceSession: { create(b: Uint8Array, o?: object): Promise<InferenceSession> } }

const golden = (name: string) => JSON.parse(readFileSync(join(SPEC, 'golden', name), 'utf8'))
const ms = (t: number) => `${Math.round(performance.now() - t)} ms`

async function main() {
  const file = (p: string) => join(MODEL_DIR, p.split('/').pop()!)
  if (!existsSync(file(MODEL_FILES[0].path))) {
    console.error(`model not found in ${MODEL_DIR} (run tools/model/fetch-model.sh)`)
    process.exit(2)
  }
  const bytes = Object.fromEntries(MODEL_FILES.map((f) => [f.key, readFileSync(file(f.path))]))
  for (const f of MODEL_FILES) {
    const sha = createHash('sha256').update(bytes[f.key]).digest('hex')
    if (sha !== f.sha256) throw new Error(`${f.path}: sha256 ${sha} != pinned ${f.sha256}`)
  }

  let ort: Ort
  let opts: object
  if (RUNTIME === 'web') {
    const web = await import('onnxruntime-web')
    web.env.wasm.numThreads = 1
    ort = web as unknown as Ort
    opts = { executionProviders: ['wasm'] }
  } else {
    ort = (await import('onnxruntime-node')) as unknown as Ort
    opts = { executionProviders: ['cpu'], ...(process.env.MK_THREADS ? { intraOpNumThreads: Number(process.env.MK_THREADS) } : {}) }
  }
  let t = performance.now()
  const session = await ort.InferenceSession.create(new Uint8Array(bytes.backbone), opts)
  const charset = parseCharset(bytes.charset.toString('utf8'))
  const engine: Engine = {
    charset,
    lm: parseCharLM(bytes.charlm),
    words: parseWordLM(bytes.words, charset),
    backbone: createBackbone(ort, session),
    baseHead: parseHead(bytes.head),
  }
  console.log(`runtime=${RUNTIME} session + LMs ready in ${ms(t)}`)
  let failures = 0
  const check = (ok: boolean, msg: string) => {
    if (!ok) failures++
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`)
  }

  // ------------------------------------------------------------ lines
  console.log('\nlines (TTA 3 vs lines.json "read")')
  const lines = golden('lines.json') as Array<{ file: string; frames: number; read: string; beam: string; greedy: string }>
  let cerSum = 0
  let beamSame = 0
  let viewMs = 0
  let views = 0
  for (const line of lines) {
    const gray = await loadGray(join(SPEC, 'fixtures', line.file))
    // single view: log-prob drift and the TTA-1 beam
    const x = preprocess(gray)
    t = performance.now()
    const f = await engine.backbone.features(x.data, x.width)
    viewMs += performance.now() - t
    views++
    const lp = applyHead(f.values, f.frames, engine.baseHead)
    const want = readFileSync(join(SPEC, 'golden/logprobs', `${line.file.split('/').pop()!.replace('.png', '')}.f32`))
    const wantLp = new Float32Array(want.buffer.slice(want.byteOffset, want.byteOffset + want.length))
    let drift = 0
    for (let i = 0; i < wantLp.length; i++) drift = Math.max(drift, Math.abs(wantLp[i] - lp.data[i]))
    const beam = labelsToText(charset, beamSearch(lp, engine.lm!, engine.words!))
    if (beam === line.beam) beamSame++
    t = performance.now()
    const r = await readLine(engine, gray, { tta: 3 })
    const lineMs = performance.now() - t
    const c = cer(line.read, r.text)
    cerSum += c
    console.log(
      `  ${line.file}  W=${x.width} T=${f.frames}/${line.frames}  max|dlp|=${drift.toExponential(1)}  tta3 ${Math.round(lineMs)} ms  CER ${(c * 100).toFixed(1)}%${r.text === line.read ? ' (exact)' : ''}${beam === line.beam ? '' : '  [beam tta1 differs]'}`,
    )
    if (r.text !== line.read) console.log(`      got:      ${r.text}\n      expected: ${line.read}`)
  }
  const meanCer = cerSum / lines.length
  console.log(`  backbone: ${(viewMs / views).toFixed(0)} ms per view on average (${RUNTIME})`)
  check(meanCer <= 0.02, `line reads: mean CER ${(meanCer * 100).toFixed(2)}% (<= 2%), beam TTA 1 identical on ${beamSame}/${lines.length}`)

  // ------------------------------------------------------------ strokes
  const strokes = golden('strokes.json') as { sets: Array<{ word: string; strokes: Array<Array<[number, number]>>; read_greedy: string }> }
  let strokeSame = 0
  for (const s of strokes.sets) {
    const f = await grayFeatures(engine, renderWord(s.strokes, s.word)!)
    if (labelsToText(charset, greedy(applyHead(f.values, f.frames, engine.baseHead))) === s.read_greedy) strokeSame++
  }
  console.log(`info stroke renders: greedy read identical on ${strokeSame}/${strokes.sets.length}`)

  // ------------------------------------------------------------ pages
  // The full read: paper crop, quarter turns (model vote), wide deskew, straightened lines.
  type GoldenRead = { paper: number[] | null; turns: number; angle: number; deskewed: [number, number]; boxes: unknown[]; lines: string[] }
  const pages = golden('pages.json') as Array<{ file: string; read: GoldenRead }>
  // pages built for this repo to exercise turning, tilt and screenshots get a looser line gate
  const loose = (f: string) => /turned-tilted|screenshot/.test(f)
  for (const page of process.env.MK_SKIP_PAGES ? [] : pages) {
    const rgba = await loadRgba(join(SPEC, 'fixtures', page.file))
    const g = page.read
    t = performance.now()
    let segMs = 0
    const res = await readPage(engine, { rgba: rgba.data, width: rgba.width, height: rgba.height }, { tta: 3 }, { onSegmented: () => (segMs = performance.now() - t) })
    const total = performance.now() - t
    const got = res.lines.map((l) => l.text)
    const n = Math.min(got.length, g.lines.length)
    let sum = 0
    for (let i = 0; i < n; i++) sum += cer(g.lines[i], got[i])
    sum += Math.abs(got.length - g.lines.length) // a missing or extra line counts as 100%
    const mean = sum / Math.max(got.length, g.lines.length, 1)
    const paper = res.paper ? [res.paper.x, res.paper.y, res.paper.w, res.paper.h] : null
    const paperOk = !paper === !g.paper && (!paper || paper.every((v, i) => Math.abs(v - g.paper![i]) <= 2))
    const size = [res.seg.deskewed.width, res.seg.deskewed.height]
    const limit = loose(page.file) ? 0.08 : 0.05
    const countOk = Math.abs(got.length - g.lines.length) <= (loose(page.file) ? 3 : 0)
    check(
      paperOk && res.turns === g.turns && res.seg.angle === g.angle && countOk && mean <= limit,
      `${page.file}: paper ${JSON.stringify(paper)}, turns ${res.turns}/${g.turns}, angle ${res.seg.angle}/${g.angle}, deskewed ${size.join('x')}/${g.deskewed.join('x')}, ${got.length}/${g.lines.length} lines, orient+segment ${Math.round(segMs)} ms, total ${(total / 1000).toFixed(1)} s, mean line CER ${(mean * 100).toFixed(2)}% (<= ${limit * 100}%)`,
    )
  }

  // ------------------------------------------------------------ trainer
  if (!process.env.MK_SKIP_TRAIN) {
    console.log('\ntrainer: 40 samples x 2 views x 60 frames, 100 steps')
    const r = mulberry32(7)
    const tv: TrainView[] = Array.from({ length: 80 }, () => ({
      frames: 60,
      features: Float32Array.from({ length: 60 * 512 }, () => r() * 2 - 1),
      label: Array.from({ length: 6 }, () => 1 + Math.floor(r() * 27)),
    }))
    const run = async (label: string, matmul?: Parameters<typeof createTrainer>[2]['matmul'], steps = 100) => {
      const tr = createTrainer(engine.baseHead as HeadData, tv, { lr: 1e-3, l2: 0.1, matmul })
      t = performance.now()
      let first = 0
      let last = 0
      for (let i = 0; i < steps; i++) {
        last = await tr.step()
        if (!i) first = last
      }
      const secs = (performance.now() - t) / 1000
      console.log(`  ${label}: ${((secs * 100) / steps).toFixed(1)} s per 100 steps (loss ${first.toFixed(3)} -> ${last.toFixed(3)})`)
      return last
    }
    const mmSession = await ort.InferenceSession.create(matmulModel(), RUNTIME === 'web' ? { executionProviders: ['wasm'] } : { intraOpNumThreads: 1 })
    const viaOrt = await run(`ORT MatMul (${RUNTIME}, 1 thread)`, createOrtMatMul(ort, mmSession))
    await run('JS loops', undefined, process.env.MK_FULL_JS ? 100 : 10)
    check(Number.isFinite(viaOrt), 'trainer ran')
  }

  console.log(failures ? `\n${failures} failure(s)` : '\nall golden checks passed')
  await session.release?.()
  process.exitCode = failures ? 1 : 0
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
