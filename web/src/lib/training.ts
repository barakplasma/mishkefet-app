/**
 * Fine-tuning rounds (spec/adapter.md): collect the user's samples, extract
 * backbone features for each training view (cached), train the head, measure
 * both heads on the held-out samples, keep the new head only if it is not worse.
 */

import adapt from '../../../spec/adapt.json'
import type { HeadData, Stroke, TrainView } from '../engine/api.ts'
import { db, headData, newId, type HeadRecord, type Sample } from './db.ts'
import type { EngineClient } from './engine.ts'
import { imageData } from './images.ts'

export const ROUND_SIZE: number = adapt.round_size
export const INACTIVITY_MS: number = adapt.inactivity_ms
const AUG: number = adapt.aug_views.web

export function fnv1a32(s: string): number {
  let h = 0x811c9dc5
  for (const byte of new TextEncoder().encode(s)) {
    h ^= byte
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h >>> 0
}

const isHeldOut = (id: string) => fnv1a32(id) % adapt.held_out_modulo === 0

export function augmentSeeds(id: string): number[] {
  const base = (fnv1a32(id) & 0xffffff) * 16
  return Array.from({ length: AUG }, (_, k) => base + k + 1)
}

export async function recordAttempt(round: number, word: string, strokes: Stroke[], readAs: string): Promise<Sample> {
  const id = newId()
  const s: Sample = { id, createdAt: Date.now(), round, word, strokes, readAs, correct: readAs.trim() === word, heldOut: isHeldOut(id) }
  await db.putSample(s)
  return s
}

/** A scan line the user corrected becomes a real-paper training sample. */
export async function recordLine(round: number, crop: Blob, text: string): Promise<void> {
  const id = newId()
  await db.putSample({ id, createdAt: Date.now(), round, word: text, lineImage: crop, readAs: '', correct: false, heldOut: false })
}

/** The head in use for "fine-tuned" reads, if any. */
export async function currentHead(): Promise<{ round: number; head: HeadData } | null> {
  const st = await db.settings()
  if (!st.tunedRound) return null
  const rec = await db.head(st.tunedRound)
  return rec ? { round: rec.round, head: headData(rec) } : null
}

export async function headForModel(model: string): Promise<HeadData | undefined> {
  if (model === 'base') return undefined
  const rec = await db.head(Number(model.slice(1)))
  return rec ? headData(rec) : undefined
}

export type RefitStage = 'collect' | 'tune' | 'test' | 'done'

export interface RefitResult {
  round: number
  accepted: boolean
  before: number
  after: number
}

async function viewsFor(engine: EngineClient, s: Sample): Promise<TrainView[]> {
  const seeds = s.strokes ? [0, ...augmentSeeds(s.id)] : [0]
  const cached = await Promise.all(seeds.map((seed) => db.features(`${s.id}:${seed}`)))
  if (cached.every(Boolean)) return cached.map((c) => ({ features: new Float32Array(c!.values), frames: c!.frames, label: c!.label }))
  const views = s.strokes
    ? (await engine.sampleFeatures(s.strokes, s.word, seeds.slice(1))).views
    : await engine.lineFeatures(await imageData(s.lineImage!), s.word)
  await Promise.all(views.map((v, k) => db.putFeatures(`${s.id}:${seeds[k] ?? k}`, { values: v.features.slice().buffer, frames: v.frames, label: v.label })))
  return views
}

export async function refit(engine: EngineClient, onStage: (stage: RefitStage, progress: number) => void): Promise<RefitResult> {
  const st = await db.settings()
  const round = st.rounds + 1
  const samples = await db.samples()
  const train = samples.filter((s) => !s.heldOut).slice(-adapt.max_train_samples)
  const held = samples.filter((s) => s.heldOut && s.strokes)

  onStage('collect', 0)
  const views: TrainView[] = []
  for (let i = 0; i < train.length; i++) {
    views.push(...(await viewsFor(engine, train[i])))
    onStage('collect', (i + 1) / train.length)
  }

  onStage('tune', 0)
  const head = await engine.train(views, { steps: adapt.steps, lr: adapt.lr, l2: adapt.l2 }, (step, steps) => onStage('tune', step / steps))

  onStage('test', 0)
  const previous = await currentHead()
  let before = 1
  let after = 1
  if (held.length) {
    const r = await engine.evaluate(held.map((s) => ({ strokes: s.strokes!, word: s.word })), previous?.head, head)
    before = r.before
    after = r.after
  }
  const accepted = after >= before - adapt.accept_if_accuracy_drop_at_most
  if (accepted) {
    const rec: HeadRecord = {
      round, createdAt: Date.now(), classes: head.classes, dim: head.dim,
      weight: head.weight.slice().buffer, bias: head.bias.slice().buffer,
      accBefore: before, accAfter: after, samples: train.length,
    }
    await db.putHead(rec)
  }
  await db.saveSettings({ ...st, rounds: round, tunedRound: accepted ? round : st.tunedRound, preferTuned: accepted ? true : st.preferTuned })
  onStage('done', 1)
  return { round, accepted, before, after }
}
