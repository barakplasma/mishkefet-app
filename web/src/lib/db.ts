/**
 * Everything the web app keeps, in one IndexedDB database on this device:
 * scans (image + every reading of it), training samples, and the fine-tuned
 * heads. Nothing leaves the browser.
 */

import type { HeadData, LineBox, LineResult, Stroke } from '../engine/api.ts'

/** Which model produced a reading: the general one, or the fine-tuned head of a round. */
export type ModelId = 'base' | `r${number}`

export interface Reading {
  model: ModelId
  lines: LineResult[]
  createdAt: number
  /** The user corrected some lines by hand. */
  edited?: boolean
}

export interface Scan {
  id: string
  title: string
  createdAt: number
  kind: 'line' | 'page'
  image: Blob
  thumb: Blob
  width: number
  height: number
  readings: Reading[]
  /** Index into readings of the one shown. */
  current: number
  /** Re-read after fine-tuning and the new reading kept. */
  reread?: boolean
}

export interface Sample {
  id: string
  createdAt: number
  round: number
  word: string
  strokes?: Stroke[]
  /** A scan line the user corrected: the crop and the corrected text. */
  lineImage?: Blob
  readAs: string
  correct: boolean
  heldOut: boolean
}

export interface HeadRecord {
  round: number
  createdAt: number
  weight: ArrayBuffer
  bias: ArrayBuffer
  classes: number
  dim: number
  accBefore: number
  accAfter: number
  samples: number
}

export interface Settings {
  seenIntro: boolean
  /** Rounds completed (accepted or not). */
  rounds: number
  /** Round of the head in use, or 0 for the general model. */
  tunedRound: number
  /** The user's choice for new scans; defaults to the fine-tuned model when one exists. */
  preferTuned: boolean
}

const DEFAULTS: Settings = { seenIntro: false, rounds: 0, tunedRound: 0, preferTuned: true }

let dbp: Promise<IDBDatabase> | null = null

function open(): Promise<IDBDatabase> {
  if (!dbp) {
    dbp = new Promise((resolve, reject) => {
      const req = indexedDB.open('mishkefet', 1)
      req.onupgradeneeded = () => {
        const db = req.result
        db.createObjectStore('scans', { keyPath: 'id' })
        db.createObjectStore('samples', { keyPath: 'id' })
        db.createObjectStore('heads', { keyPath: 'round' })
        db.createObjectStore('kv')
        db.createObjectStore('features')
      }
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
  }
  return dbp
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode)
        const req = fn(t.objectStore(store))
        t.oncomplete = () => resolve(req.result)
        t.onerror = () => reject(t.error)
        t.onabort = () => reject(t.error)
      }),
  )
}

export const db = {
  scans: (): Promise<Scan[]> => tx<Scan[]>('scans', 'readonly', (s) => s.getAll()).then((a) => a.sort((x, y) => y.createdAt - x.createdAt)),
  scan: (id: string): Promise<Scan | undefined> => tx<Scan | undefined>('scans', 'readonly', (s) => s.get(id)),
  putScan: (scan: Scan) => tx('scans', 'readwrite', (s) => s.put(scan)),
  deleteScan: (id: string) => tx('scans', 'readwrite', (s) => s.delete(id)),

  samples: (): Promise<Sample[]> => tx<Sample[]>('samples', 'readonly', (s) => s.getAll()).then((a) => a.sort((x, y) => x.createdAt - y.createdAt)),
  putSample: (sample: Sample) => tx('samples', 'readwrite', (s) => s.put(sample)),

  heads: (): Promise<HeadRecord[]> => tx<HeadRecord[]>('heads', 'readonly', (s) => s.getAll()),
  head: (round: number): Promise<HeadRecord | undefined> => tx<HeadRecord | undefined>('heads', 'readonly', (s) => s.get(round)),
  putHead: (h: HeadRecord) => tx('heads', 'readwrite', (s) => s.put(h)),

  settings: async (): Promise<Settings> => ({ ...DEFAULTS, ...((await tx<Partial<Settings> | undefined>('kv', 'readonly', (s) => s.get('settings'))) ?? {}) }),
  saveSettings: (v: Settings) => tx('kv', 'readwrite', (s) => s.put(v, 'settings')),

  /** Backbone features of one training view, cached so later rounds only pay for new samples. */
  features: (key: string): Promise<{ values: ArrayBuffer; frames: number; label: number[] } | undefined> => tx('features', 'readonly', (s) => s.get(key)),
  putFeatures: (key: string, v: { values: ArrayBuffer; frames: number; label: number[] }) => tx('features', 'readwrite', (s) => s.put(v, key)),

  /** "Forget my handwriting": samples and every fine-tuned head. Scans stay. */
  async forgetHandwriting(): Promise<void> {
    await tx('samples', 'readwrite', (s) => s.clear())
    await tx('features', 'readwrite', (s) => s.clear())
    await tx('heads', 'readwrite', (s) => s.clear())
    const st = await db.settings()
    await db.saveSettings({ ...st, rounds: 0, tunedRound: 0 })
  },
}

export function headData(h: HeadRecord): HeadData {
  return { classes: h.classes, dim: h.dim, weight: new Float32Array(h.weight), bias: new Float32Array(h.bias) }
}

export function newId(): string {
  return crypto.randomUUID ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

export type { LineBox, LineResult }
