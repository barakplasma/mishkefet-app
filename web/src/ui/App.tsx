/**
 * Mishkefet web app. Screens follow the design prototype: HOME, CAMERA,
 * PROCESSING, RESULT, LIST, TRAIN INTRO, TRAIN, REFIT, COMPARE. Scans,
 * training samples and fine-tuned heads live in IndexedDB on this device.
 */

import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import type { HeadData, LineBox, LineResult } from '../engine/api.ts'
import { db, newId, type ModelId, type Scan, type Settings } from '../lib/db.ts'
import { CancelledError, getEngine, type ModelState } from '../lib/engine.ts'
import { blobFromRGBA, imageData, picturesFromFile, thumbnail, type Picture } from '../lib/images.ts'
import { defaultTitle, keepLine } from '../lib/text.ts'
import { currentHead, headForModel } from '../lib/training.ts'
import { Camera } from './Camera.tsx'
import { Compare, type CompareState } from './Compare.tsx'
import { Home } from './Home.tsx'
import { Processing, type ProcState } from './Processing.tsx'
import { Refit } from './Refit.tsx'
import { Result } from './Result.tsx'
import { ScanList } from './ScanList.tsx'
import { AboutSheet, ComparePickSheet, ImportSheet } from './Sheets.tsx'
import { Train, TrainIntro } from './Train.tsx'

export type Screen = 'home' | 'camera' | 'proc' | 'result' | 'list' | 'intro' | 'train' | 'refit' | 'compare'
type Sheet = null | 'import' | 'compare' | 'about'

const DEPTH: Record<Screen, number> = { home: 0, list: 1, camera: 1, intro: 1, proc: 2, train: 2, result: 3, refit: 3, compare: 4 }

function screenStyle(k: Screen, screen: Screen): CSSProperties {
  if (k === screen) return { transform: 'none', opacity: 1, pointerEvents: 'auto', zIndex: 2 }
  if (k === 'proc' || k === 'refit') return { transform: 'scale(.94)', opacity: 0, pointerEvents: 'none' }
  if (DEPTH[k] < DEPTH[screen]) return { transform: 'translateX(30%)', opacity: 0, pointerEvents: 'none' }
  return { transform: 'translateX(-100%)', opacity: 0, pointerEvents: 'none' }
}

/** Which head reads a scan: the user's choice, fine-tuned by default when one exists. */
export interface ModelChoice {
  tuned: { round: number; head: HeadData } | null
  useTuned: boolean
}

export const modelIdOf = (c: ModelChoice): ModelId => (c.useTuned && c.tuned ? `r${c.tuned.round}` : 'base')

export function App() {
  const engine = getEngine()
  const [model, setModel] = useState<ModelState>(engine.state)
  const [screen, setScreen] = useState<Screen>('home')
  const [sheet, setSheet] = useState<Sheet>(null)
  const [scans, setScans] = useState<Scan[]>([])
  const [settings, setSettings] = useState<Settings | null>(null)
  const [choice, setChoice] = useState<ModelChoice>({ tuned: null, useTuned: true })
  const [openId, setOpenId] = useState<string | null>(null)
  const [proc, setProc] = useState<ProcState | null>(null)
  const [compare, setCompare] = useState<CompareState | null>(null)
  const [toast, setToast] = useState<{ text: string; key: number } | null>(null)
  const [roundCount, setRoundCount] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const cancelRef = useRef<() => void>(() => {})
  const toastTimer = useRef(0)

  const showToast = useCallback((text: string) => {
    window.clearTimeout(toastTimer.current)
    setToast({ text, key: Date.now() })
    toastTimer.current = window.setTimeout(() => setToast(null), 2400)
  }, [])

  const refresh = useCallback(async () => {
    const [all, st, tuned, samples] = await Promise.all([db.scans(), db.settings(), currentHead(), db.samples()])
    setScans(all)
    setSettings(st)
    setChoice({ tuned, useTuned: st.preferTuned })
    setRoundCount(samples.filter((s) => s.round === st.rounds + 1 && s.strokes).length)
  }, [])

  useEffect(() => {
    const off = engine.subscribe(setModel)
    void engine.load().catch(() => undefined)
    void refresh()
    try {
      void navigator.storage?.persist?.()
    } catch {
      /* not supported */
    }
    return () => {
      off()
    }
  }, [engine, refresh])

  useEffect(() => {
    const d = document.documentElement.dataset
    d.mkModel = model.phase
    if (model.backend) d.mkBackend = model.backend
  }, [model.phase, model.backend])

  const setUseTuned = useCallback(
    async (useTuned: boolean) => {
      setChoice((c) => ({ ...c, useTuned }))
      const st = await db.settings()
      await db.saveSettings({ ...st, preferTuned: useTuned })
    },
    [],
  )

  // ------------------------------------------------------------ reading
  /** Reads one picture into a new scan, driving the processing screen. */
  const readPicture = useCallback(
    async (pic: Picture, kind: 'line' | 'page', index: number, count: number): Promise<Scan | null> => {
      const url = URL.createObjectURL(pic.blob)
      const head = choice.useTuned ? choice.tuned?.head : undefined
      setProc({ image: url, width: pic.width, height: pic.height, boxes: [], done: 0, total: 0, lastText: '', picture: index + 1, pictures: count, kind })
      const rgba = await imageData(pic.blob)
      let lines: LineResult[]
      // a page is stored as the engine prepared it: cropped to the paper, upright, deskewed
      let stored = { blob: pic.blob, width: pic.width, height: pic.height }
      let prepared: Promise<void> = Promise.resolve()
      try {
        if (kind === 'line') {
          const r = await engine.readLine(rgba, head)
          lines = [{ ...r, box: { x: 0, y: 0, w: pic.width, h: pic.height } }]
          setProc((p) => p && { ...p, boxes: [{ x: 0, y: 0, w: pic.width, h: pic.height }], total: 1, done: 1, lastText: r.text })
        } else {
          const job = engine.readPage(rgba, head, {
            segmented: (page) => {
              prepared = blobFromRGBA(page.image.rgba, page.image.width, page.image.height).then((blob) => {
                stored = { blob, width: page.image.width, height: page.image.height }
                const shown = URL.createObjectURL(blob)
                setProc((p) => p && { ...p, image: shown, width: stored.width, height: stored.height })
              })
              setProc((p) => p && { ...p, boxes: page.boxes, total: page.boxes.length })
            },
            line: (i, total, r) => setProc((p) => p && { ...p, done: i + 1, total, lastText: r.text || p.lastText }),
          })
          cancelRef.current = job.cancel
          lines = await job.done
          await prepared
        }
      } finally {
        cancelRef.current = () => {}
      }
      const now = Date.now()
      const kept = lines.filter(keepLine)
      const scan: Scan = {
        id: newId(), title: defaultTitle(kept.map((l) => l.text), now) || pic.name, createdAt: now, kind,
        image: stored.blob, thumb: await thumbnail(stored.blob), width: stored.width, height: stored.height,
        readings: [{ model: modelIdOf(choice), lines: kept, createdAt: now }], current: 0,
      }
      await db.putScan(scan)
      URL.revokeObjectURL(url)
      return scan
    },
    [choice, engine],
  )

  const readPictures = useCallback(
    async (pics: Picture[], kind: 'line' | 'page') => {
      if (!pics.length) return
      setSheet(null)
      setScreen('proc')
      let last: Scan | null = null
      try {
        for (let i = 0; i < pics.length; i++) last = await readPicture(pics[i], kind, i, pics.length)
      } catch (e) {
        if (!(e instanceof CancelledError)) showToast('משהו השתבש בקריאה. נסו שוב.')
        setProc(null)
        setScreen('home')
        return
      }
      await refresh()
      setProc(null)
      if (last) {
        setOpenId(last.id)
        setScreen(pics.length > 1 ? 'list' : 'result')
      }
    },
    [readPicture, refresh, showToast],
  )

  const onFiles = useCallback(
    async (files: FileList | null) => {
      if (!files?.length) return
      const pics: Picture[] = []
      for (const f of Array.from(files)) {
        try {
          pics.push(...(await picturesFromFile(f)))
        } catch {
          showToast(`לא הצלחתי לפתוח את ${f.name}`)
        }
      }
      void readPictures(pics, 'page')
    },
    [readPictures, showToast],
  )

  const pickFiles = useCallback(() => fileInput.current?.click(), [])

  /** Re-read an existing scan with a given model (toggle on the result, or compare after a refit). */
  const reread = useCallback(
    async (scan: Scan, modelId: ModelId): Promise<LineResult[]> => {
      const head = await headForModel(modelId)
      const rgba = await imageData(scan.image)
      if (scan.kind === 'line') {
        const r = await engine.readLine(rgba, head)
        return [{ ...r, box: { x: 0, y: 0, w: scan.width, h: scan.height } }]
      }
      // same boxes on the stored (prepared) page, so old and new lines pair by box
      const boxes = scan.readings[scan.current].lines.map((l) => l.box).filter((b): b is LineBox => !!b)
      if (!boxes.length) return (await engine.readPage(rgba, head, {}).done).filter(keepLine)
      return engine.readBoxes(rgba, boxes, head)
    },
    [engine],
  )

  const openScan = scans.find((s) => s.id === openId) ?? null
  const hasTuned = !!choice.tuned

  const go = (s: Screen) => {
    setSheet(null)
    setScreen(s)
  }

  return (
    <div className="app" dir="rtl" lang="he">
      <input ref={fileInput} type="file" multiple accept="image/*,.heic,.heif,application/pdf,.pdf" hidden onChange={(e) => {
        void onFiles(e.target.files)
        e.target.value = ''
      }} />

      <div className="screen" style={screenStyle('home', screen)}>
        <Home
          model={model} settings={settings} roundCount={roundCount} hasScans={scans.length > 0}
          onList={() => go('list')} onImport={() => setSheet('import')} onCamera={() => go('camera')} onGallery={pickFiles}
          onTrain={() => go(settings?.seenIntro ? 'train' : 'intro')} onAbout={() => setSheet('about')}
        />
      </div>

      <div className="screen camera" style={screenStyle('camera', screen)}>
        {screen === 'camera' && (
          <Camera
            choice={choice} onUseTuned={setUseTuned}
            onClose={() => go('home')} onGallery={pickFiles}
            onShot={(pic, kind) => void readPictures([pic], kind)}
          />
        )}
      </div>

      <div className="screen" style={screenStyle('proc', screen)}>
        {proc && <Processing state={proc} model={model} onCancel={() => {
          cancelRef.current()
          setScreen('home')
        }} />}
      </div>

      <div className="screen" style={screenStyle('result', screen)}>
        {openScan && (
          <Result
            key={openScan.id} scan={openScan} choice={choice} roundCount={roundCount}
            onBack={() => go('list')} onSaved={refresh} toast={showToast} reread={reread}
            onDeleted={async () => {
              await db.deleteScan(openScan.id)
              await refresh()
              go('list')
              showToast('הסריקה נמחקה')
            }}
          />
        )}
      </div>

      <div className="screen" style={screenStyle('list', screen)}>
        <ScanList scans={scans} onBack={() => go('home')} onOpen={(id) => {
          setOpenId(id)
          go('result')
        }} onDelete={async (ids) => {
          for (const id of ids) await db.deleteScan(id)
          await refresh()
          showToast(ids.length === 1 ? 'סריקה אחת נמחקה' : `${ids.length} סריקות נמחקו`)
        }} onImport={() => setSheet('import')} />
      </div>

      <div className="screen" style={screenStyle('intro', screen)}>
        <TrainIntro onClose={() => go('home')} onStart={async () => {
          const st = await db.settings()
          await db.saveSettings({ ...st, seenIntro: true })
          await refresh()
          go('train')
        }} />
      </div>

      <div className="screen" style={screenStyle('train', screen)}>
        {screen === 'train' && settings && (
          <Train
            engine={engine} settings={settings} choice={choice} scans={scans}
            onClose={async () => {
              await refresh()
              go('home')
            }}
            onCount={setRoundCount}
            onRoundComplete={async () => {
              await refresh()
              go('refit')
            }}
          />
        )}
      </div>

      <div className="screen dark" style={screenStyle('refit', screen)}>
        {screen === 'refit' && (
          <Refit
            engine={engine}
            onDone={async () => {
              await refresh()
              go('train')
            }}
            onCompare={async () => {
              await refresh()
              setSheet('compare')
            }}
          />
        )}
      </div>

      <div className="screen" style={screenStyle('compare', screen)}>
        {compare && openScan && (
          <Compare
            state={compare} scan={openScan}
            onBack={() => go('list')}
            onKeep={async (keepNew) => {
              if (keepNew && compare.after) {
                const readings = [...openScan.readings, { model: compare.model, lines: compare.after, createdAt: Date.now() }]
                await db.putScan({ ...openScan, readings, current: readings.length - 1, reread: true })
                await refresh()
                go('result')
                showToast('הקריאה החדשה נשמרה')
              } else {
                go('list')
                showToast('הקריאה הקודמת נשארה')
              }
              setCompare(null)
            }}
          />
        )}
      </div>

      {sheet && <div className="scrim" onClick={() => setSheet(null)} />}
      {sheet === 'import' && (
        <ImportSheet hasTuned={hasTuned} useTuned={choice.useTuned} onUseTuned={setUseTuned}
          onCamera={() => go('camera')} onFiles={pickFiles} />
      )}
      {sheet === 'compare' && (
        <ComparePickSheet scans={scans} onPick={async (scan) => {
          setSheet(null)
          const tuned = await currentHead()
          if (!tuned) return
          setOpenId(scan.id)
          const model: ModelId = `r${tuned.round}`
          const before = scan.readings[scan.current]
          setCompare({ model, before: before.lines, after: null, mode: 'after', beforeModel: before.model })
          setScreen('compare')
          try {
            const after = await reread(scan, model)
            setCompare((c) => c && { ...c, after })
          } catch {
            showToast('משהו השתבש בקריאה. נסו שוב.')
            go('list')
          }
        }} />
      )}
      {sheet === 'about' && (
        <AboutSheet model={model} settings={settings} hasTuned={hasTuned} onForget={async () => {
          await db.forgetHandwriting()
          await refresh()
          setSheet(null)
          showToast('הדוגמאות וההתאמה נמחקו')
        }} />
      )}

      {toast && <div className="toast" key={toast.key}>{toast.text}</div>}
    </div>
  )
}
