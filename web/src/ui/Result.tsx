import { useEffect, useMemo, useRef, useState } from 'react'
import type { LineResult } from '../engine/api.ts'
import { db, type ModelId, type Scan } from '../lib/db.ts'
import { cropLine } from '../lib/images.ts'
import { dateLabel, linesLabel } from '../lib/text.ts'
import { recordLine } from '../lib/training.ts'
import type { ModelChoice } from './App.tsx'
import { BackIcon, CheckIcon, CopyIcon, DownloadIcon, GalleryIcon, PencilIcon, ShareIcon, TrashIcon } from './icons.tsx'
import { ModelToggle } from './Sheets.tsx'

const IMG_H = 170

export function Result(props: {
  scan: Scan
  choice: ModelChoice
  roundCount: number
  onBack: () => void
  onSaved: () => Promise<void>
  onDeleted: () => void
  toast: (t: string) => void
  reread: (scan: Scan, model: ModelId) => Promise<LineResult[]>
}) {
  const { scan, choice } = props
  const reading = scan.readings[scan.current]
  const [active, setActive] = useState(0)
  const [showImage, setShowImage] = useState(true)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<string[]>([])
  const [title, setTitle] = useState(scan.title)
  const [copied, setCopied] = useState(false)
  const [busy, setBusy] = useState(false)
  const [frameW, setFrameW] = useState(360)
  const frame = useRef<HTMLDivElement>(null)
  const url = useMemo(() => URL.createObjectURL(scan.image), [scan.image])
  useEffect(() => () => URL.revokeObjectURL(url), [url])
  useEffect(() => {
    const el = frame.current
    if (!el) return
    const ro = new ResizeObserver(() => setFrameW(el.clientWidth))
    ro.observe(el)
    return () => ro.disconnect()
  }, [showImage])

  const lines = reading.lines
  const tunedId: ModelId | null = choice.tuned ? `r${choice.tuned.round}` : null
  const isTuned = reading.model !== 'base'

  const switchModel = async (useTuned: boolean) => {
    const target: ModelId = useTuned && tunedId ? tunedId : 'base'
    if (target === reading.model) return
    const existing = scan.readings.map((r, i) => [r, i] as const).filter(([r]) => r.model === target).pop()
    if (existing) {
      await db.putScan({ ...scan, current: existing[1] })
      await props.onSaved()
      return
    }
    setBusy(true)
    try {
      const fresh = await props.reread(scan, target)
      const readings = [...scan.readings, { model: target, lines: fresh, createdAt: Date.now() }]
      await db.putScan({ ...scan, readings, current: readings.length - 1 })
      await props.onSaved()
    } catch {
      props.toast('משהו השתבש בקריאה. נסו שוב.')
    } finally {
      setBusy(false)
    }
  }

  const startEdit = () => {
    setDraft(lines.map((l) => l.text))
    setTitle(scan.title)
    setEditing(true)
  }

  const saveEdit = async () => {
    const changed = draft.map((t, i) => (t.trim() !== lines[i].text.trim() ? i : -1)).filter((i) => i >= 0)
    const newLines = lines.map((l, i) => ({ ...l, text: draft[i].trim() }))
    const readings = scan.readings.map((r, i) => (i === scan.current ? { ...r, lines: newLines, edited: r.edited || changed.length > 0 } : r))
    await db.putScan({ ...scan, title: title.trim() || scan.title, readings })
    // corrected lines teach the model real paper ink (spec/training.md)
    const st = await db.settings()
    for (const i of changed) {
      const box = lines[i].box
      if (!box || !newLines[i].text) continue
      const crop = await cropLine(scan.image, box)
      const c = document.createElement('canvas')
      c.width = crop.width
      c.height = crop.height
      c.getContext('2d')!.putImageData(crop, 0, 0)
      const blob = await new Promise<Blob | null>((r) => c.toBlob(r, 'image/png'))
      if (blob) await recordLine(st.rounds + 1, blob, newLines[i].text)
    }
    setEditing(false)
    await props.onSaved()
    if (changed.length) props.toast('התיקונים נשמרו, וישמשו ללימוד')
  }

  const text = lines.map((l) => l.text).join('\n')
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      props.toast('לא הצלחתי להעתיק')
    }
  }
  const share = async () => {
    if (navigator.share) {
      try {
        await navigator.share({ title: scan.title, text })
      } catch {
        /* dismissed */
      }
    } else void copy()
  }

  /** Phones get the share sheet with the picture (it offers "Save Image"); elsewhere it downloads. */
  const saveImage = async () => {
    const name = `${scan.title.replace(/[\\/:*?"<>|]+/g, ' ').trim() || 'mishkefet'}.jpg`
    const file = new File([scan.image], name, { type: scan.image.type || 'image/jpeg' })
    if (navigator.canShare?.({ files: [file] })) {
      try {
        await navigator.share({ files: [file] })
      } catch {
        /* dismissed */
      }
      return
    }
    const a = document.createElement('a')
    a.href = URL.createObjectURL(file)
    a.download = name
    a.click()
    window.setTimeout(() => URL.revokeObjectURL(a.href), 4000)
  }

  // image frame: contain the scan into frameW x IMG_H
  const s = Math.min(frameW / scan.width, IMG_H / scan.height)
  const ox = (frameW - scan.width * s) / 2
  const oy = (IMG_H - scan.height * s) / 2

  return (
    <>
      <div className="bar" style={{ height: 52, padding: '8px 8px 0 10px' }}>
        <button className="icon-btn" onClick={editing ? () => setEditing(false) : props.onBack} aria-label="חזרה">
          <BackIcon />
        </button>
        <div style={{ display: 'flex', gap: 4 }}>
          {editing ? (
            <>
              <button className="icon-btn" onClick={props.onDeleted} aria-label="מחיקה"><TrashIcon /></button>
              <button className="icon-btn on" onClick={() => void saveEdit()} aria-label="שמירה"><CheckIcon /></button>
            </>
          ) : (
            <>
              <button className={`icon-btn ${showImage ? 'on' : ''}`} onClick={() => setShowImage((v) => !v)} aria-label="תמונה"><GalleryIcon size={22} /></button>
              <button className="icon-btn" onClick={startEdit} aria-label="עריכה"><PencilIcon /></button>
            </>
          )}
        </div>
      </div>

      <div className="res-head">
        <div className="display res-title">
          {editing ? <input value={title} onChange={(e) => setTitle(e.target.value)} aria-label="כותרת" /> : scan.title}
        </div>
        <div className="meta">
          <span>{dateLabel(scan.createdAt)}</span>
          <span className="chip">{linesLabel(lines.length)}</span>
          <span className="chip">{scan.kind === 'page' ? 'דף' : 'שורה'}</span>
          {isTuned && <span className="chip sage">מותאם לכתב היד שלך</span>}
        </div>
        {choice.tuned && !editing && (
          <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
            <ModelToggle useTuned={isTuned} onChange={(v) => void switchModel(v)} />
            {busy && <span className="rereading"><span className="spinner" />קוראת מחדש</span>}
          </div>
        )}
      </div>

      {showImage && (
        <div className="res-image" ref={frame}>
          <img src={url} alt="" style={{ left: ox, top: oy, width: scan.width * s, height: scan.height * s }} />
          {lines.map((l, i) =>
            l.box ? (
              <div key={i} className={`res-box ${i === active ? 'on' : ''}`} onClick={() => setActive(i)}
                style={{ left: ox + l.box.x * s, top: oy + l.box.y * s, width: Math.max(6, l.box.w * s), height: Math.max(5, l.box.h * s) }} />
            ) : null,
          )}
        </div>
      )}

      <div className="scroll">
        <div className="lines">
          {lines.map((l, i) => (
            <div key={i} className={`line ${i === active ? 'on' : ''}`} onClick={() => setActive(i)} role="button" tabIndex={0}>
              <span className="n">שורה {i + 1}</span>
              {editing ? (
                <textarea dir="rtl" rows={Math.max(1, Math.ceil((draft[i]?.length ?? 0) / 32))} value={draft[i] ?? ''}
                  onChange={(e) => setDraft((d) => d.map((x, k) => (k === i ? e.target.value : x)))} />
              ) : (
                <span className="t">{l.text}</span>
              )}
            </div>
          ))}
          {!lines.length && <div className="empty">לא מצאתי כאן כתב יד. נסו לצלם שוב, קרוב יותר ובאור טוב.</div>}
        </div>
      </div>

      {!editing && (
        <div className="actions">
          <button className={`cta ${copied ? 'sage' : ''}`} onClick={() => void copy()}>
            <CopyIcon />
            {copied ? 'הועתק' : 'העתקת הטקסט'}
          </button>
          <button className="round-btn" onClick={() => void saveImage()} aria-label="שמירת התמונה"><DownloadIcon /></button>
          <button className="round-btn" onClick={() => void share()} aria-label="שיתוף"><ShareIcon /></button>
        </div>
      )}
    </>
  )
}
