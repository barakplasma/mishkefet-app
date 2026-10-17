import { useEffect, useMemo, useState } from 'react'
import type { Scan } from '../lib/db.ts'
import { dateLabel } from '../lib/text.ts'
import { BackIcon, CheckIcon, ImportIcon } from './icons.tsx'

export function ScanList(props: { scans: Scan[]; onBack: () => void; onOpen: (id: string) => void; onDelete: (ids: string[]) => void; onImport: () => void }) {
  const [selecting, setSelecting] = useState(false)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const thumbs = useMemo(() => new Map(props.scans.map((s) => [s.id, URL.createObjectURL(s.thumb)])), [props.scans])
  useEffect(() => () => thumbs.forEach((u) => URL.revokeObjectURL(u)), [thumbs])

  const toggle = (id: string) =>
    setPicked((p) => {
      const n = new Set(p)
      if (n.has(id)) n.delete(id)
      else n.add(id)
      return n
    })

  return (
    <>
      <div className="bar" style={{ height: 52, marginTop: 8, padding: '0 8px 0 12px' }}>
        <button className="icon-btn" onClick={props.onBack} aria-label="חזרה"><BackIcon /></button>
        {props.scans.length > 0 && (
          selecting ? (
            <div style={{ display: 'flex', gap: 6 }}>
              <button className="pill small" disabled={!picked.size} style={{ color: picked.size ? 'var(--red)' : undefined }} onClick={() => {
                props.onDelete([...picked])
                setPicked(new Set())
                setSelecting(false)
              }}>מחיקה{picked.size ? ` (${picked.size})` : ''}</button>
              <button className="pill small" onClick={() => {
                setSelecting(false)
                setPicked(new Set())
              }}>ביטול</button>
            </div>
          ) : (
            <button className="pill small" onClick={() => setSelecting(true)}>בחירה</button>
          )
        )}
      </div>
      <div className="display list-title">סריקות</div>
      <div className="scroll">
        {props.scans.map((s) => {
          const r = s.readings[s.current]
          const preview = r.lines.find((l) => l.text.trim())?.text ?? ''
          return (
            <button key={s.id} className="row" onClick={() => (selecting ? toggle(s.id) : props.onOpen(s.id))}>
              <div>
                {selecting && <span className={`check ${picked.has(s.id) ? 'on' : ''}`}>{picked.has(s.id) && <CheckIcon size={14} sw={3} />}</span>}
                <img className="thumb" src={thumbs.get(s.id)} alt="" />
                <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline' }}>
                    <span className="title">{s.title}</span>
                    <span className="sub" style={{ fontSize: 13, flex: 'none' }}>{r.lines.length === 1 ? 'שורה' : `${r.lines.length} שורות`}</span>
                  </div>
                  <div className="preview">{preview}</div>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 12.5 }} className="sub">
                    <span>{dateLabel(s.createdAt)}</span>
                    {s.reread && <span className="chip sage">נקרא מחדש</span>}
                  </div>
                </div>
              </div>
            </button>
          )
        })}
        {!props.scans.length && (
          <div className="empty">
            עוד אין סריקות. צלמו דף או בחרו תמונה, והקריאה תישמר כאן, רק במכשיר הזה.
            <div style={{ marginTop: 16 }}><button className="pill" onClick={props.onImport}><ImportIcon />סריקה חדשה</button></div>
          </div>
        )}
      </div>
    </>
  )
}
