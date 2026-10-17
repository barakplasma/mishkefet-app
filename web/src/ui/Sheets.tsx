import { useMemo, useState } from 'react'
import type { Scan, Settings } from '../lib/db.ts'
import type { ModelState } from '../lib/engine.ts'
import { dateLabel, linesLabel } from '../lib/text.ts'
import { ChevronIcon } from './icons.tsx'

/** General model or the user's fine-tuned one. Shown only once a fine-tuned head exists. */
export function ModelToggle({ useTuned, onChange, dark }: { useTuned: boolean; onChange: (v: boolean) => void; dark?: boolean }) {
  return (
    <div className="seg" style={dark ? { background: 'rgba(226,232,206,.12)' } : undefined} role="radiogroup" aria-label="מודל">
      {([[false, 'כללי'], [true, 'מותאם אישית']] as const).map(([v, label]) => (
        <button key={label} role="radio" aria-checked={useTuned === v} className={useTuned === v ? 'on' : ''} onClick={() => onChange(v)}
          style={dark ? { color: useTuned === v ? '#262626' : '#e2e8ce', background: useTuned === v ? '#e2e8ce' : 'transparent' } : undefined}>
          {label}
        </button>
      ))}
    </div>
  )
}

export function ImportSheet(props: { hasTuned: boolean; useTuned: boolean; onUseTuned: (v: boolean) => void; onCamera: () => void; onFiles: () => void }) {
  return (
    <div className="sheet" role="dialog" aria-label="סריקה חדשה">
      <div className="grip" />
      <h2>סריקה חדשה</h2>
      {props.hasTuned && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
          <span className="sub" style={{ fontSize: 15 }}>לקרוא עם</span>
          <ModelToggle useTuned={props.useTuned} onChange={props.onUseTuned} />
        </div>
      )}
      <button className="sheet-row" onClick={props.onCamera}>צילום<span className="sub">שורה או דף</span></button>
      <button className="sheet-row" onClick={props.onFiles}>בחירה מהגלריה<span className="sub">כמה תמונות בבת אחת</span></button>
      <button className="sheet-row" onClick={props.onFiles}>קובץ<span className="sub ltr" style={{ fontFamily: 'ui-monospace, Menlo, monospace' }}>.jpg .png .heic .pdf</span></button>
    </div>
  )
}

export function ComparePickSheet({ scans, onPick }: { scans: Scan[]; onPick: (s: Scan) => void }) {
  // least confident readings first: the ones most likely to change (the ordering is the only place confidence is used)
  const ordered = useMemo(() => {
    const mean = (s: Scan) => {
      const ls = s.readings[s.current].lines
      return ls.length ? ls.reduce((a, l) => a + l.confidence, 0) / ls.length : 1
    }
    return [...scans].sort((a, b) => mean(a) - mean(b)).slice(0, 8)
  }, [scans])
  const thumbs = useMemo(() => new Map(ordered.map((s) => [s.id, URL.createObjectURL(s.thumb)])), [ordered])
  return (
    <div className="sheet" role="dialog" aria-label="על איזו סריקה">
      <div className="grip" />
      <h2>על איזו סריקה?</h2>
      <div className="sub" style={{ fontSize: 15, marginTop: -6 }}>נקרא אותה שוב עם המודל המותאם ונראה מה השתנה.</div>
      {ordered.map((s) => (
        <button key={s.id} className="sheet-row pick" onClick={() => onPick(s)}>
          <img src={thumbs.get(s.id)} alt="" style={{ width: 34, height: 42, borderRadius: 7, objectFit: 'cover', flex: 'none' }} />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
            <span style={{ fontSize: 16, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.title}</span>
            <span className="sub" style={{ fontSize: 13 }}>{dateLabel(s.createdAt)} · {linesLabel(s.readings[s.current].lines.length)}</span>
          </div>
          <ChevronIcon />
        </button>
      ))}
      {!ordered.length && <div className="sub" style={{ fontSize: 15 }}>עוד אין סריקות. צלמו דף, ואז אפשר להשוות.</div>}
    </div>
  )
}

export function AboutSheet(props: { model: ModelState; settings: Settings | null; hasTuned: boolean; onForget: () => void }) {
  const [confirm, setConfirm] = useState(false)
  const base = import.meta.env.BASE_URL
  return (
    <div className="sheet" role="dialog" aria-label="אודות">
      <div className="grip" />
      <h2>משקפת</h2>
      <div style={{ fontSize: 15, lineHeight: 1.6 }} className="sub">
        קריאת כתב יד עברי שרצה כולה בדפדפן, במכשיר שלכם. התמונות, הקריאות והדוגמאות נשמרות רק כאן, ושום דבר לא עולה לשרת.
      </div>
      <div style={{ fontSize: 14, lineHeight: 1.6 }} className="sub">
        המודל: <a href="https://huggingface.co/itayinbar/Mishkefet-v1" target="_blank" rel="noreferrer">Mishkefet-v1</a> (CC BY-NC-SA 4.0)
        {props.model.backend && <> · <span className="ltr">{props.model.backend === 'webgpu' ? 'WebGPU' : 'WASM'}</span></>}
        {props.settings && props.settings.rounds > 0 && <> · {props.settings.rounds} סבבי התאמה</>}
      </div>
      {props.hasTuned || (props.settings?.rounds ?? 0) > 0 ? (
        confirm ? (
          <button className="sheet-row danger" onClick={props.onForget}>למחוק את הדוגמאות ואת ההתאמה<span className="sub">אי אפשר לבטל</span></button>
        ) : (
          <button className="sheet-row" onClick={() => setConfirm(true)}>לשכוח את כתב היד שלי<span className="sub">הסריקות נשארות</span></button>
        )
      ) : null}
      <a className="sheet-row" href={`${base}privacy.html`} style={{ textDecoration: 'none' }}>פרטיות<ChevronIcon /></a>
      <a className="sheet-row" href="https://github.com/itayinbarr/mishkefet-app" target="_blank" rel="noreferrer" style={{ textDecoration: 'none' }}>קוד פתוח<ChevronIcon /></a>
    </div>
  )
}
