import type { LineBox } from '../engine/api.ts'
import type { ModelState } from '../lib/engine.ts'

export interface ProcState {
  image: string
  width: number
  height: number
  boxes: LineBox[]
  done: number
  total: number
  lastText: string
  picture: number
  pictures: number
  kind: 'line' | 'page'
}

const FRAME_W = 300
const FRAME_H = 210

/** The binoculars sweep across the line being read; read lines turn sage. */
export function Processing({ state, model, onCancel }: { state: ProcState; model: ModelState; onCancel: () => void }) {
  const s = Math.min(FRAME_W / state.width, FRAME_H / state.height)
  const ox = (FRAME_W - state.width * s) / 2
  const oy = (FRAME_H - state.height * s) / 2
  const pct = state.total ? Math.round((state.done / state.total) * 100) : 0
  const cur = Math.min(state.done, state.boxes.length - 1)
  const box = state.boxes[cur]
  const lens = box
    ? { x: ox + (box.x + box.w / 2) * s - 50, y: oy + (box.y + box.h / 2) * s - 26, on: state.done < state.total }
    : { x: FRAME_W / 2 - 50, y: FRAME_H / 2 - 26, on: true }
  const loading = model.phase === 'loading' && model.total > 0
  const sub = loading
    ? `מורידה את המודל · ${Math.round(model.loaded / 1e6)} / ${Math.round(model.total / 1e6)} MB`
    : !state.total
      ? 'מחפשת שורות'
      : state.done >= state.total
        ? 'סיימתי'
        : `שורה ${state.done + 1} מתוך ${state.total}`

  return (
    <div className="proc">
      <div className="proc-page">
        <img src={state.image} alt="" />
        {state.boxes.map((b, i) => (
          <div key={i} className="proc-box" style={{
            left: ox + b.x * s, top: oy + b.y * s, width: Math.max(4, b.w * s), height: Math.max(3, b.h * s),
            background: i < state.done ? 'rgba(172,191,164,.55)' : i === state.done ? 'rgba(255,127,17,.32)' : 'rgba(38,38,38,.10)',
          }} />
        ))}
        <svg className="lens" width="100" height="52" viewBox="0 0 100 52" fill="none" stroke="#ff7f11" strokeWidth={3} strokeLinecap="round"
          style={{ transform: `translate(${lens.x}px, ${lens.y}px)`, opacity: lens.on ? 1 : 0 }}>
          <circle cx="26" cy="26" r="22" />
          <circle cx="74" cy="26" r="22" />
          <path d="M45 18h10" />
        </svg>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
        <div className="proc-pct ltr">{pct}%</div>
        <div style={{ fontSize: 21, fontWeight: 500 }}>קוראת את כתב היד</div>
        <div className="sub" style={{ fontSize: 15 }}>
          {sub}
          {state.pictures > 1 && ` · תמונה ${state.picture} מתוך ${state.pictures}`}
        </div>
      </div>
      <div className="proc-line" key={state.lastText}>{state.lastText}</div>
      <div style={{ position: 'absolute', bottom: 30, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
        <div className="badge">על המכשיר · שום דבר לא עולה לשרת</div>
        <button className="link-btn" onClick={onCancel}>ביטול</button>
      </div>
    </div>
  )
}
