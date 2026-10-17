import { useEffect, useRef, useState } from 'react'
import type { Stroke } from '../engine/api.ts'
import { db } from '../lib/db.ts'
import type { EngineClient } from '../lib/engine.ts'
import { refit, ROUND_SIZE, type RefitResult, type RefitStage } from '../lib/training.ts'
import { CheckIcon } from './icons.tsx'

const RING = 578

/** A sample's strokes as a tiny picture for the grid. */
function StrokeTile({ strokes }: { strokes: Stroke[] }) {
  const pts = strokes.flat()
  if (!pts.length) return null
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const x0 = Math.min(...xs)
  const y0 = Math.min(...ys)
  const w = Math.max(1, Math.max(...xs) - x0)
  const h = Math.max(1, Math.max(...ys) - y0)
  const pad = Math.max(w, h) * 0.08
  return (
    <svg viewBox={`${x0 - pad} ${y0 - pad} ${w + 2 * pad} ${h + 2 * pad}`} width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
      {strokes.map((s, i) => (
        <polyline key={i} points={s.map((p) => p.join(',')).join(' ')} fill="none" stroke="#e2e8ce" strokeWidth={Math.max(w, h) / 22} strokeLinecap="round" strokeLinejoin="round" />
      ))}
    </svg>
  )
}

export function Refit({ engine, onDone, onCompare }: { engine: EngineClient; onDone: () => void; onCompare: () => void }) {
  const [stage, setStage] = useState(0)
  const [pct, setPct] = useState(0)
  const [tiles, setTiles] = useState<Stroke[][]>([])
  const [held, setHeld] = useState(0)
  const [round, setRound] = useState(1)
  const [result, setResult] = useState<RefitResult | null>(null)
  const [failed, setFailed] = useState(false)
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true
    void (async () => {
      const [samples, st] = await Promise.all([db.samples(), db.settings()])
      setRound(st.rounds + 1)
      setTiles(samples.filter((s) => s.strokes && s.round === st.rounds + 1).map((s) => s.strokes!).slice(-ROUND_SIZE))
      setHeld(samples.filter((s) => s.heldOut && s.strokes).length)
      // let the grid be seen before it starts dissolving into the ring
      await new Promise((r) => setTimeout(r, 1600))
      try {
        const r = await refit(engine, (s: RefitStage, p: number) => {
          const idx = s === 'collect' ? 0 : s === 'tune' ? 1 : s === 'test' ? 2 : 3
          setStage((cur) => Math.max(cur, idx === 0 && p > 0.02 ? 1 : idx))
          setPct(s === 'collect' ? Math.round(p * 40) : s === 'tune' ? 40 + Math.round(p * 60) : 100)
        })
        setStage(2)
        await new Promise((r2) => setTimeout(r2, 900))
        setResult(r)
        setStage(3)
      } catch {
        setFailed(true)
        setStage(3)
      }
    })()
  }, [engine])

  const steps: Array<[string, string]> = [
    [`אוספת את ${tiles.length || ROUND_SIZE} הדוגמאות שלך`, 'כל מילה שכתבת, יחד עם מה שהיה צריך להיות כתוב בה.'],
    ['מכווננת את המודל', 'רק השכבה האחרונה זזה, כך שמה שמשקפת כבר יודעת לא נשכח.'],
    ['בודקת על מילים שלא ראתה', held ? `${held} מהמילים שלך נשמרו בצד, כדי למדוד אם באמת השתפרה.` : 'מילים שנשמרו בצד, כדי למדוד אם באמת השתפרה.'],
  ]
  const done = stage === 3
  const ringOn = stage === 1 || stage === 2
  const pctTo = (x: number) => `${Math.round(x * 100)}%`

  return (
    <div className="refit">
      <div style={{ textAlign: 'center', fontSize: 14, color: 'rgba(226,232,206,.6)', flex: 'none' }}>התאמה אישית · סבב {round}</div>
      <div className="refit-stage">
        <div className="tiles" style={{ opacity: stage === 0 ? 1 : 0, transform: `scale(${stage === 0 ? 1 : 0.35})` }}>
          {Array.from({ length: ROUND_SIZE }, (_, i) => (
            <div key={i}>{tiles[i] ? <StrokeTile strokes={tiles[i]} /> : null}</div>
          ))}
        </div>
        <div className="ring" style={{ opacity: ringOn ? 1 : 0, transform: `scale(${ringOn ? 1 : 0.7})` }}>
          <div className="glow" />
          <svg width="200" height="200" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="92" fill="none" stroke="rgba(226,232,206,.12)" strokeWidth="6" />
            <circle cx="100" cy="100" r="92" fill="none" stroke="#ff7f11" strokeWidth="6" strokeLinecap="round" strokeDasharray={RING}
              strokeDashoffset={(RING * (1 - pct / 100)).toFixed(1)} style={{ transition: 'stroke-dashoffset .12s linear' }} />
          </svg>
          <div className="pct ltr">{stage === 2 ? <CheckIcon size={56} sw={2.6} /> : `${pct}%`}</div>
        </div>
        <div className="scores" style={{ opacity: done ? 1 : 0, transform: `translateY(${done ? 0 : 16}px)` }}>
          {failed ? (
            <div style={{ fontSize: 17, textAlign: 'center', padding: '0 30px', lineHeight: 1.5 }}>משהו השתבש בהתאמה. הדוגמאות שמורות, ונסה שוב בסבב הבא.</div>
          ) : result ? (
            <>
              <div style={{ fontSize: 15, color: 'rgba(226,232,206,.6)' }}>דיוק על כתב היד שלך</div>
              <div className="pair ltr">
                <div><span style={{ fontSize: 56, color: 'rgba(226,232,206,.45)' }}>{pctTo(result.before)}</span><span className="lbl">לפני</span></div>
                <div><span style={{ fontSize: 104, color: result.accepted ? '#ff7f11' : 'rgba(226,232,206,.45)', fontWeight: 700 }}>{pctTo(result.after)}</span><span className="lbl">אחרי</span></div>
              </div>
              {!result.accepted && (
                <div style={{ fontSize: 15, color: 'rgba(226,232,206,.75)', textAlign: 'center', padding: '6px 30px 0', lineHeight: 1.5 }}>
                  הפעם לא השתפרה, אז נשארת הגרסה הקודמת. עוד סבב בדרך כלל עוזר.
                </div>
              )}
            </>
          ) : null}
        </div>
      </div>
      <div className="refit-steps">
        {steps.map(([h, d], i) => (
          <div key={i} style={{ opacity: stage >= i ? 1 : 0.35 }}>
            <div className={`dot ${stage > i ? 'done' : stage === i ? 'active' : ''}`}>{stage > i && <CheckIcon size={12} sw={3.6} />}</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}><span className="h">{h}</span><span className="d">{d}</span></div>
          </div>
        ))}
      </div>
      <div className="refit-cta" style={{ opacity: done ? 1 : 0, pointerEvents: done ? 'auto' : 'none' }}>
        {result?.accepted && <button className="cta" onClick={onCompare}>לנסות שוב על סריקה קודמת</button>}
        <button className="link-btn" onClick={onDone}>להמשיך לכתוב</button>
      </div>
    </div>
  )
}
