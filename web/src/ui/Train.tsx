import { useCallback, useEffect, useRef, useState } from 'react'
import trainWords from '../../../spec/train-words.json'
import type { Stroke } from '../engine/api.ts'
import { db, type Scan, type Settings } from '../lib/db.ts'
import type { EngineClient } from '../lib/engine.ts'
import { INACTIVITY_MS, recordAttempt, ROUND_SIZE } from '../lib/training.ts'
import { makePicker, type WordPicker } from '../lib/words.ts'
import type { ModelChoice } from './App.tsx'
import { CheckIcon, CloseIcon } from './icons.tsx'

export function TrainIntro({ onClose, onStart }: { onClose: () => void; onStart: () => void }) {
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', paddingBottom: 30 }}>
      <div className="bar" style={{ height: 56, padding: '0 10px' }}>
        <button className="icon-btn" onClick={onClose} aria-label="סגירה"><CloseIcon /></button>
      </div>
      <div className="scroll" style={{ padding: '8px 26px 0', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div className="display intro-title">ללמד את משקפת לקרוא אותך</div>
        <div className="sub" style={{ fontSize: 16, lineHeight: 1.55 }}>כל כתב יד שונה. כמה דקות של כתיבה על המסך, והמודל מתכוונן לאיך שאתם כותבים.</div>
        <div className="steps-list">
          <div><span className="num">1</span><div><div className="h">כתבו את המילה שמופיעה</div><div className="d">באצבע או בעט, כמו שאתם כותבים על נייר.</div></div></div>
          <div><span className="num">2</span><div><div className="h">עצרו, והיא ממשיכה לבד</div><div className="d">ירוק אם קראה נכון, אדום אם לא. גם הטעויות נשמרות ללימוד.</div></div></div>
          <div><span className="num">3</span><div><div className="h">כל {ROUND_SIZE} מילים, התאמה</div><div className="d">המודל מתאמן מחדש על הדוגמאות שלכם, ואפשר לנסות שוב סריקה ישנה.</div></div></div>
        </div>
      </div>
      <div style={{ padding: '0 16px', display: 'flex', flexDirection: 'column', gap: 12, flex: 'none' }}>
        <div className="badge" style={{ alignSelf: 'center', background: 'transparent' }}>הכל על המכשיר. הדוגמאות לא עולות לשום מקום.</div>
        <button className="cta" onClick={onStart}>להתחיל לכתוב</button>
      </div>
    </div>
  )
}

type Phase = 'idle' | 'drawing' | 'waiting' | 'reading' | 'ok' | 'bad' | 'out'

const INK = '#262626'
const GREEN = '#2f9a4f'
const RED = '#ff1b1c'

/** Words of the user's scans from lines the model was least sure of (never shown as such). */
function unsureScanWords(scans: Scan[]): string[] {
  const lines = scans.flatMap((s) => s.readings[s.current].lines)
  if (!lines.length) return []
  const sorted = [...lines].sort((a, b) => a.confidence - b.confidence)
  return sorted.slice(0, Math.ceil(sorted.length / 3)).flatMap((l) => l.text.split(/\s+/))
}

export function Train(props: {
  engine: EngineClient
  settings: Settings
  choice: ModelChoice
  scans: Scan[]
  onClose: () => void
  onCount: (n: number) => void
  onRoundComplete: () => void
}) {
  const { engine } = props
  const round = props.settings.rounds + 1
  const [count, setCount] = useState(0)
  const [word, setWord] = useState('')
  const [phase, setPhase] = useState<Phase>('idle')
  const [readAs, setReadAs] = useState('')
  const [lastOk, setLastOk] = useState(false)
  const [empty, setEmpty] = useState(true)
  const picker = useRef<WordPicker | null>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const strokes = useRef<Stroke[]>([])
  const current = useRef<Stroke | null>(null)
  const waitTimer = useRef(0)
  const timers = useRef<number[]>([])
  const phaseRef = useRef(phase)
  phaseRef.current = phase

  useEffect(() => {
    let alive = true
    void db.samples().then((samples) => {
      if (!alive) return
      const attempts = samples.filter((s) => s.strokes).map((s) => ({ word: s.word, readAs: s.readAs }))
      picker.current = makePicker(trainWords as string[], attempts, unsureScanWords(props.scans))
      const n = samples.filter((s) => s.round === round && s.strokes).length
      setCount(n)
      setWord(picker.current.next())
    })
    return () => {
      alive = false
      timers.current.forEach(clearTimeout)
      clearTimeout(waitTimer.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const later = (fn: () => void, ms: number) => timers.current.push(window.setTimeout(fn, ms))

  // canvas sizing and drawing
  const redraw = useCallback((color: string) => {
    const c = canvas.current
    if (!c) return
    const g = c.getContext('2d')!
    const d = window.devicePixelRatio || 1
    g.setTransform(d, 0, 0, d, 0, 0)
    g.clearRect(0, 0, c.width, c.height)
    g.strokeStyle = color
    g.lineWidth = 4.5
    g.lineCap = 'round'
    g.lineJoin = 'round'
    for (const s of strokes.current) {
      g.beginPath()
      s.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)))
      if (s.length === 1) g.lineTo(s[0][0] + 0.1, s[0][1])
      g.stroke()
    }
  }, [])

  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const fit = () => {
      const d = window.devicePixelRatio || 1
      c.width = Math.max(1, Math.round(c.clientWidth * d))
      c.height = Math.max(1, Math.round(c.clientHeight * d))
      redraw(INK)
    }
    const ro = new ResizeObserver(fit)
    ro.observe(c)
    return () => ro.disconnect()
  }, [redraw])

  const point = (e: React.PointerEvent): [number, number] => {
    const r = canvas.current!.getBoundingClientRect()
    return [Math.round((e.clientX - r.left) * 10) / 10, Math.round((e.clientY - r.top) * 10) / 10]
  }

  const evaluate = async () => {
    const target = word
    setPhase('reading')
    const head = props.choice.tuned?.head
    let read = ''
    try {
      read = await engine.readStrokes(strokes.current, target, head)
    } catch {
      read = ''
    }
    const ok = read.trim() === target
    setReadAs(read.trim())
    setLastOk(ok)
    redraw(ok ? GREEN : RED)
    setPhase(ok ? 'ok' : 'bad')
    if (navigator.vibrate) navigator.vibrate(ok ? 12 : [10, 40, 10])
    await recordAttempt(round, target, strokes.current, read.trim())
    later(() => setPhase('out'), 1150)
    later(() => advance(), 1400)
  }

  const advance = () => {
    const n = count + 1
    strokes.current = []
    redraw(INK)
    setEmpty(true)
    setCount(n)
    props.onCount(n)
    if (n >= ROUND_SIZE) {
      props.onRoundComplete()
      return
    }
    setWord(picker.current!.next())
    setPhase('idle')
  }

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!['idle', 'drawing', 'waiting'].includes(phaseRef.current)) return
    clearTimeout(waitTimer.current)
    e.currentTarget.setPointerCapture?.(e.pointerId)
    current.current = [point(e)]
    strokes.current.push(current.current)
    redraw(INK)
    setEmpty(false)
    setPhase('drawing')
  }
  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const s = current.current
    if (!s) return
    const events = (e.nativeEvent as PointerEvent).getCoalescedEvents?.() ?? [e.nativeEvent]
    const g = canvas.current!.getContext('2d')!
    g.strokeStyle = INK
    g.lineWidth = 4.5
    g.lineCap = 'round'
    for (const ev of events) {
      const r = canvas.current!.getBoundingClientRect()
      const p: [number, number] = [Math.round((ev.clientX - r.left) * 10) / 10, Math.round((ev.clientY - r.top) * 10) / 10]
      const last = s[s.length - 1]
      s.push(p)
      g.beginPath()
      g.moveTo(last[0], last[1])
      g.lineTo(p[0], p[1])
      g.stroke()
    }
  }
  const onUp = () => {
    if (!current.current) return
    current.current = null
    setPhase('waiting')
    waitTimer.current = window.setTimeout(() => void evaluate(), INACTIVITY_MS)
  }

  const clear = () => {
    clearTimeout(waitTimer.current)
    strokes.current = []
    redraw(INK)
    setEmpty(true)
    setPhase('idle')
  }
  const skip = () => {
    clear()
    setWord(picker.current?.next() ?? word)
  }

  const ok = phase === 'ok'
  const bad = phase === 'bad'
  const showOk = ok || (phase === 'out' && lastOk)
  const out = phase === 'out'
  const fbText = ok
    ? `נכון · קראתי „${readAs}"`
    : bad
      ? readAs ? `קראתי „${readAs}" · נשמר ללימוד` : 'לא הצלחתי לקרוא · נשמר ללימוד'
      : phase === 'waiting' || phase === 'reading'
        ? 'ממתינה שתסיימו…'
        : phase === 'drawing'
          ? ''
          : 'עצרו כשתסיימו, ונמשיך לבד'
  const fbColor = ok ? GREEN : bad ? '#d81516' : 'var(--sub)'
  const card = { opacity: out ? 0 : 1, transform: `translateY(${out ? -10 : 0}px)` }

  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', paddingBottom: 30 }}>
      <div className="bar" style={{ height: 56, padding: '0 10px' }}>
        <button className="icon-btn" onClick={props.onClose} aria-label="סגירה"><CloseIcon /></button>
        <div className="ltr" style={{ fontSize: 15, fontWeight: 500, fontVariantNumeric: 'tabular-nums' }}>{count} / {ROUND_SIZE}</div>
        <button className="link-btn" style={{ height: 40, padding: '0 14px', fontSize: 14 }} onClick={skip}>דילוג</button>
      </div>
      <div className="ticks">
        {Array.from({ length: ROUND_SIZE }, (_, i) => <div key={i} className={i < count ? 'done' : ''} />)}
      </div>
      <div className="prompt" style={card}>
        <div className="sub" style={{ fontSize: 14 }}>כתבו</div>
        <div className="word">{word}</div>
      </div>
      <div className="pad" style={{ ...card, borderColor: ok ? GREEN : bad ? RED : 'transparent' }}>
        <div className="baseline" />
        {empty && <div className="placeholder">כתבו כאן</div>}
        <canvas ref={canvas} onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp} />
        <div className="verdict" style={{ background: showOk ? GREEN : RED, transform: `scale(${ok || bad ? 1 : 0.4})`, opacity: ok || bad ? 1 : 0 }}>
          {showOk ? <CheckIcon size={28} sw={2.8} /> : <CloseIcon size={24} />}
        </div>
        <div className="wait" style={{ width: phase === 'waiting' ? '100%' : '0%', transition: phase === 'waiting' ? `width ${INACTIVITY_MS}ms linear` : 'none' }} />
      </div>
      <div className="train-foot">
        <div style={{ fontSize: 15, fontWeight: 500, color: fbColor }}>{fbText}</div>
        <button className="pill small" onClick={clear}>ניקוי</button>
      </div>
    </div>
  )
}
