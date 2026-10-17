import { useState } from 'react'
import type { LineResult } from '../engine/api.ts'
import type { ModelId, Scan } from '../lib/db.ts'
import { pairLines, wordDiff } from '../lib/text.ts'
import { BackIcon } from './icons.tsx'

export interface CompareState {
  model: ModelId
  beforeModel: ModelId
  before: LineResult[]
  /** null while the re-read is running. */
  after: LineResult[] | null
  mode: 'before' | 'after'
}

export function Compare(props: { state: CompareState; scan: Scan; onBack: () => void; onKeep: (keepNew: boolean) => void }) {
  const { state } = props
  const [mode, setMode] = useState<'before' | 'after'>('after')
  const after = state.after
  let changed = 0
  let total = 0
  const rows = after
    ? pairLines(state.before, after).map(([b, a]) => {
        const tokens = wordDiff(b, a, mode)
        const afterTokens = mode === 'after' ? tokens : wordDiff(b, a, 'after')
        changed += afterTokens.filter((t) => t.changed).length
        total += afterTokens.length
        return tokens
      })
    : []
  const round = state.model.startsWith('r') ? state.model.slice(1) : ''

  return (
    <>
      <div className="bar" style={{ height: 52, padding: '8px 8px 0 12px' }}>
        <button className="icon-btn" onClick={props.onBack} aria-label="חזרה"><BackIcon /></button>
        <div className="seg">
          <button className={mode === 'before' ? 'on' : ''} onClick={() => setMode('before')}>לפני</button>
          <button className={mode === 'after' ? 'on' : ''} onClick={() => setMode('after')}>אחרי</button>
        </div>
      </div>
      <div style={{ padding: '2px 22px 14px', flex: 'none' }}>
        <div className="display res-title">{props.scan.title}</div>
        <div className="meta">
          {after ? (
            <span className="chip" style={{ background: mode === 'after' ? 'var(--sage)' : 'rgba(255,27,28,.16)', color: '#262626', fontWeight: 500 }}>
              {changed ? `${changed} מתוך ${total} מילים השתנו` : 'אין שינויים'}
            </span>
          ) : (
            <span className="rereading" style={{ padding: 0 }}><span className="spinner" />קוראת מחדש עם המודל המותאם</span>
          )}
          <span>{mode === 'after' ? `מודל מותאם · סבב ${round}` : state.beforeModel === 'base' ? 'משקפת v1, לפני ההתאמה' : `מודל מותאם · סבב ${state.beforeModel.slice(1)}`}</span>
        </div>
      </div>
      <div className="scroll">
        <div className="lines">
          {rows.map((tokens, i) => (
            <div key={i} className="line" style={{ cursor: 'default' }}>
              <span className="n">שורה {i + 1}</span>
              <span className="words">
                {tokens.map((t, k) => (
                  <span key={k} className={t.changed ? (mode === 'after' ? 'diff-a' : 'diff-b') : ''}>{t.text}</span>
                ))}
              </span>
            </div>
          ))}
        </div>
      </div>
      <div style={{ padding: '10px 16px 30px', display: 'flex', flexDirection: 'column', gap: 4, flex: 'none' }}>
        <button className="cta" disabled={!after} style={{ height: 58, borderRadius: 29, opacity: after ? 1 : 0.5 }} onClick={() => props.onKeep(true)}>לשמור את הקריאה החדשה</button>
        <button className="link-btn" style={{ height: 40, fontSize: 14 }} onClick={() => props.onKeep(false)}>להשאיר את הקודמת</button>
      </div>
    </>
  )
}
