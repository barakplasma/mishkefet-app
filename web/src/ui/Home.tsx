import type { Settings } from '../lib/db.ts'
import type { ModelState } from '../lib/engine.ts'
import { ROUND_SIZE } from '../lib/training.ts'
import { GalleryIcon, ImportIcon, Logo, MenuIcon, PencilIcon } from './icons.tsx'

const MB = (n: number) => Math.round(n / 1e6)

export function Home(props: {
  model: ModelState
  settings: Settings | null
  roundCount: number
  hasScans: boolean
  onList: () => void
  onImport: () => void
  onCamera: () => void
  onGallery: () => void
  onTrain: () => void
  onAbout: () => void
}) {
  const { model, settings, roundCount } = props
  const round = (settings?.rounds ?? 0) + 1
  const sub = roundCount === 0 ? `סבב ${round} · ${ROUND_SIZE} מילים עד ההתאמה הבאה` : `${roundCount} מתוך ${ROUND_SIZE} מילים עד ההתאמה הבאה`
  return (
    <>
      <div className="bar home-top">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <button className="icon-btn" onClick={props.onList} aria-label="סריקות" style={{ marginTop: 8 }}>
            <MenuIcon />
          </button>
          <button className="brand" onClick={props.onAbout} style={{ border: 0, background: 'transparent', color: 'inherit', cursor: 'pointer', padding: 0 }} aria-label="אודות">
            <span style={{ marginTop: 9, display: 'flex' }}><Logo /></span>
            <span className="display">משקפת</span>
          </button>
        </div>
        <button className="icon-btn" onClick={props.onImport} aria-label="סריקה חדשה" style={{ marginTop: 8 }}>
          <ImportIcon />
        </button>
      </div>

      <div className="home-mid">
        <button className="orb" onClick={props.onCamera} aria-label="צילום">
          <div>
            <Logo width={124} height={80} sw={2.6} />
          </div>
        </button>
        <div className="home-hint">לחצו לצילום כתב יד</div>
        <button className="pill" style={{ marginTop: 18 }} onClick={props.onGallery}>
          <GalleryIcon />
          בחירה מהגלריה
        </button>
        {model.phase === 'loading' && model.total > 0 && (
          <div className="model-note">
            מורידה את המודל, פעם אחת בלבד · <span className="ltr">{MB(model.loaded)} / {MB(model.total)} MB</span>
          </div>
        )}
        {model.phase === 'error' && <div className="model-note">לא הצלחתי לטעון את המודל. בדקו את החיבור ונסו לרענן.</div>}
      </div>

      <div className="home-bottom">
        <button className="train-card" onClick={props.onTrain}>
          <div className="dot">
            <PencilIcon />
          </div>
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={{ fontSize: 16, fontWeight: 600 }}>ללמד את משקפת את כתב היד שלך</span>
            <span style={{ fontSize: 13 }} className="sub">{sub}</span>
            <div className="progress">
              <div style={{ width: `${Math.round((roundCount / ROUND_SIZE) * 100)}%` }} />
            </div>
          </div>
        </button>
      </div>
    </>
  )
}
