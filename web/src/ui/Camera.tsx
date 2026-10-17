import { useEffect, useRef, useState } from 'react'
import { pictureFromVideo, type Picture } from '../lib/images.ts'
import type { ModelChoice } from './App.tsx'
import { CloseIcon, GalleryIcon } from './icons.tsx'
import { ModelToggle } from './Sheets.tsx'

type Mode = 'line' | 'page'

function guideSize(mode: Mode, w: number, h: number) {
  return mode === 'page'
    ? { w: Math.min(290, w - 40), h: Math.min(400, h - 70) }
    : { w: Math.min(320, w - 30), h: 90 }
}

export function Camera(props: {
  choice: ModelChoice
  onUseTuned: (v: boolean) => void
  onClose: () => void
  onGallery: () => void
  onShot: (pic: Picture, kind: Mode) => void
}) {
  const [mode, setMode] = useState<Mode>('page')
  const [error, setError] = useState<string | null>(null)
  const [size, setSize] = useState({ w: 360, h: 520 })
  const video = useRef<HTMLVideoElement>(null)
  const finder = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let stream: MediaStream | null = null
    let alive = true
    const start = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('הדפדפן הזה לא נותן גישה למצלמה. אפשר לבחור תמונה מהגלריה.')
        return
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 3840 }, height: { ideal: 2160 } },
          audio: false,
        })
        if (!alive) return stream.getTracks().forEach((t) => t.stop())
        if (video.current) {
          video.current.srcObject = stream
          await video.current.play().catch(() => undefined)
        }
      } catch {
        setError('אין גישה למצלמה. אפשר לבחור תמונה מהגלריה.')
      }
    }
    void start()
    return () => {
      alive = false
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  useEffect(() => {
    const el = finder.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const guide = guideSize(mode, size.w, size.h)

  const shoot = async () => {
    const v = video.current
    if (!v || !v.videoWidth) return
    // map the guide rectangle (centred in the finder) into video pixels, the video being object-fit: cover
    const scale = Math.max(size.w / v.videoWidth, size.h / v.videoHeight)
    const ox = (size.w - v.videoWidth * scale) / 2
    const oy = (size.h - v.videoHeight * scale) / 2
    const gx = (size.w - guide.w) / 2
    const gy = (size.h - guide.h) / 2
    const pad = mode === 'line' ? 12 : 0
    const x = Math.max(0, (gx - pad - ox) / scale / v.videoWidth)
    const y = Math.max(0, (gy - pad - oy) / scale / v.videoHeight)
    const w = Math.min(1 - x, (guide.w + 2 * pad) / scale / v.videoWidth)
    const h = Math.min(1 - y, (guide.h + 2 * pad) / scale / v.videoHeight)
    props.onShot(await pictureFromVideo(v, { x, y, w, h }), mode)
  }

  return (
    <>
      <div className="bar" style={{ height: 56, padding: '0 10px' }}>
        <button className="icon-btn" onClick={props.onClose} aria-label="סגירה">
          <CloseIcon />
        </button>
        <div className="seg">
          <button className={mode === 'line' ? 'on' : ''} onClick={() => setMode('line')}>שורה</button>
          <button className={mode === 'page' ? 'on' : ''} onClick={() => setMode('page')}>דף שלם</button>
        </div>
        <div style={{ width: 46 }} />
      </div>
      <div className="viewfinder" ref={finder}>
        <video ref={video} playsInline muted />
        {error ? (
          <div className="cam-error">
            <div>{error}</div>
            <button className="pill" onClick={props.onGallery}><GalleryIcon />בחירה מהגלריה</button>
          </div>
        ) : (
          <>
            <div className="guide" style={{ width: guide.w, height: guide.h }}>
              <i /><i /><i /><i />
            </div>
            <div className="cam-hint">{mode === 'page' ? 'יישרו את הדף בתוך המסגרת' : 'שורה אחת בתוך המסגרת'}</div>
          </>
        )}
      </div>
      {props.choice.tuned && (
        <div style={{ display: 'flex', justifyContent: 'center', paddingTop: 10 }}>
          <ModelToggle useTuned={props.choice.useTuned} onChange={props.onUseTuned} dark />
        </div>
      )}
      <div className="bar" style={{ height: 130, padding: '0 34px 20px' }}>
        <div style={{ width: 52 }} />
        <button className="shutter" onClick={() => void shoot()} aria-label="צילום" disabled={!!error}>
          <div />
        </button>
        <button className="icon-btn" onClick={props.onGallery} aria-label="גלריה" style={{ width: 52, height: 52, borderRadius: 14, border: '1.5px solid rgba(226,232,206,.4)' }}>
          <GalleryIcon size={22} />
        </button>
      </div>
    </>
  )
}
