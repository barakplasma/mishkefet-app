/** Inline SVG icons from the design. The logo is a pair of binoculars (משקפת). */

const round = { strokeLinecap: 'round', strokeLinejoin: 'round' } as const

export function Logo({ width = 36, height = 24, sw = 3.6 }: { width?: number; height?: number; sw?: number }) {
  return (
    <svg width={width} height={height} viewBox="0 0 36 24" fill="none" stroke="#ff7f11" strokeWidth={sw} strokeLinecap="round" aria-hidden="true">
      <circle cx="9" cy="14" r="7" />
      <circle cx="27" cy="14" r="7" />
      <path d="M16 11h4" />
    </svg>
  )
}

export function MenuIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" aria-hidden="true">
      <path d="M4 7h16M4 12h16M4 17h10" />
    </svg>
  )
}

export function ImportIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} {...round} aria-hidden="true">
      <path d="M12 15V4M7.5 8.5L12 4l4.5 4.5M5 15v4h14v-4" />
    </svg>
  )
}

export function GalleryIcon({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} {...round} aria-hidden="true">
      <rect x="3.5" y="5" width="17" height="14" rx="3" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="M5 17l5-4.5 3.5 3 2.5-2 3.5 3" />
    </svg>
  )
}

export function BackIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} {...round} aria-hidden="true">
      <path d="M9 5l7 7-7 7" />
    </svg>
  )
}

export function ChevronIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} {...round} aria-hidden="true" style={{ opacity: 0.5 }}>
      <path d="M15 5l-7 7 7 7" />
    </svg>
  )
}

export function CloseIcon({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" aria-hidden="true">
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  )
}

export function PencilIcon({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} {...round} aria-hidden="true">
      <path d="M4.5 19.5h4l10-10-4-4-10 10z" />
    </svg>
  )
}

export function CheckIcon({ size = 22, sw = 2.4 }: { size?: number; sw?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={sw} {...round} aria-hidden="true">
      <path d="M5 12.5l4.5 4.5L19 7.5" />
    </svg>
  )
}

export function CopyIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.9} strokeLinejoin="round" aria-hidden="true">
      <rect x="8.5" y="8.5" width="11" height="11" rx="2.5" />
      <path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5" />
    </svg>
  )
}

export function ShareIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.9} {...round} aria-hidden="true">
      <path d="M12 14V4M7.5 8.5L12 4l4.5 4.5M5 13v6h14v-6" />
    </svg>
  )
}

export function TrashIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} {...round} aria-hidden="true">
      <path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" />
    </svg>
  )
}
