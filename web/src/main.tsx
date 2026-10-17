import '@fontsource/karantina/400.css'
import '@fontsource/karantina/700.css'
import '@fontsource/amatic-sc/700.css'
import '@fontsource-variable/rubik/wght.css'
import './theme.css'
import './app.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './ui/App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Offline shell: caches the app and the ONNX Runtime files after first use.
// The model itself lives in its own Cache (see src/engine/store.ts).
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, { scope: import.meta.env.BASE_URL }).catch(() => undefined)
  })
}
