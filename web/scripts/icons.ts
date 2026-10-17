/**
 * Renders the home-screen icons (orange binoculars on #e2e8ce) into public/icons
 * with Playwright's Chromium. Run once after changing the artwork: node scripts/icons.ts
 */
import { chromium } from '@playwright/test'
import { resolve } from 'node:path'

const LOGO = `<circle cx="9" cy="14" r="7"/><circle cx="27" cy="14" r="7"/><path d="M16 11h4"/>`
const svg = (scale: number) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100%" height="100%"><rect width="100" height="100" fill="#e2e8ce"/><g transform="translate(50 52) scale(${scale}) translate(-18 -14)" fill="none" stroke="#ff7f11" stroke-width="3.2" stroke-linecap="round">${LOGO}</g></svg>`

const out = resolve(import.meta.dirname, '../public/icons')
const jobs: Array<[string, number, number]> = [
  ['icon-192.png', 192, 2.3],
  ['icon-512.png', 512, 2.3],
  ['apple-touch-icon.png', 180, 2.2],
  ['maskable-512.png', 512, 1.7],
]
const browser = await chromium.launch()
const page = await browser.newPage()
for (const [name, size, scale] of jobs) {
  await page.setViewportSize({ width: size, height: size })
  await page.setContent(`<html><body style="margin:0">${svg(scale)}</body></html>`)
  await page.screenshot({ path: resolve(out, name), clip: { x: 0, y: 0, width: size, height: size } })
}
await browser.close()
console.log('icons written to', out)
