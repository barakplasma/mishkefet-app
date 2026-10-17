import { defineConfig, devices } from '@playwright/test'

/**
 * E2E runs against the production build (`vite preview`) at /mishkefet-app/, exactly as
 * GitHub Pages serves it. Projects emulate an iPhone (WebKit) and an Android
 * phone (Chromium). The real-model test is opt-in: MK_E2E_MODEL=1.
 */
const MODEL = !!process.env.MK_E2E_MODEL

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  workers: 1,
  timeout: MODEL ? 15 * 60 * 1000 : 60 * 1000,
  reporter: process.env.CI ? 'line' : 'list',
  use: {
    baseURL: 'http://localhost:4173/mishkefet-app/',
    trace: 'retain-on-failure',
    serviceWorkers: 'block',
    locale: 'he-IL',
  },
  projects: [
    { name: 'iphone-webkit', use: { ...devices['iPhone 15'] } },
    {
      name: 'android-chromium',
      use: {
        ...devices['Pixel 7'],
        launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=metal'] },
      },
    },
  ],
  webServer: {
    command: 'npm run build && npx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173/mishkefet-app/',
    reuseExistingServer: !process.env.CI,
    timeout: 180 * 1000,
  },
})
