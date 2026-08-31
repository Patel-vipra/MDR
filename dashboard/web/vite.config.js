import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative /api base in the client + this proxy for local dev. In production
// (PLAN.md §10.2) Caddy serves dist/ and reverse-proxies /api, so no build-time
// host is ever baked in.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } } },
  build: { outDir: 'dist', sourcemap: false },
})
