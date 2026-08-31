// Relative base: Vite proxies /api in dev, Caddy proxies it in prod (PLAN.md §10.2).
const BASE = '/api'

async function get(path) {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) throw new Error(`${path} → ${r.status}`)
  return r.json()
}

export const getHealth   = () => get('/health')
export const getCorpus   = () => get('/corpus')
export const getGraph    = (q = '') => get(`/graph${q}`)
export const getGraphTruth = () => get('/graph/truth')
export const getEval     = () => get('/eval/results')
export const getPersonas = () => get('/personas')
export const getTraces   = () => get('/traces')
export const getTrace    = (id) => get(`/traces/${id}`)

/**
 * Streams a live query. The pipeline is ~54s p50 on local Qwen3-4B, so this
 * reports every node as it completes rather than resolving once at the end —
 * onNode fires ~9 times, onDone once.
 *
 * Uses fetch + a ReadableStream reader rather than EventSource because
 * EventSource cannot issue a POST, and the query/persona belong in a body.
 */
export async function streamQuery({ query, persona }, { onStart, onNode, onDone, onError, signal }) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ query, persona }),
    signal,
  })
  if (!res.ok || !res.body) throw new Error(`query → ${res.status}`)

  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    const frames = buf.split('\n\n')
    buf = frames.pop() ?? ''
    for (const frame of frames) {
      let ev = null, payload = null
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) ev = line.slice(6).trim()
        else if (line.startsWith('data:')) {
          try { payload = JSON.parse(line.slice(5).trim()) } catch { /* partial frame */ }
        }
      }
      if (!ev || payload == null) continue
      if (ev === 'start') onStart?.(payload)
      else if (ev === 'node') onNode?.(payload)
      else if (ev === 'done') onDone?.(payload)
      else if (ev === 'error') onError?.(payload)
    }
  }
}
