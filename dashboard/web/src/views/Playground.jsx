import { useEffect, useMemo, useRef, useState } from 'react'
import { getPersonas, getTrace, getTraces, streamQuery } from '../lib/api'
import { DOMAIN_HEX } from '../lib/constants'
import ChunkTable from '../components/ChunkTable'
import SpanDetail from '../components/SpanDetail'
import Waterfall from '../components/Waterfall'

const PRESET_IDS = ['Q-001', 'Q-002', 'Q-051', 'Q-066', 'Q-076']

export default function Playground() {
  const [personas, setPersonas] = useState([])
  const [presets, setPresets] = useState([])
  const [persona, setPersona] = useState('admin')
  const [text, setText] = useState('')
  const [timings, setTimings] = useState({})
  const [trace, setTrace] = useState(null)
  const [selected, setSelected] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const abort = useRef(null)

  useEffect(() => {
    getPersonas().then((d) => setPersonas(d.personas)).catch(() => {})
    getTraces().then((all) => setPresets(all.filter((t) => PRESET_IDS.includes(t.query_id)))).catch(() => {})
  }, [])

  const reset = () => { setTimings({}); setTrace(null); setSelected(null); setError(null) }

  async function run(q = text, p = persona) {
    if (!q.trim() || running) return
    reset(); setRunning(true)
    abort.current = new AbortController()
    try {
      await streamQuery({ query: q, persona: p }, {
        signal: abort.current.signal,
        onNode: (d) => {
          if (d.elapsed != null) setTimings((t) => ({ ...t, [d.node]: d.elapsed }))
          setTrace((prev) => ({ ...(prev || {}), ...d, persona: p, query: q, timings: undefined }))
        },
        onDone: (d) => setTrace({ ...d, persona: p, query: q, timings: d.node_timings || {} }),
        onError: (d) => setError(d.error),
      })
    } catch (e) {
      if (e.name !== 'AbortError') setError(String(e))
    } finally { setRunning(false) }
  }

  /** Load a committed trace instantly — the demo path shouldn't always cost ~54s. */
  async function replay(id) {
    reset()
    const t = await getTrace(id)
    setText(t.query); setPersona(t.persona)
    setTrace({ ...t, timings: t.node_timings || {} })
    setTimings(t.node_timings || {})
  }

  const merged = useMemo(() => (trace ? { ...trace, timings: trace.timings ?? timings } : null), [trace, timings])
  const conf = merged?.confidence ?? null
  const confColor = conf == null ? 'var(--x)' : conf >= 0.7 ? '#047857' : conf > 0 ? '#B45309' : '#B91C1C'

  return (
    <div className="grid grid-rows-[auto_auto_1fr] overflow-hidden h-full">
      <div className="px-4 py-3 bg-panel border-b border-line">
        <div className="flex gap-1.5 items-center flex-wrap mb-2.5">
          <span className="meta mr-1">asking as</span>
          {personas.map((p) => (
            <button key={p.id} onClick={() => setPersona(p.id)}
              className={`border rounded-full px-3 py-1 text-[12.5px] flex items-center gap-1.5 transition-colors
                ${persona === p.id ? 'bg-ink text-white border-ink' : 'bg-panel text-ink-2 border-line hover:border-line-2 hover:text-ink'}`}>
              <span className="w-1.5 h-1.5 rounded-full"
                style={{ background: p.allowed_domains.length > 1 ? '#8A93A3' : DOMAIN_HEX[p.allowed_domains[0]] }} />
              {p.id}
            </button>
          ))}
        </div>
        <div className="flex gap-2">
          <input value={text} onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && run()}
            placeholder="Ask the corpus…"
            className="flex-1 bg-panel border border-line rounded-md px-3 py-2 text-sm min-w-0 focus:outline-none focus:border-acc focus:ring-[3px] focus:ring-acc-soft" />
          <button onClick={() => run()} disabled={running || !text.trim()}
            className="bg-ink text-white rounded-md px-5 py-2 font-semibold text-[13.5px] disabled:opacity-40 hover:bg-[#26303F] transition-colors">
            {running ? 'Running…' : 'Run'}
          </button>
        </div>
      </div>

      <div className="flex gap-1.5 px-4 py-2.5 bg-panel-2 border-b border-line overflow-x-auto">
        <span className="meta self-center mr-1 shrink-0">replay</span>
        {presets.map((p) => (
          <button key={p.query_id} onClick={() => replay(p.query_id)}
            className="bg-panel border border-line rounded px-2.5 py-1 text-xs text-ink-2 whitespace-nowrap flex gap-1.5 items-center hover:border-line-2 hover:text-ink">
            <b className="font-mono text-[10.5px] text-acc font-medium">{p.query_id}</b>
            {p.type.replace('_', '-')}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_452px] overflow-hidden">
        <div className="pane">
          <div className="card">
            <div className="ch"><span className="ct">Answer</span><span className="flex-1" />
              {merged && (merged.refused ? <span className="tag t-ba">denied</span>
                : merged.abstained ? <span className="tag t-wa">abstained</span>
                : merged.answer ? <span className="tag t-ok">answered</span> : null)}
            </div>
            <div className="p-3.5">
              {error && <div className="tag t-ba">{error}</div>}
              {!merged?.answer && !error && <div className="blank">{running ? 'Running the pipeline…' : 'Ask a question, or replay a committed trace'}</div>}
              {merged?.answer && (
                <>
                  <div className={`text-base leading-relaxed ${merged.refused ? 'border-l-2 border-bad pl-3' : merged.abstained ? 'border-l-2 border-warn pl-3' : ''}`}>
                    {String(merged.answer).split(/(\[[A-Z]{2,5}-\d+:\d+\])/g).map((part, i) =>
                      /^\[[A-Z]{2,5}-\d+:\d+\]$/.test(part)
                        ? <span key={i} className="inline-block font-mono text-[11.5px] text-acc bg-acc-soft border border-acc-line px-1 rounded mx-0.5">{part.slice(1, -1)}</span>
                        : <span key={i}>{part}</span>)}
                  </div>
                  {conf != null && (
                    <div className="flex items-center gap-2.5 mt-3.5 pt-3 border-t border-line">
                      <span className="meta">groundedness</span>
                      <div className="flex-1 h-[5px] bg-panel-3 rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-[width] duration-700" style={{ width: `${conf * 100}%`, background: confColor }} />
                      </div>
                      <span className="font-mono text-xs tabular-nums" style={{ color: confColor }}>{(conf * 100).toFixed(0)}%</span>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          <div className="card mt-3.5">
            <div className="ch"><span className="ct">Retrieved</span><span className="flex-1" />
              <span className="meta">{merged?.reranked?.length ? `${merged.reranked.length} of ${merged.n_fused ?? (merged.fused || []).length}` : ''}</span>
            </div>
            <ChunkTable chunks={merged?.reranked} grades={merged?.grades} />
          </div>
        </div>

        <div className="pane border-l border-line bg-panel-2">
          <div className="card">
            <div className="ch"><span className="ct">Trace</span><span className="flex-1" />
              <span className="meta">{merged?.latency_seconds ? `${Number(merged.latency_seconds).toFixed(1)}s` : running ? 'running…' : ''}</span>
            </div>
            <div className="p-3.5">
              <Waterfall timings={merged?.timings ?? timings} refused={merged?.refused}
                selected={selected} onSelect={setSelected} running={running} />
              <SpanDetail node={selected} trace={merged} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
