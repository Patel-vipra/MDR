import { useEffect, useState } from 'react'
import { getHealth } from '../lib/api'

const KPI = ({ n, l }) => <div className="kpi"><b className="kpiN">{n}</b><span className="kpiL">{l}</span></div>
const Row = ({ k, v }) => (
  <div className="flex justify-between text-[13px] py-1"><span className="text-ink-3">{k}</span><span className="font-mono text-xs">{String(v)}</span></div>
)
const statusTag = (s) => (s === 'up' || s === 'healthy' || s === 'remote' ? 't-ok' : s === 'degraded' ? 't-wa' : 't-ba')

export default function Health() {
  const [h, setH] = useState(null)
  useEffect(() => {
    const load = () => getHealth().then(setH).catch(() => {})
    load()
    const id = setInterval(load, 10000)   // §9.2: /api/health is polled by the dashboard
    return () => clearInterval(id)
  }, [])
  if (!h) return <div className="pane"><div className="blank">Loading health…</div></div>
  const m = h.memory

  return (
    <div className="grid grid-rows-[auto_1fr] overflow-hidden h-full">
      <div className="flex bg-panel border-b border-line overflow-x-auto">
        <KPI n={h.status} l="overall" />
        <KPI n={`${m.system_used_gb} GB`} l="system memory used" />
        <KPI n={`${m.system_total_gb} GB`} l="available" />
        <KPI n={h.qdrant.points ?? '—'} l="vectors" />
        <KPI n={`${Math.floor(h.uptime_seconds / 60)}m`} l="uptime" />
      </div>

      <div className="pane">
        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(250px,1fr))' }}>
          <div className="card"><div className="ch"><span className="ct">Language model</span><span className="flex-1" />
            <span className={`tag ${statusTag(h.llm.status)}`}>{h.llm.status}</span></div>
            <div className="p-3.5"><Row k="provider" v={h.llm.provider} /><Row k="model" v={h.llm.model} />
              {h.llm.error && <Row k="error" v={h.llm.error.slice(0, 40)} />}</div></div>

          <div className="card"><div className="ch"><span className="ct">Vector store</span><span className="flex-1" />
            <span className={`tag ${statusTag(h.qdrant.status)}`}>{h.qdrant.status}</span></div>
            <div className="p-3.5"><Row k="collection" v={h.qdrant.collection} /><Row k="points" v={h.qdrant.points ?? '—'} />
              <Row k="url" v={h.qdrant.url} /></div></div>

          <div className="card"><div className="ch"><span className="ct">Retrieval models</span><span className="flex-1" />
            <span className="tag t-ok">resident</span></div>
            <div className="p-3.5"><Row k="embeddings" v={h.models.embeddings} /><Row k="reranker" v={h.models.reranker} />
              <Row k="sparse" v={h.models.sparse} /></div></div>

          <div className="card"><div className="ch"><span className="ct">Indexes</span><span className="flex-1" />
            <span className={`tag ${h.indexes.graph_pickle ? 't-ok' : 't-ba'}`}>{h.indexes.graph_pickle ? 'loaded' : 'missing'}</span></div>
            <div className="p-3.5"><Row k="graph pickle" v={h.indexes.graph_pickle} /><Row k="bm25 pickle" v={h.indexes.bm25_pickle} />
              <Row k="graph size" v={h.indexes.graph_mb ? `${h.indexes.graph_mb} MB` : '—'} /></div></div>

          <div className="card"><div className="ch"><span className="ct">Pipeline</span><span className="flex-1" />
            <span className={`tag ${h.pipeline_loaded ? 't-ok' : 't-ba'}`}>{h.pipeline_loaded ? 'loaded' : 'down'}</span></div>
            <div className="p-3.5">{h.warmed && Object.entries(h.warmed).map(([k, v]) => <Row key={k} k={k} v={v} />)}</div></div>

          <div className="card"><div className="ch"><span className="ct">Memory</span></div>
            <div className="p-3.5">
              <Row k="api process rss" v={`${m.process_rss_mb} MB`} />
              <Row k="system used" v={`${m.system_used_gb} / ${m.system_total_gb} GB`} />
              <div className="h-1.5 bg-panel-3 rounded-full overflow-hidden mt-2">
                <div className="h-full rounded-full" style={{ width: `${m.percent}%`, background: m.percent > 85 ? '#B91C1C' : '#047857' }} />
              </div>
            </div></div>
        </div>
      </div>
    </div>
  )
}
