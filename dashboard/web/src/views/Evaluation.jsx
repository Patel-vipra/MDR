import { useEffect, useState } from 'react'
import { getEval } from '../lib/api'

const ADDED = { A0: 'dense only', A1: '+ contextual', A2: '+ hybrid · rerank', A3: '+ CRAG', A4: '+ ABAC', A5: '+ GraphRAG' }
const KPI = ({ n, l }) => <div className="kpi"><b className="kpiN">{n}</b><span className="kpiL">{l}</span></div>
const f3 = (v) => (v == null ? '—' : Number(v).toFixed(3))

function Bars({ rows, max, fmt, color }) {
  return (
    <div className="flex flex-col gap-1.5">
      {rows.map(([c, v]) => (
        <div key={c} className="grid grid-cols-[26px_1fr_44px] gap-2 items-center">
          <span className="font-mono text-[11px] text-ink-3">{c}</span>
          <span className="h-[17px] bg-panel-3 rounded-[3px] overflow-hidden block">
            <span className="block h-full rounded-[3px] transition-[width] duration-700"
              style={{ width: `${(v / max) * 100}%`, background: color(v) }} />
          </span>
          <span className="font-mono text-[11px] text-right tabular-nums text-ink-2">{fmt(v)}</span>
        </div>
      ))}
    </div>
  )
}

export default function Evaluation() {
  const [data, setData] = useState(null)
  useEffect(() => { getEval().then(setData).catch(() => {}) }, [])
  if (!data) return <div className="pane"><div className="blank">Loading evaluation…</div></div>

  const rows = data.order.map((c) => ({ c, ...data.configs[c] })).filter((r) => r.n)
  const bestHit = Math.max(...rows.map((r) => r.hit_at_5 ?? 0))
  const bestMh = Math.max(...rows.map((r) => r.multi_hop_accuracy ?? 0))
  const maxP50 = Math.max(...rows.map((r) => r.latency?.p50 ?? 0))
  const totalRuns = rows.reduce((s, r) => s + r.n, 0)
  const errs = rows.reduce((s, r) => s + (r.errors || 0), 0)
  const a5 = data.configs.A5 || {}

  return (
    <div className="grid grid-rows-[auto_1fr] overflow-hidden h-full">
      <div className="flex bg-panel border-b border-line overflow-x-auto">
        <KPI n={totalRuns.toLocaleString()} l="runs" />
        <KPI n={errs} l="errors" />
        <KPI n={rows.length} l="configs" />
        <KPI n={a5.abac_leakage_rate === 0 ? '0' : a5.abac_leakage_rate ?? '—'} l="abac leakage" />
        <KPI n={a5.abac_denial_accuracy != null ? `${(a5.abac_denial_accuracy * 100).toFixed(0)}%` : '—'} l="denial accuracy" />
      </div>

      <div className="pane">
        <div className="card overflow-hidden mb-3.5">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr>{['config', 'added', 'faithful', 'ans. rel', 'ctx recall', 'hit@5', 'mrr', 'multi-hop', 'p50', 'p95'].map((h) => (
                <th key={h} className="font-mono text-[10.5px] uppercase tracking-wider text-ink-3 font-medium text-left px-3 py-2 bg-panel-2 border-b border-line whitespace-nowrap">{h}</th>
              ))}</tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.c} className="hover:bg-panel-2">
                  <td className="px-3 py-2 border-b border-line font-mono text-[11.5px] font-medium">{r.c}</td>
                  <td className="px-3 py-2 border-b border-line text-ink-3 whitespace-nowrap">{ADDED[r.c]}</td>
                  <td className="px-3 py-2 border-b border-line tabular-nums text-ink-2">{f3(r.faithfulness)}</td>
                  <td className="px-3 py-2 border-b border-line tabular-nums text-ink-2">{f3(r.answer_relevancy)}</td>
                  <td className="px-3 py-2 border-b border-line tabular-nums text-ink-2">{f3(r.context_recall)}</td>
                  <td className={`px-3 py-2 border-b border-line tabular-nums ${r.hit_at_5 === bestHit ? 'text-acc font-semibold' : 'text-ink-2'}`}>{f3(r.hit_at_5)}</td>
                  <td className="px-3 py-2 border-b border-line tabular-nums text-ink-2">{f3(r.mrr)}</td>
                  <td className={`px-3 py-2 border-b border-line tabular-nums ${r.multi_hop_accuracy === bestMh ? 'text-acc font-semibold' : 'text-ink-2'}`}>{f3(r.multi_hop_accuracy)}</td>
                  <td className="px-3 py-2 border-b border-line tabular-nums text-ink-2">{r.latency?.p50?.toFixed(2)}s</td>
                  <td className="px-3 py-2 border-b border-line tabular-nums text-ink-2">{r.latency?.p95?.toFixed(2)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="grid gap-3 lg:grid-cols-2">
          <div className="card"><div className="ch"><span className="ct">hit@5</span></div><div className="p-3.5">
            <Bars rows={rows.map((r) => [r.c, r.hit_at_5 ?? 0])} max={1} fmt={(v) => (v * 100).toFixed(1)}
              color={(v) => (v === bestHit ? '#4338CA' : '#0369A1')} /></div></div>
          <div className="card"><div className="ch"><span className="ct">p50 latency</span></div><div className="p-3.5">
            <Bars rows={rows.map((r) => [r.c, r.latency?.p50 ?? 0])} max={maxP50} fmt={(v) => `${v.toFixed(1)}s`}
              color={(v) => (v > 10 ? '#B91C1C' : '#047857')} /></div></div>
        </div>

        <div className="grid gap-3 mt-3.5" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(260px,1fr))' }}>
          <div className="rounded-md p-3 text-[13px] leading-snug bg-ok-bg border border-ok/20">
            <b className="block mb-0.5 text-[12.5px]">Hybrid retrieval wins</b>
            hit@5 84 → 96%, multi-hop nearly doubles, for half a second of latency.
          </div>
          <div className="rounded-md p-3 text-[13px] leading-snug bg-bad-bg border border-bad/20">
            <b className="block mb-0.5 text-[12.5px]">CRAG did not pay off</b>
            5× slower, faithfulness −8pts, multi-hop −11pts on this corpus.
          </div>
          <div className="rounded-md p-3 text-[13px] leading-snug bg-warn-bg border border-warn/20">
            <b className="block mb-0.5 text-[12.5px]">A prediction that failed</b>
            The plan predicted GraphRAG would rescue multi-hop. It didn't — reported, not buried.
          </div>
        </div>
      </div>
    </div>
  )
}
