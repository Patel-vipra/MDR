import { useEffect, useMemo, useState } from 'react'
import { getCorpus } from '../lib/api'
import { DOMAIN_HEX } from '../lib/constants'

const KPI = ({ n, l }) => <div className="kpi"><b className="kpiN">{n}</b><span className="kpiL">{l}</span></div>

export default function Corpus() {
  const [data, setData] = useState(null)
  const [f, setF] = useState({ domain: null, variant: null, sensitivity: null })

  useEffect(() => { getCorpus().then(setData).catch(() => {}) }, [])

  const shown = useMemo(() => (data?.documents || []).filter((d) =>
    (!f.domain || d.domain === f.domain) &&
    (!f.variant || d.variant === f.variant) &&
    (!f.sensitivity || d.sensitivity === f.sensitivity)), [data, f])

  if (!data) return <div className="pane"><div className="blank">Loading corpus…</div></div>
  const t = data.totals
  const toggle = (k, v) => setF((s) => ({ ...s, [k]: s[k] === v ? null : v }))
  const Btn = ({ k, v, children }) => (
    <button onClick={() => toggle(k, v)} className={`fb ${f[k] === v ? 'fb-on' : ''}`}>{children}</button>
  )

  return (
    <div className="grid grid-rows-[auto_auto_1fr] overflow-hidden h-full">
      <div className="flex bg-panel border-b border-line overflow-x-auto">
        <KPI n={t.documents} l="documents" />
        <KPI n={t.chunks} l="chunks" />
        <KPI n={t.degraded} l="ocr recovered" />
        <KPI n={data.ocr.cer != null ? `${(data.ocr.cer * 100).toFixed(2)}%` : '—'} l="aggregate cer" />
        <KPI n={data.ocr.entity_survival != null ? `${(data.ocr.entity_survival * 100).toFixed(0)}%` : '—'} l="entity survival" />
      </div>

      <div className="flex gap-1.5 items-center px-4 py-2.5 bg-panel-2 border-b border-line flex-wrap">
        {['finance', 'hr', 'legal', 'engineering'].map((d) => <Btn key={d} k="domain" v={d}>{d} {t.by_domain[d]}</Btn>)}
        <span className="w-px h-4 bg-line mx-1" />
        <Btn k="variant" v="clean">clean {t.clean}</Btn>
        <Btn k="variant" v="degraded">degraded {t.degraded}</Btn>
        <span className="w-px h-4 bg-line mx-1" />
        {['internal', 'confidential', 'restricted'].map((s) => <Btn key={s} k="sensitivity" v={s}>{s} {t.by_sensitivity[s]}</Btn>)}
        <span className="meta ml-auto">{shown.length} shown</span>
      </div>

      <div className="pane">
        <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(168px,1fr))' }}>
          {shown.map((d) => (
            <div key={d.doc_id} className="bg-panel border border-line rounded-md px-2.5 py-2 hover:border-line-2"
              style={{ borderLeftWidth: 3, borderLeftColor: DOMAIN_HEX[d.domain] }}>
              <div className="font-mono text-xs font-medium" style={{ color: DOMAIN_HEX[d.domain] }}>{d.doc_id}</div>
              <div className="text-[10.5px] text-ink-3 mt-0.5 truncate">{d.doc_type?.replace(/_/g, ' ')}</div>
              <div className="flex gap-1 mt-1.5 flex-wrap">
                <span className="tag t-nu">{d.chunks}c</span>
                {d.variant === 'degraded' && <span className="tag t-wa">cer {d.cer != null ? (d.cer * 100).toFixed(1) : '—'}</span>}
                {d.sensitivity === 'restricted' && <span className="tag t-ba">restricted</span>}
                {d.sensitivity === 'confidential' && <span className="tag t-ac">confidential</span>}
              </div>
            </div>
          ))}
          {!shown.length && <div className="blank">No documents match those filters</div>}
        </div>
      </div>
    </div>
  )
}
