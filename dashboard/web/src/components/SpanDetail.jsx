/** Inspector for one selected span — the third pane of the standard trace-UI layout. */
export default function SpanDetail({ node, trace }) {
  if (!node || !trace) return null
  const t = trace
  const rows = [['node', node], ['duration', `${(t.timings?.[node] ?? 0).toFixed(3)}s`]]

  const grades = t.grades || []
  const push = (k, v) => rows.push([k, String(v)])
  if (node === 'abac_gate') { push('persona', t.persona); push('domains', (t.allowed_domains || []).join(', ') || '—') }
  if (node === 'domain_router') push('classified', t.domain || '—')
  if (node === 'query_analyzer') { push('multi-hop', !!t.multi_hop); push('retries', t.retry_count ?? 0) }
  if (node === 'hybrid_retrieval') { push('dense', t.n_dense_hits ?? 0); push('sparse', t.n_sparse_hits ?? 0); push('fused', t.n_fused ?? 0) }
  if (node === 'reranker') push('kept', t.n_reranked ?? (t.reranked || []).length)
  if (node === 'crag_grader') {
    push('correct', grades.filter((g) => g.grade === 'correct').length)
    push('ambiguous', grades.filter((g) => g.grade === 'ambiguous').length)
    push('incorrect', grades.filter((g) => g.grade === 'incorrect').length)
  }
  if (node === 'graphrag_augmenter') {
    push('trigger', t.trigger_reason || '—')
    push('edges', (t.graph_context || '').split('\n').filter(Boolean).length)
  }
  if (node === 'generator') push('citations', (t.citations || []).length)
  if (node === 'hallucination_check') { push('confidence', t.confidence); push('abstained', !!t.abstained) }
  if (node === 'refusal') push('reason', t.refusal_reason || '—')

  const edges = node === 'graphrag_augmenter' && t.graph_context
    ? t.graph_context.split('\n').filter(Boolean).slice(0, 5) : []

  return (
    <>
      <div className="bg-panel-2 border border-line rounded-md p-3 mt-3">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3 text-[12.5px] py-0.5">
            <span className="text-ink-3">{k}</span>
            <span className="font-mono text-[11.5px] text-right">{v}</span>
          </div>
        ))}
      </div>
      {edges.length > 0 && (
        <div className="mt-3 px-3 py-2.5 bg-acc-soft border border-acc-line rounded-md">
          {edges.map((l, i) => {
            const m = l.match(/^(\S+)\s+([A-Z_]+)\s+(.+?)\s*\[/)
            return (
              <div key={i} className="font-mono text-[11px] text-ink-2 py-0.5">
                {m ? <>{m[1]} <em className="not-italic text-acc font-medium">{m[2]}</em> {m[3]}</> : l.slice(0, 54)}
              </div>
            )
          })}
        </div>
      )}
    </>
  )
}
