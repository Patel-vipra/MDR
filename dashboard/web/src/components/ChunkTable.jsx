import { DOMAIN_HEX, gradeClass } from '../lib/constants'

/** Retrieved chunks with their CRAG grade. Grade reason is the title attribute —
 *  it's the most interesting field and worth surfacing on hover. */
export default function ChunkTable({ chunks, grades }) {
  if (!chunks?.length) return <div className="blank">—</div>
  const byKey = {}
  for (const g of grades || []) byKey[`${g.doc_id}:${g.chunk_ix}`] = g

  return (
    <table className="w-full border-collapse text-[13px]">
      <thead>
        <tr>
          {['chunk', 'text', 'grade'].map((h) => (
            <th key={h} className="font-mono text-[10.5px] uppercase tracking-wider text-ink-3 font-medium text-left px-2.5 py-1.5 border-b border-line">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {chunks.map((c) => {
          const g = byKey[`${c.doc_id}:${c.chunk_ix}`]
          const grade = g?.grade ?? '—'
          return (
            <tr key={`${c.doc_id}:${c.chunk_ix}`} className="hover:bg-panel-2">
              <td className="px-2.5 py-1.5 border-b border-line font-mono text-[11.5px] whitespace-nowrap">
                <span style={{ color: DOMAIN_HEX[c.domain] }}>●</span> {c.doc_id}:{c.chunk_ix}
              </td>
              <td className="px-2.5 py-1.5 border-b border-line text-ink-3 text-xs truncate max-w-[1px]">
                {(c.chunk_text || '').replace(/\s+/g, ' ').slice(0, 90)}
              </td>
              <td className="px-2.5 py-1.5 border-b border-line">
                <span className={`tag ${gradeClass(grade)}`} title={g?.reason || ''}>{grade}</span>
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}
