import { NODE_HEX, NODE_ORDER, REFUSAL_PATH } from '../lib/constants'

/**
 * Horizontal span waterfall — the pattern LangSmith/Langfuse/Phoenix and general
 * APM tools use, chosen over a vertical list because bar length encodes duration.
 * On this pipeline that instantly shows generator + crag_grader dominating, which
 * a plain list hides entirely.
 *
 * `timings` fills in progressively as SSE node events arrive.
 */
export default function Waterfall({ timings, refused, selected, onSelect, running }) {
  const seq = refused ? REFUSAL_PATH : NODE_ORDER
  const ran = seq.filter((n) => timings[n] != null)
  const total = ran.reduce((s, n) => s + timings[n], 0) || 1

  let acc = 0
  const offset = {}
  for (const n of seq) {
    if (timings[n] != null) { offset[n] = acc; acc += timings[n] }
  }

  return (
    <div className="flex flex-col gap-0.5">
      {seq.map((n) => {
        const has = timings[n] != null
        const d = has ? timings[n] : 0
        return (
          <button
            key={n}
            onClick={() => has && onSelect?.(n === selected ? null : n)}
            className={`grid grid-cols-[128px_1fr_44px] gap-2 items-center py-1 px-1 rounded text-left
              transition-opacity ${has ? 'opacity-100' : running ? 'opacity-30' : 'opacity-40'}
              ${selected === n ? 'bg-acc-soft' : 'hover:bg-panel-2'}`}
          >
            <span className="font-mono text-[11.5px] text-ink-2 truncate flex items-center gap-1.5">
              <i className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: NODE_HEX[n] }} />
              {n}
            </span>
            <span className="h-[17px] bg-panel-3 rounded-[3px] relative overflow-hidden">
              {has && (
                <span
                  className="absolute inset-y-0 rounded-[3px] transition-[width,left] duration-500"
                  style={{ left: `${(offset[n] / total) * 100}%`,
                           width: `${Math.max(1.2, (d / total) * 100)}%`,
                           background: NODE_HEX[n] }}
                />
              )}
            </span>
            <span className="font-mono text-[11px] text-ink-2 text-right tabular-nums">
              {has ? d.toFixed(2) : ''}
            </span>
          </button>
        )
      })}
    </div>
  )
}
