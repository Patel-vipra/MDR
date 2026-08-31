import { useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { getGraph, getGraphTruth } from '../lib/api'
import { TYPE_HEX } from '../lib/constants'

const KPI = ({ n, l }) => <div className="kpi"><b className="kpiN">{n}</b><span className="kpiL">{l}</span></div>

export default function Graph() {
  const [data, setData] = useState(null)
  const [truth, setTruth] = useState(null)
  const [typeFilter, setTypeFilter] = useState(null)
  const [hover, setHover] = useState(null)
  const [showTruth, setShowTruth] = useState(false)
  const box = useRef(null)
  const [size, setSize] = useState({ w: 800, h: 520 })

  useEffect(() => { getGraph().then(setData).catch(() => {}) }, [])
  useEffect(() => { if (showTruth && !truth) getGraphTruth().then(setTruth).catch(() => {}) }, [showTruth, truth])

  useEffect(() => {
    if (!box.current) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(box.current)
    return () => ro.disconnect()
  }, [data])

  const active = showTruth && truth ? truth : data

  // react-force-graph mutates the objects it's given (adding x/y/vx/vy), so hand
  // it a fresh shallow copy — reusing the cached API response across a filter
  // change otherwise carries stale positions and throws on unknown node refs.
  const gd = useMemo(() => {
    if (!active) return { nodes: [], links: [] }
    let nodes = active.nodes
    if (typeFilter) nodes = nodes.filter((n) => n.type === typeFilter)
    const ids = new Set(nodes.map((n) => n.id))
    return {
      nodes: nodes.map((n) => ({ ...n })),
      links: active.edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({ ...e })),
    }
  }, [active, typeFilter])

  if (!data) return <div className="pane"><div className="blank">Loading graph…</div></div>
  const types = Object.entries(data.node_types).sort((a, b) => b[1] - a[1])

  return (
    <div className="grid grid-rows-[auto_1fr] overflow-hidden h-full">
      <div className="flex bg-panel border-b border-line overflow-x-auto">
        <KPI n={data.counts.nodes} l="nodes extracted" />
        <KPI n={data.counts.edges} l="edges extracted" />
        <KPI n="91.3%" l="node recall" />
        <KPI n="71.5%" l="node precision" />
        <KPI n="41.0%" l="edge precision" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_240px] overflow-hidden">
        <div ref={box} className="relative bg-panel overflow-hidden">
          {size.w > 0 && (
            <ForceGraph2D
              graphData={gd} width={size.w} height={size.h}
              backgroundColor="#FFFFFF" cooldownTicks={120}
              nodeRelSize={5} nodeVal={1}
              nodeColor={(n) => TYPE_HEX[n.type] || '#64748B'}
              linkColor={() => 'rgba(201,207,218,.9)'} linkWidth={1}
              onNodeHover={setHover}
              nodeCanvasObjectMode={() => 'after'}
              nodeCanvasObject={(n, ctx, scale) => {
                if (!hover || hover.id !== n.id) return
                const label = n.name || n.id
                ctx.font = `500 ${12 / scale}px "Geist Mono", monospace`
                const w = ctx.measureText(label).width + 8 / scale
                ctx.fillStyle = 'rgba(255,255,255,.95)'
                ctx.fillRect(n.x - w / 2, n.y - 20 / scale, w, 14 / scale)
                ctx.strokeStyle = '#DFE3EA'; ctx.lineWidth = 1 / scale
                ctx.strokeRect(n.x - w / 2, n.y - 20 / scale, w, 14 / scale)
                ctx.fillStyle = '#0E1420'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
                ctx.fillText(label, n.x, n.y - 13 / scale)
              }}
            />
          )}
        </div>

        <div className="border-l border-line bg-panel overflow-y-auto p-3">
          <div className="ct mb-2">View</div>
          <button onClick={() => setShowTruth((s) => !s)} className={`fb w-full mb-3 ${showTruth ? 'fb-on' : ''}`}>
            {showTruth ? 'ground truth' : 'extracted'}
          </button>

          <div className="ct mb-2">Node types</div>
          {types.map(([t, n]) => (
            <button key={t} onClick={() => setTypeFilter((f) => (f === t ? null : t))}
              className={`flex items-center gap-2 text-[12.5px] py-1 w-full text-left transition-opacity
                ${typeFilter && typeFilter !== t ? 'opacity-35' : 'opacity-100'} ${typeFilter === t ? 'text-ink font-medium' : 'text-ink-2'} hover:text-ink`}>
              <span className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ background: TYPE_HEX[t] || '#64748B' }} />
              {t}<span className="ml-auto font-mono text-[11px] text-ink-3">{n}</span>
            </button>
          ))}

          <div className="ct mt-4 mb-1.5">Selected</div>
          <div className="meta">{hover ? `${hover.name || hover.id} · ${hover.type}` : 'hover a node'}</div>

          <div className="rounded-md p-3 text-[13px] leading-snug bg-warn-bg border border-warn/20 mt-4">
            <b className="block mb-0.5 text-[12.5px]">41% edge precision</b>
            Most extracted relationships are noise. This is the ceiling on multi-hop performance.
          </div>
        </div>
      </div>
    </div>
  )
}
