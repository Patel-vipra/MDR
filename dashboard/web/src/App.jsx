import { useEffect, useState } from 'react'
import { getHealth } from './lib/api'
import Corpus from './views/Corpus'
import Evaluation from './views/Evaluation'
import Graph from './views/Graph'
import Health from './views/Health'
import Playground from './views/Playground'

const ICONS = {
  play: <><circle cx="12" cy="12" r="8.5" /><path d="M10.5 9l5 3-5 3z" /></>,
  corpus: <><rect x="3.5" y="4" width="7" height="16" rx="1" /><rect x="13.5" y="4" width="7" height="9" rx="1" /><path d="M13.5 17.5h7" /></>,
  graph: <><circle cx="6" cy="7" r="2.2" /><circle cx="18" cy="6" r="2.2" /><circle cx="12" cy="17.5" r="2.2" /><path d="M8 8.5l3 7M16.3 8l-3.3 7.5" /></>,
  evaluation: <path d="M4 19V10M9.3 19V5M14.6 19v-6M20 19v-9" />,
  health: <path d="M3 12.5h3.5l2-5.5 3 12 3-8.5 1.5 2H21" />,
}

const VIEWS = [
  { id: 'play', label: 'Playground', icon: 'play', C: Playground },
  { id: 'corpus', label: 'Corpus', icon: 'corpus', C: Corpus },
  { id: 'graph', label: 'Graph', icon: 'graph', C: Graph },
  { id: 'evaluation', label: 'Evaluation', icon: 'evaluation', C: Evaluation },
  { id: 'health', label: 'Health', icon: 'health', C: Health },
]

export default function App() {
  const [view, setView] = useState('play')
  const [status, setStatus] = useState(null)

  useEffect(() => {
    const load = () => getHealth().then((h) => setStatus(h.status)).catch(() => setStatus('down'))
    load()
    const id = setInterval(load, 15000)
    return () => clearInterval(id)
  }, [])

  const Active = VIEWS.find((v) => v.id === view).C
  const dot = status === 'healthy' ? 'bg-ok' : status === 'degraded' ? 'bg-warn' : 'bg-bad'
  const pill = status === 'healthy' ? 'text-ok bg-ok-bg border-ok/20'
    : status === 'degraded' ? 'text-warn bg-warn-bg border-warn/20' : 'text-bad bg-bad-bg border-bad/20'

  return (
    <div className="h-full grid grid-rows-[46px_1fr]">
      <header className="bg-panel border-b border-line flex items-center gap-3.5 px-4">
        <div className="w-[22px] h-[22px] rounded-md bg-acc grid place-items-center text-white font-bold text-[11px] shrink-0">◈</div>
        <div className="font-semibold text-sm tracking-tight">Orbital</div>
        <div className="text-ink-3 text-[13px]">/</div>
        <div className="text-[13px] text-ink-2">{VIEWS.find((v) => v.id === view).label}</div>
        <div className="flex-1" />
        <div className={`flex items-center gap-1.5 font-mono text-[11px] border px-2.5 py-1 rounded ${pill}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />{status ?? '…'}
        </div>
      </header>

      <div className="grid grid-cols-[186px_1fr] max-sm:grid-cols-1 overflow-hidden">
        <nav className="bg-panel border-r border-line p-2 flex flex-col overflow-y-auto max-sm:hidden">
          <div className="flex flex-col gap-px">
            {VIEWS.map((v) => (
              <button key={v.id} onClick={() => setView(v.id)}
                className={`flex items-center gap-2.5 px-2.5 py-1.5 rounded-md text-left text-[13.5px] w-full transition-colors
                  ${view === v.id ? 'bg-acc-soft text-acc font-semibold' : 'text-ink-2 hover:bg-panel-3 hover:text-ink'}`}>
                <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0" fill="none" stroke="currentColor" strokeWidth="1.7">{ICONS[v.icon]}</svg>
                {v.label}
              </button>
            ))}
          </div>
          <div className="mt-auto pt-2.5 border-t border-line px-2.5">
            {[['docs', '100'], ['chunks', '261'], ['graph', '410n'], ['queries', '80'], ['runs', '1,440']].map(([k, val]) => (
              <div key={k} className="flex justify-between text-[11.5px] py-0.5 text-ink-3">
                {k}<b className="font-mono font-medium text-ink-2">{val}</b>
              </div>
            ))}
          </div>
        </nav>

        {/* mobile tab bar */}
        <nav className="hidden max-sm:flex fixed bottom-0 inset-x-0 z-50 bg-panel border-t border-line p-1">
          {VIEWS.map((v) => (
            <button key={v.id} onClick={() => setView(v.id)}
              className={`flex-1 flex flex-col items-center gap-0.5 py-1.5 rounded text-[10.5px] ${view === v.id ? 'text-acc' : 'text-ink-3'}`}>
              <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.7">{ICONS[v.icon]}</svg>
              {v.label}
            </button>
          ))}
        </nav>

        <main className="overflow-hidden max-sm:pb-14"><Active /></main>
      </div>
    </div>
  )
}
