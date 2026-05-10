import { useState } from 'react';
import { Header } from './components/Header';
import { GeneratePage } from './pages/GeneratePage';
import { ComparePage } from './pages/ComparePage';
import { LiveComparePage } from './pages/LiveComparePage';
import { EvalPage } from './pages/EvalPage';

type Tab = 'generate' | 'compare' | 'live' | 'eval';

const TABS: { id: Tab; label: string; desc: string }[] = [
  { id: 'generate', label: 'Generate',       desc: 'Live kernel generation & benchmark' },
  { id: 'compare',  label: 'Compare Models', desc: 'Curated side-by-side diff examples' },
  { id: 'live',     label: 'Live Compare',   desc: 'Both models run live — judge-proof demo' },
  { id: 'eval',     label: 'Eval Results',   desc: '50-sample held-out evaluation dashboard' },
];

export default function App() {
  const [tab, setTab] = useState<Tab>('generate');

  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      {/* Tab navigation */}
      <div className="border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-10">
        <div className="mx-auto flex max-w-7xl px-6">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`group relative px-5 py-3.5 text-sm font-medium transition border-b-2 ${
                tab === t.id
                  ? 'border-amd-red text-white'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:border-slate-600'
              }`}
            >
              {t.label}
              {/* Tooltip on hover */}
              <span className="pointer-events-none absolute left-1/2 top-full mt-1 -translate-x-1/2 whitespace-nowrap rounded bg-slate-800 px-2 py-1 text-[10px] text-slate-300 opacity-0 transition-opacity group-hover:opacity-100">
                {t.desc}
              </span>
            </button>
          ))}
        </div>
      </div>

      <main className="flex-1">
        {tab === 'generate' && <GeneratePage />}
        {tab === 'compare'  && <ComparePage />}
        {tab === 'live'     && <LiveComparePage />}
        {tab === 'eval'     && <EvalPage />}
      </main>

      <footer className="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
        ROCmForge · AMD Developer Hackathon · Fine-tuned Qwen2.5-Coder-7B on MI300X
        <span className="mx-3 text-slate-700">·</span>
        <a
          href="https://github.com/Ku-JingHao/rocmforge"
          target="_blank"
          rel="noreferrer"
          className="hover:text-slate-300 transition"
        >
          github.com/Ku-JingHao/rocmforge
        </a>
        <span className="mx-3 text-slate-700">·</span>
        <a
          href="https://huggingface.co/jinghao57/rocmforge-7b"
          target="_blank"
          rel="noreferrer"
          className="hover:text-slate-300 transition"
        >
          huggingface.co/jinghao57/rocmforge-7b
        </a>
      </footer>
    </div>
  );
}
