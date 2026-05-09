import { EXAMPLES } from '../lib/examples';
import type { ExampleEntry } from '../types';

interface Props {
  onPick: (example: ExampleEntry) => void;
}

const langBadge: Record<string, string> = {
  pytorch: 'bg-orange-900/40 text-orange-300 border-orange-800',
  cuda: 'bg-emerald-900/40 text-emerald-300 border-emerald-800',
  triton: 'bg-purple-900/40 text-purple-300 border-purple-800',
  english: 'bg-sky-900/40 text-sky-300 border-sky-800',
};

export function ExamplesGallery({ onPick }: Props) {
  return (
    <div className="panel p-4">
      <div className="mb-3 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-slate-200">Demo examples</h3>
        <span className="text-xs text-slate-500">click to load</span>
      </div>
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2 lg:grid-cols-3">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.id}
            onClick={() => onPick(ex)}
            className="rounded-md border border-slate-800 bg-slate-900/60 p-3 text-left transition hover:border-amd-red/50 hover:bg-slate-900"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium text-slate-200">{ex.title}</div>
                <div className="mt-1 line-clamp-2 text-xs text-slate-500">{ex.description}</div>
              </div>
              <span
                className={`shrink-0 rounded-md border px-2 py-0.5 text-[10px] uppercase tracking-wider ${langBadge[ex.language]}`}
              >
                {ex.language}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
