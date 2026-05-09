import type { BenchmarkResult } from '../types';

interface Props {
  result: BenchmarkResult;
}

interface StatProps {
  label: string;
  value: string;
  hint?: string;
  highlight?: boolean;
}

function Stat({ label, value, hint, highlight }: StatProps) {
  return (
    <div className="stat-card">
      <div className="text-[11px] uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-1 text-2xl font-bold ${highlight ? 'text-amd-red' : 'text-slate-100'}`}>
        {value}
      </div>
      {hint && <div className="mt-0.5 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

function fmt(v: number | null | undefined, digits = 2, suffix = ''): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${v.toFixed(digits)}${suffix}`;
}

export function StatsPanel({ result }: Props) {
  if (!result.compile_success) {
    return (
      <div className="panel border-rose-900/50 bg-rose-950/20 p-4">
        <div className="text-sm font-semibold text-rose-400">Compilation failed</div>
        <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap text-xs text-rose-300/80">
          {result.compile_error || 'Unknown error'}
        </pre>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      <Stat
        label="ROCmForge"
        value={fmt(result.tflops, 2, ' TF')}
        hint="Generated kernel"
        highlight
      />
      <Stat
        label="Kernel time"
        value={fmt(result.kernel_time_ms, 3, ' ms')}
        hint="Per invocation"
      />
      <Stat
        label="Occupancy"
        value={fmt(result.occupancy_pct, 1, '%')}
        hint="Wave-64 utilization"
      />
      <Stat
        label="HBM3 BW"
        value={fmt(result.memory_bw_gb_s, 0, ' GB/s')}
        hint="Memory throughput"
      />
      <Stat
        label="vs eager"
        value={result.speedup_vs_eager ? `${fmt(result.speedup_vs_eager)}×` : '—'}
        hint="PyTorch baseline"
      />
      <Stat
        label="vs torch.compile"
        value={result.speedup_vs_compile ? `${fmt(result.speedup_vs_compile)}×` : '—'}
        hint="Inductor compiled"
      />
      <Stat
        label="% of rocBLAS"
        value={fmt(result.pct_of_rocblas, 1, '%')}
        hint="Hand-tuned ceiling"
      />
      <Stat
        label="Status"
        value="✓ Compiled"
        hint="hipcc + rocprof"
      />
    </div>
  );
}
