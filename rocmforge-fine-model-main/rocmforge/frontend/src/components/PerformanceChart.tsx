import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { BenchmarkResult } from '../types';

interface Props {
  result: BenchmarkResult;
}

const COLOR_BASELINE = '#475569';
const COLOR_COMPILED = '#0EA5E9';
const COLOR_ROCBLAS = '#F59E0B';
const COLOR_ROCMFORGE = '#ED1C24';

export function PerformanceChart({ result }: Props) {
  const data = [
    {
      name: 'PyTorch eager',
      tflops: result.baselines.eager ?? 0,
      color: COLOR_BASELINE,
    },
    {
      name: 'torch.compile',
      tflops: result.baselines.compiled ?? 0,
      color: COLOR_COMPILED,
    },
    {
      name: 'rocBLAS',
      tflops: result.baselines.rocblas ?? 0,
      color: COLOR_ROCBLAS,
    },
    {
      name: 'ROCmForge',
      tflops: result.tflops ?? 0,
      color: COLOR_ROCMFORGE,
    },
  ].filter((d) => d.tflops > 0);

  if (data.length === 0) {
    return (
      <div className="panel flex h-64 items-center justify-center text-slate-500">
        No performance data available
      </div>
    );
  }

  return (
    <div className="panel p-4">
      <div className="mb-4 flex items-baseline justify-between">
        <h3 className="text-base font-semibold text-slate-200">Performance vs. baselines</h3>
        <span className="text-xs text-slate-500">TFLOPS · higher is better</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
          <YAxis stroke="#94a3b8" fontSize={12} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              border: '1px solid #334155',
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(v: number) => [`${v.toFixed(2)} TFLOPS`, 'Performance']}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Bar dataKey="tflops" name="TFLOPS" radius={[6, 6, 0, 0]}>
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
