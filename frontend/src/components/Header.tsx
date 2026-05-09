import { useEffect, useState } from 'react';
import { getHealth } from '../lib/api';
import type { HealthResponse } from '../types';

export function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let mounted = true;
    getHealth()
      .then((h) => {
        if (mounted) setHealth(h);
      })
      .catch(() => {
        if (mounted) setHealth(null);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const statusColor =
    health?.status === 'ok'
      ? 'bg-emerald-500'
      : health?.status === 'degraded'
        ? 'bg-amber-500'
        : 'bg-rose-500';

  return (
    <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <div className="flex items-baseline gap-3">
          <h1 className="text-2xl font-bold tracking-tight">
            ROCm<span className="text-amd-red">Forge</span>
          </h1>
          <span className="text-xs text-slate-500">The AMD Performance Compiler</span>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${statusColor}`} />
            <span className="text-slate-300">
              {health
                ? `${health.model_name} ${health.status === 'ok' ? '· ready' : '· loading'}`
                : 'connecting...'}
            </span>
          </div>
          {health?.gpu_name && (
            <span className="rounded-md border border-slate-700 px-2 py-1 text-slate-300">
              {health.gpu_name}
            </span>
          )}
        </div>
      </div>
    </header>
  );
}
