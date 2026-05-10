import type {
  CompileRequest,
  CompileResponse,
  CompareResponse,
  BenchmarkResult,
  FullPipelineResponse,
  HealthResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return resp.json();
}

export async function getHealth(): Promise<HealthResponse> {
  return jsonFetch<HealthResponse>('/api/health');
}

export async function compile(req: CompileRequest): Promise<CompileResponse> {
  return jsonFetch<CompileResponse>('/api/compile', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function benchmark(hip_code: string, operation = 'auto'): Promise<BenchmarkResult> {
  return jsonFetch<BenchmarkResult>('/api/benchmark', {
    method: 'POST',
    body: JSON.stringify({
      hip_code,
      operation,
      problem_size: { M: 4096, N: 4096, K: 4096 },
      dtype: 'fp16',
    }),
  });
}

export async function liveCompare(req: CompileRequest): Promise<CompareResponse> {
  return jsonFetch<CompareResponse>('/api/compare', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

export async function fullPipeline(req: CompileRequest): Promise<FullPipelineResponse> {
  return jsonFetch<FullPipelineResponse>('/api/full_pipeline', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Stream compile output via Server-Sent Events.
 * Calls onChunk for each token chunk and onDone with the final HIP code.
 */
export function compileStream(
  req: CompileRequest,
  onChunk: (chunk: string) => void,
  onDone: (hipCode: string) => void,
  onError: (err: Error) => void,
): () => void {
  let cancelled = false;
  let buffer = '';

  fetch(`${API_BASE}/api/compile/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
    .then(async (resp) => {
      if (!resp.ok || !resp.body) {
        throw new Error(`Stream error: ${resp.status} ${resp.statusText}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      while (!cancelled) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() ?? '';

        for (const ev of events) {
          if (!ev.trim()) continue;

          let eventName = 'message';
          let data = '';
          for (const line of ev.split('\n')) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim();
            if (line.startsWith('data:')) data = line.slice(5).trim();
          }
          if (!data) continue;

          try {
            const parsed = JSON.parse(data);
            if (eventName === 'done') {
              onDone(parsed.hip_code || '');
            } else if (parsed.chunk) {
              onChunk(parsed.chunk);
            }
          } catch {
            // ignore malformed events
          }
        }
      }
    })
    .catch((err: Error) => {
      if (!cancelled) onError(err);
    });

  return () => {
    cancelled = true;
  };
}
