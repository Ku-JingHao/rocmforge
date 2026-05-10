import { useState } from 'react';
import { liveCompare } from '../lib/api';
import { EXAMPLES } from '../lib/examples';
import type { CompareResponse, InputLanguage } from '../types';

// ─── AMD pattern detector ─────────────────────────────────────────────────────

interface Pattern {
  label: string;
  regex: RegExp;
  color: string;
}

const AMD_PATTERNS: Pattern[] = [
  { label: 'MFMA intrinsic',        regex: /__builtin_amdgcn_mfma/,           color: 'bg-purple-900/50 text-purple-200 border-purple-700' },
  { label: 'Wavefront-64',          regex: /offset\s*=\s*32|wavefront/i,      color: 'bg-sky-900/50 text-sky-200 border-sky-700' },
  { label: 'gfx942 targeting',      regex: /gfx942|gfx9/,                     color: 'bg-amd-red/20 text-red-200 border-red-700' },
  { label: '__launch_bounds__',     regex: /__launch_bounds__/,               color: 'bg-amber-900/50 text-amber-200 border-amber-700' },
  { label: 'AMD shuffle (no mask)', regex: /__shfl_down(?!_sync)/,            color: 'bg-teal-900/50 text-teal-200 border-teal-700' },
  { label: 'FP16 / half type',      regex: /\bhalf\b|hip_fp16|__half/,        color: 'bg-indigo-900/50 text-indigo-200 border-indigo-700' },
  { label: 'LDS staging',           regex: /__shared__.*\[.*\+\s*\d/,         color: 'bg-green-900/50 text-green-200 border-green-700' },
  { label: '__restrict__ qualifier',regex: /__restrict__/,                    color: 'bg-slate-700/60 text-slate-200 border-slate-600' },
];

function detectPatterns(code: string): Pattern[] {
  return AMD_PATTERNS.filter((p) => p.regex.test(code));
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function PatternBadges({ code, empty }: { code: string; empty?: boolean }) {
  if (empty) return null;
  const found = detectPatterns(code);
  if (found.length === 0) {
    return (
      <div className="flex flex-wrap gap-1 px-3 pb-2">
        <span className="rounded border border-rose-800 bg-rose-950/40 px-2 py-0.5 text-[10px] text-rose-300">
          No MI300X-specific patterns detected
        </span>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap gap-1 px-3 pb-2">
      {found.map((p) => (
        <span
          key={p.label}
          className={`rounded border px-2 py-0.5 text-[10px] font-medium ${p.color}`}
        >
          ✓ {p.label}
        </span>
      ))}
    </div>
  );
}

interface OutputPaneProps {
  label: string;
  modelName: string;
  code: string;
  timeS: number;
  isGood: boolean;
  unavailable?: boolean;
  unavailableMsg?: string;
  loading?: boolean;
}

function OutputPane({
  label,
  modelName,
  code,
  timeS,
  isGood,
  unavailable,
  unavailableMsg,
  loading,
}: OutputPaneProps) {
  const patternCount = code ? detectPatterns(code).length : 0;

  return (
    <div
      className={`flex min-w-0 flex-1 flex-col overflow-hidden rounded-lg border ${
        isGood ? 'border-green-800/60' : 'border-rose-800/60'
      }`}
    >
      {/* Header */}
      <div
        className={`flex items-center justify-between gap-2 px-3 py-2 text-xs ${
          isGood
            ? 'bg-green-950/50 border-b border-green-900/50'
            : 'bg-rose-950/40 border-b border-rose-900/40'
        }`}
      >
        <div className="flex flex-col">
          <span className={`font-semibold ${isGood ? 'text-green-200' : 'text-rose-200'}`}>
            {label}
          </span>
          <span className="text-[10px] text-slate-500 truncate max-w-[240px]">{modelName}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {code && (
            <span className="text-slate-500">
              {timeS > 0 ? `${timeS.toFixed(1)}s` : ''}
            </span>
          )}
          {code && (
            <span
              className={`rounded px-2 py-0.5 text-[10px] font-medium ${
                patternCount > 0
                  ? 'bg-green-900/50 text-green-300'
                  : 'bg-rose-900/50 text-rose-300'
              }`}
            >
              {patternCount > 0 ? `${patternCount} AMD patterns` : 'No AMD patterns'}
            </span>
          )}
        </div>
      </div>

      {/* Body */}
      {loading ? (
        <div className="flex flex-1 items-center justify-center bg-slate-950/80 py-12 text-sm text-slate-500">
          <span className="animate-pulse">Generating…</span>
        </div>
      ) : unavailable ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-slate-950/80 p-6">
          <div className="text-3xl">🔌</div>
          <div className="text-sm font-medium text-slate-300">Base model not loaded</div>
          <div className="max-w-xs text-center text-xs text-slate-500">
            {unavailableMsg ||
              'The base model loads on first request (~30s). Start the server with ROCMFORGE_BASE_MODEL_PATH set, then try again.'}
          </div>
        </div>
      ) : code ? (
        <>
          <pre className="flex-1 overflow-auto bg-slate-950/80 p-3 text-xs leading-5 text-slate-300">
            <code>{code}</code>
          </pre>
          <PatternBadges code={code} />
        </>
      ) : (
        <div className="flex flex-1 items-center justify-center bg-slate-950/80 py-12 text-sm text-slate-500">
          Output will appear here
        </div>
      )}
    </div>
  );
}

// ─── Diff summary ─────────────────────────────────────────────────────────────

function DiffSummary({ result }: { result: CompareResponse }) {
  const tuned = detectPatterns(result.rocmforge.hip_code);
  const base = detectPatterns(result.base.hip_code);
  const baseLabels = new Set(base.map((p) => p.label));
  const gained = tuned.filter((p) => !baseLabels.has(p.label));
  const shared = tuned.filter((p) => baseLabels.has(p.label));

  return (
    <div className="panel p-4 space-y-3">
      <div className="text-sm font-semibold text-slate-200">What fine-tuning added</div>
      <div className="grid gap-3 md:grid-cols-3">
        {/* Gained by fine-tuning */}
        <div className="rounded-lg border border-green-900/50 bg-green-950/20 p-3">
          <div className="mb-2 text-[11px] uppercase tracking-wider text-green-400">
            Added by fine-tuning ({gained.length})
          </div>
          {gained.length === 0 ? (
            <p className="text-xs text-slate-500">None detected above threshold</p>
          ) : (
            <div className="flex flex-wrap gap-1">
              {gained.map((p) => (
                <span key={p.label} className={`rounded border px-2 py-0.5 text-[10px] ${p.color}`}>
                  {p.label}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Present in both */}
        <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-3">
          <div className="mb-2 text-[11px] uppercase tracking-wider text-slate-400">
            Present in both ({shared.length})
          </div>
          {shared.length === 0 ? (
            <p className="text-xs text-slate-500">—</p>
          ) : (
            <div className="flex flex-wrap gap-1">
              {shared.map((p) => (
                <span key={p.label} className={`rounded border px-2 py-0.5 text-[10px] ${p.color}`}>
                  {p.label}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Timing */}
        <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-3">
          <div className="mb-2 text-[11px] uppercase tracking-wider text-slate-400">
            Generation time
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-400">ROCmForge</span>
              <span className="font-semibold text-green-300">
                {result.rocmforge.time_s.toFixed(1)}s
              </span>
            </div>
            {result.base.available && (
              <div className="flex justify-between">
                <span className="text-slate-400">Base model</span>
                <span className="text-slate-300">{result.base.time_s.toFixed(1)}s</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

type RunState = 'idle' | 'running' | 'done' | 'error';

export function LiveComparePage() {
  const [language, setLanguage] = useState<InputLanguage>('cuda');
  const [inputCode, setInputCode] = useState(EXAMPLES.find((e) => e.id === 'cuda-softmax')?.code ?? EXAMPLES[0].code);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [runState, setRunState] = useState<RunState>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  async function runComparison() {
    setRunState('running');
    setResult(null);
    setErrorMsg(null);
    try {
      const resp = await liveCompare({
        input_code: inputCode,
        input_lang: language,
        target: 'mi300x',
        temperature: 0.2,
        max_tokens: 1024,
      });
      setResult(resp);
      setRunState('done');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
      setRunState('error');
    }
  }

  return (
    <div className="mx-auto w-full max-w-7xl px-6 py-6 space-y-5">
      {/* Header */}
      <div className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold text-slate-100">
              Live Comparison —{' '}
              <span className="text-slate-400 font-normal text-base">
                Base model
              </span>
              {' '}vs{' '}
              <span className="text-amd-red">ROCmForge</span>
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Same prompt, both models run live on the server. ROCmForge fine-tuned output
              on the right; base Qwen2.5-Coder-7B on the left. AMD-specific patterns detected
              automatically.
            </p>
          </div>
        </div>
      </div>

      {/* Input + controls */}
      <div className="panel overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/60 px-3 py-2">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400">Input language:</span>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as InputLanguage)}
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200"
            >
              <option value="pytorch">PyTorch</option>
              <option value="cuda">CUDA</option>
              <option value="triton">Triton</option>
              <option value="english">English</option>
            </select>
            {/* Quick example loader */}
            <span className="text-xs text-slate-600">|</span>
            {EXAMPLES.slice(0, 4).map((ex) => (
              <button
                key={ex.id}
                onClick={() => { setLanguage(ex.language); setInputCode(ex.code); setResult(null); }}
                className="text-xs text-slate-500 hover:text-slate-300 transition"
              >
                {ex.title}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            {runState === 'done' && (
              <span className="text-xs font-medium text-green-400">✓ Complete</span>
            )}
            {runState === 'error' && (
              <span className="text-xs font-medium text-red-400">Error</span>
            )}
            <button
              onClick={runComparison}
              disabled={runState === 'running'}
              className="btn-primary text-sm"
            >
              {runState === 'running' ? 'Running…' : 'Run Live Comparison'}
            </button>
          </div>
        </div>
        <textarea
          value={inputCode}
          onChange={(e) => setInputCode(e.target.value)}
          className="w-full resize-none bg-slate-950/60 p-4 font-mono text-xs text-slate-300 focus:outline-none"
          rows={10}
          spellCheck={false}
          placeholder="Paste PyTorch, CUDA, Triton, or describe your kernel in English…"
        />
      </div>

      {errorMsg && (
        <div className="panel border-amber-900/50 bg-amber-950/20 p-3 text-sm text-amber-300">
          {errorMsg}
        </div>
      )}

      {/* Side-by-side outputs */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2" style={{ minHeight: 440 }}>
        <OutputPane
          label="Base Qwen2.5-Coder-7B"
          modelName="No fine-tuning — standard instruction-following model"
          code={result?.base.hip_code ?? ''}
          timeS={result?.base.time_s ?? 0}
          isGood={false}
          unavailable={result !== null && !result.base.available}
          unavailableMsg={result?.base.error ?? undefined}
          loading={runState === 'running'}
        />
        <OutputPane
          label="ROCmForge (fine-tuned)"
          modelName="Qwen2.5-Coder-7B + QLoRA on 25K AMD kernel examples"
          code={result?.rocmforge.hip_code ?? ''}
          timeS={result?.rocmforge.time_s ?? 0}
          isGood={true}
          loading={runState === 'running'}
        />
      </div>

      {/* Diff summary — only shown after a completed run */}
      {result && runState === 'done' && <DiffSummary result={result} />}

      {/* How it works note */}
      <div className="panel border-slate-700/40 p-4 text-xs text-slate-500 space-y-1">
        <span className="font-semibold text-slate-400">How this works:</span>{' '}
        Both models receive the exact same system prompt and user input. The base model
        (Qwen2.5-Coder-7B-Instruct) is loaded from Hugging Face on the first request.
        ROCmForge is the same base model fine-tuned on 25K curated AMD HIP kernel examples
        using QLoRA on a single MI300X. Patterns like MFMA intrinsics, wavefront-64
        reductions, and gfx942 targeting are automatically detected in both outputs and
        highlighted above.
      </div>
    </div>
  );
}
