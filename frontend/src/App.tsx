import { useState } from 'react';
import { Header } from './components/Header';
import { CodeEditor } from './components/CodeEditor';
import { ExamplesGallery } from './components/ExamplesGallery';
import { PerformanceChart } from './components/PerformanceChart';
import { StatsPanel } from './components/StatsPanel';
import { compileStream, fullPipeline } from './lib/api';
import { EXAMPLES } from './lib/examples';
import type { BenchmarkResult, ExampleEntry, InputLanguage } from './types';

type Phase = 'idle' | 'generating' | 'compiling' | 'benchmarking' | 'done' | 'error';

const PHASE_LABEL: Record<Phase, string> = {
  idle: 'Ready',
  generating: 'Generating HIP kernel...',
  compiling: 'Compiling with hipcc...',
  benchmarking: 'Profiling on MI300X...',
  done: 'Complete',
  error: 'Error',
};

export default function App() {
  const [language, setLanguage] = useState<InputLanguage>('pytorch');
  const [inputCode, setInputCode] = useState<string>(EXAMPLES[0].code);
  const [outputCode, setOutputCode] = useState<string>('');
  const [streaming, setStreaming] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null);

  function loadExample(ex: ExampleEntry) {
    setLanguage(ex.language);
    setInputCode(ex.code);
    setOutputCode('');
    setBenchmark(null);
    setPhase('idle');
    setErrorMsg(null);
  }

  async function runFullPipeline() {
    setPhase('generating');
    setErrorMsg(null);
    setOutputCode('');
    setBenchmark(null);

    try {
      const resp = await fullPipeline({
        input_code: inputCode,
        input_lang: language,
        target: 'mi300x',
        temperature: 0.2,
        max_tokens: 2048,
      });

      setOutputCode(resp.compile.hip_code);

      if (resp.benchmark) {
        setBenchmark(resp.benchmark);
        setPhase('done');
      } else if (resp.benchmark_error) {
        setErrorMsg(`Benchmark unavailable: ${resp.benchmark_error}`);
        setPhase('done');
      } else {
        setPhase('done');
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
      setPhase('error');
    }
  }

  function startStream() {
    setPhase('generating');
    setErrorMsg(null);
    setOutputCode('');
    setBenchmark(null);
    setStreaming(true);

    let acc = '';
    const cancel = compileStream(
      {
        input_code: inputCode,
        input_lang: language,
        target: 'mi300x',
        temperature: 0.2,
        max_tokens: 2048,
      },
      (chunk) => {
        acc += chunk;
        setOutputCode(acc);
      },
      (hipCode) => {
        setOutputCode(hipCode || acc);
        setPhase('done');
        setStreaming(false);
      },
      (err) => {
        setErrorMsg(err.message);
        setPhase('error');
        setStreaming(false);
      },
    );

    return cancel;
  }

  return (
    <div className="flex min-h-screen flex-col">
      <Header />

      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-6">
        <div className="mb-6 panel p-4">
          <h2 className="text-lg font-semibold text-slate-100">
            Turn any GPU code into a hand-tuned MI300X kernel
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Paste PyTorch, CUDA, or Triton (or describe what you want in English).
            ROCmForge fine-tuned Qwen2.5-Coder-7B emits optimized HIP code, compiles it
            with <code className="text-amd-red">hipcc</code>, and benchmarks it against{' '}
            <code className="text-slate-300">torch.compile</code> and{' '}
            <code className="text-slate-300">rocBLAS</code>.
          </p>
        </div>

        <div className="mb-6">
          <ExamplesGallery onPick={loadExample} />
        </div>

        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-400">Input:</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as InputLanguage)}
              className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-200"
            >
              <option value="pytorch">PyTorch</option>
              <option value="cuda">CUDA</option>
              <option value="triton">Triton</option>
              <option value="english">Natural language</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">{PHASE_LABEL[phase]}</span>
            <button
              onClick={startStream}
              disabled={streaming || phase === 'generating'}
              className="btn-secondary"
            >
              Stream generate
            </button>
            <button
              onClick={runFullPipeline}
              disabled={streaming || phase === 'generating'}
              className="btn-primary"
            >
              Compile & Benchmark
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="panel overflow-hidden">
            <div className="border-b border-slate-800 bg-slate-900/60 px-3 py-2 text-xs uppercase tracking-wider text-slate-400">
              Input — {language}
            </div>
            <CodeEditor
              value={inputCode}
              onChange={setInputCode}
              language={language}
              height="520px"
            />
          </div>
          <div className="panel overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900/60 px-3 py-2 text-xs uppercase tracking-wider text-slate-400">
              <span>Output — HIP / ROCm</span>
              {outputCode && (
                <button
                  onClick={() => navigator.clipboard.writeText(outputCode)}
                  className="rounded border border-slate-700 px-2 py-0.5 text-[10px] hover:bg-slate-800"
                >
                  copy
                </button>
              )}
            </div>
            <CodeEditor value={outputCode} language="hip" readOnly height="520px" />
          </div>
        </div>

        {errorMsg && (
          <div className="mt-4 panel border-amber-900/50 bg-amber-950/20 p-3 text-sm text-amber-300">
            {errorMsg}
          </div>
        )}

        {benchmark && (
          <div className="mt-6 space-y-4">
            <StatsPanel result={benchmark} />
            <PerformanceChart result={benchmark} />
          </div>
        )}
      </main>

      <footer className="border-t border-slate-800 py-4 text-center text-xs text-slate-500">
        ROCmForge · AMD Developer Hackathon · Fine-tuned on AMD MI300X
      </footer>
    </div>
  );
}
