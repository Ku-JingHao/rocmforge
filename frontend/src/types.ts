export type InputLanguage = 'pytorch' | 'cuda' | 'triton' | 'english';

export interface CompileRequest {
  input_code: string;
  input_lang: InputLanguage;
  target?: 'mi300x';
  temperature?: number;
  max_tokens?: number;
}

export interface CompileResponse {
  request_id: string;
  hip_code: string;
  raw_output: string;
  extraction_warnings: string[];
}

export interface BenchmarkResult {
  compile_success: boolean;
  compile_error?: string | null;
  tflops?: number | null;
  kernel_time_ms?: number | null;
  occupancy_pct?: number | null;
  memory_bw_gb_s?: number | null;
  baselines: {
    eager?: number | null;
    compiled?: number | null;
    rocblas?: number | null;
  };
  speedup_vs_eager?: number | null;
  speedup_vs_compile?: number | null;
  pct_of_rocblas?: number | null;
}

export interface FullPipelineResponse {
  compile: CompileResponse;
  benchmark: BenchmarkResult | null;
  benchmark_error?: string;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  model_loaded: boolean;
  model_name: string;
  gpu_name?: string | null;
  gpu_memory_gb?: number | null;
  rocm_version?: string | null;
}

export interface ExampleEntry {
  id: string;
  title: string;
  description: string;
  language: InputLanguage;
  code: string;
}
