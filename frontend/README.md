# ROCmForge — Frontend

A live demo UI for the ROCmForge AMD Performance Compiler.

## Stack

- **Vite + React 18 + TypeScript** — fast dev loop, modern ergonomics
- **Tailwind CSS** — utility-first styling, AMD-themed
- **Monaco Editor** — VS Code's editor for input/output code panes
- **Recharts** — TFLOPS comparison bar chart
- **Server-Sent Events** — streams tokens from the model in real-time

## Run locally

```bash
cd rocmforge/frontend
npm install
npm run dev
```

Open http://localhost:5173

By default the dev server proxies `/api/*` to `http://localhost:8001` (the FastAPI
backend). Override with:

```bash
VITE_API_URL=https://your-mi300x-host:8001 npm run dev
```

## Build for production

```bash
npm run build
npm run preview
```

The static bundle goes to `dist/`. You can serve it from the FastAPI app or
any static host (Nginx, Caddy, S3, Cloudflare Pages).

## Layout

```
src/
├── App.tsx                # main page — split editor + run buttons + results
├── main.tsx               # React entry point
├── index.css              # Tailwind + global styles
├── types.ts               # shared TypeScript types
├── components/
│   ├── Header.tsx         # AMD-branded header with health/GPU status
│   ├── CodeEditor.tsx     # Monaco wrapper
│   ├── ExamplesGallery.tsx # demo safety-net (pre-baked examples)
│   ├── StatsPanel.tsx     # TFLOPS, occupancy, kernel time stats
│   └── PerformanceChart.tsx # Recharts bar comparison
└── lib/
    ├── api.ts             # fetch + SSE clients for the backend
    └── examples.ts        # pre-baked demo prompts
```

## Endpoints used

| Endpoint                | Purpose                                |
| ----------------------- | -------------------------------------- |
| `GET  /api/health`      | Show GPU/model status in header        |
| `POST /api/full_pipeline` | One-shot generate + compile + benchmark |
| `POST /api/compile/stream` | SSE token streaming for live demo    |

See [`../inference/server.py`](../inference/server.py) for the backend.
