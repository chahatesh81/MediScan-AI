import {
  Activity,
  CircleDot,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

function WorkspaceHeader() {
  return (
    <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-5 sm:px-6 lg:flex-row lg:items-center lg:justify-between lg:px-8">
        <div className="flex items-center gap-4">
          <div className="rounded-2xl border border-cyan-500/20 bg-cyan-500/10 p-3 text-cyan-300">
            <Activity className="size-6" />
          </div>

          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-lg font-semibold tracking-tight text-slate-100">
                MediScan AI
              </h1>

              <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-300">
                Analysis workspace
              </span>
            </div>

            <p className="mt-1 text-sm text-slate-500">
              Explainable chest X-ray decision support
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-400">
            <CircleDot className="size-3.5 text-emerald-400" />
            V1 policy active
          </div>

          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-400">
            <ShieldCheck className="size-3.5 text-cyan-300" />
            Review safeguards
          </div>

          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs text-slate-400">
            <Sparkles className="size-3.5 text-violet-300" />
            Grad-CAM enabled
          </div>
        </div>
      </div>
    </header>
  )
}

export default WorkspaceHeader
