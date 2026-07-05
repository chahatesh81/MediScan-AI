import {
  AlertTriangle,
  BrainCircuit,
  Eye,
  ShieldAlert,
} from 'lucide-react'
import type { AnalysisResponse } from '../types/api'

interface AnalysisResultProps {
  result: AnalysisResponse
  originalUrl: string
  overlayUrl: string | null
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function AnalysisResult({
  result,
  originalUrl,
  overlayUrl,
}: AnalysisResultProps) {
  const isPneumonia =
    result.decision.final_label === 'PNEUMONIA'

  return (
    <section className="mt-8 space-y-5">
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(260px,0.8fr)]">
        <article className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 sm:p-7">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            Authoritative result
          </p>

          <div className="mt-6 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p
                className={
                  isPneumonia
                    ? 'text-4xl font-semibold tracking-tight text-amber-300 sm:text-5xl'
                    : 'text-4xl font-semibold tracking-tight text-emerald-300 sm:text-5xl'
                }
              >
                {result.decision.final_label}
              </p>

              <p className="mt-3 text-sm text-slate-400">
                Decision source:{' '}
                <span className="font-medium text-slate-300">
                  {result.decision.source}
                </span>
              </p>
            </div>

            <div className="sm:text-right">
              <p className="text-3xl font-semibold tracking-tight sm:text-4xl">
                {percentage(
                  result.primary_prediction.probability,
                )}
              </p>

              <p className="mt-2 text-xs text-slate-500">
                V1 pneumonia probability
              </p>
            </div>
          </div>
        </article>

        <article className="rounded-3xl border border-violet-500/20 bg-violet-500/[0.06] p-6 sm:p-7">
          <BrainCircuit className="size-6 text-violet-400" />

          <p className="mt-5 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
            Exploratory V3 signal
          </p>

          <p className="mt-3 text-2xl font-semibold">
            {result.secondary_signal.predicted_label}
          </p>

          <p className="mt-2 text-sm text-slate-400">
            {percentage(
              result.secondary_signal.probability,
            )}{' '}
            pneumonia probability
          </p>

          <p className="mt-5 text-xs leading-5 text-slate-500">
            Informational only. Automatic override is disabled.
          </p>
        </article>
      </div>

      {result.decision.manual_review_recommended && (
        <div className="flex gap-4 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5">
          <AlertTriangle className="mt-0.5 size-5 shrink-0 text-amber-300" />

          <div>
            <h3 className="font-medium text-amber-200">
              Manual review recommended
            </h3>

            <p className="mt-1 text-sm leading-6 text-amber-100/70">
              The primary and exploratory models produced a
              safety-relevant disagreement.
            </p>
          </div>
        </div>
      )}

      <div className="grid gap-5 md:grid-cols-2">
        <article className="overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/70">
          <div className="flex items-center gap-3 border-b border-slate-800 px-5 py-4">
            <Eye className="size-5 text-cyan-400" />

            <div>
              <h3 className="font-medium">
                Original X-ray
              </h3>

              <p className="mt-0.5 text-xs text-slate-500">
                Uploaded source image
              </p>
            </div>
          </div>

          <div className="flex aspect-[4/3] min-h-64 items-center justify-center bg-black/30 p-3 sm:p-4">
            <img
              src={originalUrl}
              alt="Original chest X-ray"
              className="h-full max-h-[640px] w-full rounded-2xl object-contain"
            />
          </div>
        </article>

        <article className="overflow-hidden rounded-3xl border border-violet-500/20 bg-slate-900/70">
          <div className="flex items-center gap-3 border-b border-slate-800 px-5 py-4">
            <BrainCircuit className="size-5 text-violet-400" />

            <div>
              <h3 className="font-medium">
                Grad-CAM overlay
              </h3>

              <p className="mt-0.5 text-xs text-slate-500">
                V1 attribution visualization
              </p>
            </div>
          </div>

          <div className="flex aspect-[4/3] min-h-64 items-center justify-center bg-black/30 p-3 sm:p-4">
            {overlayUrl ? (
              <img
                src={overlayUrl}
                alt="Grad-CAM explanation overlay"
                className="h-full max-h-[640px] w-full rounded-2xl object-contain"
              />
            ) : (
              <div className="px-6 text-center">
                <BrainCircuit className="mx-auto size-8 text-slate-700" />

                <p className="mt-3 text-sm text-slate-500">
                  Explanation visualization unavailable.
                </p>
              </div>
            )}
          </div>
        </article>
      </div>

      {result.explanation_quality.display_warning && (
        <div className="flex gap-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5">
          <ShieldAlert className="mt-0.5 size-5 shrink-0 text-rose-300" />

          <div>
            <h3 className="font-medium text-rose-200">
              Explanation reliability warning
            </h3>

            <p className="mt-1 text-sm leading-6 text-rose-100/70">
              Quality status:{' '}
              {result.explanation_quality.quality_status}.
              The visualization may emphasize non-anatomical
              regions and must not be interpreted as lesion
              segmentation.
            </p>

            {result.explanation_quality.attribution_note && (
              <p className="mt-2 text-xs leading-5 text-rose-100/60">
                {
                  result.explanation_quality
                    .attribution_note
                }
              </p>
            )}
          </div>
        </div>
      )}

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <p className="text-sm leading-6 text-slate-400">
          {result.disclaimer}
        </p>
      </div>
    </section>
  )
}

export default AnalysisResult
