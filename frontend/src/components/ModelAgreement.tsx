import {
  AlertTriangle,
  CheckCircle2,
  GitCompareArrows,
  ShieldCheck,
} from 'lucide-react'
import type { AnalysisResponse } from '../types/api'

interface ModelAgreementProps {
  result: AnalysisResponse
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function ModelAgreement({
  result,
}: ModelAgreementProps) {
  const modelsAgree =
    result.primary_prediction.label
    === result.secondary_signal.predicted_label

  const probabilityGap = Math.abs(
    result.primary_prediction.probability
      - result.secondary_signal.probability,
  )

  return (
    <article className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 sm:p-7">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex items-start gap-4">
          <div
            className={
              modelsAgree
                ? 'rounded-2xl bg-emerald-500/10 p-3 text-emerald-300'
                : 'rounded-2xl bg-amber-500/10 p-3 text-amber-300'
            }
          >
            {modelsAgree ? (
              <CheckCircle2 className="size-6" />
            ) : (
              <AlertTriangle className="size-6" />
            )}
          </div>

          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
              Model agreement
            </p>

            <h3 className="mt-2 text-xl font-semibold">
              {modelsAgree
                ? 'Models agree'
                : 'Models disagree'}
            </h3>

            <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">
              V1 remains the authoritative decision source.
              V3 is an exploratory safety signal and cannot
              automatically override the primary model.
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-950/40 px-4 py-3 sm:text-right">
          <p className="text-xs text-slate-500">
            Probability gap
          </p>

          <p className="mt-1 text-xl font-semibold">
            {percentage(probabilityGap)}
          </p>
        </div>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
          <div className="flex items-center gap-2 text-cyan-300">
            <ShieldCheck className="size-4" />
            <span className="text-xs font-semibold uppercase tracking-[0.14em]">
              Primary V1
            </span>
          </div>

          <div className="mt-3 flex items-end justify-between gap-4">
            <p className="font-semibold">
              {result.primary_prediction.label}
            </p>

            <p className="text-sm text-slate-400">
              {percentage(
                result.primary_prediction.probability,
              )}
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-950/30 p-4">
          <div className="flex items-center gap-2 text-violet-300">
            <GitCompareArrows className="size-4" />
            <span className="text-xs font-semibold uppercase tracking-[0.14em]">
              Exploratory V3
            </span>
          </div>

          <div className="mt-3 flex items-end justify-between gap-4">
            <p className="font-semibold">
              {result.secondary_signal.predicted_label}
            </p>

            <p className="text-sm text-slate-400">
              {percentage(
                result.secondary_signal.probability,
              )}
            </p>
          </div>
        </div>
      </div>
    </article>
  )
}

export default ModelAgreement
