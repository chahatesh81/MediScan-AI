import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  Gauge,
  ShieldAlert,
  Target,
} from 'lucide-react'
import type { AnalysisResponse } from '../types/api'
import ExplanationInspector from './ExplanationInspector'
import MetricCard from './MetricCard'
import ModelAgreement from './ModelAgreement'

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

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="V1 probability"
          value={percentage(
            result.primary_prediction.probability,
          )}
          detail={`Decision threshold: ${percentage(
            result.primary_prediction.threshold,
          )}`}
          icon={Activity}
          tone={
            isPneumonia ? 'amber' : 'emerald'
          }
        />

        <MetricCard
          label="V3 probability"
          value={percentage(
            result.secondary_signal.probability,
          )}
          detail={`Exploratory threshold: ${percentage(
            result.secondary_signal.threshold,
          )}`}
          icon={Gauge}
          tone="violet"
        />

        <MetricCard
          label="Thorax attention"
          value={percentage(
            result.explanation_quality
              .thorax_energy_ratio,
          )}
          detail="Geometric proxy, not an anatomical lung mask"
          icon={Target}
          tone="cyan"
        />

        <MetricCard
          label="Border attention"
          value={percentage(
            result.explanation_quality
              .border_energy_ratio,
          )}
          detail={
            result.explanation_quality.peak_in_border
              ? 'Peak attribution occurs near the image border'
              : 'Peak attribution is outside the border region'
          }
          icon={ShieldAlert}
          tone={
            result.explanation_quality
              .border_energy_ratio >= 0.5
              ? 'amber'
              : 'emerald'
          }
        />
      </div>

      <ModelAgreement result={result} />

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

      <ExplanationInspector
        result={result}
        originalUrl={originalUrl}
        overlayUrl={overlayUrl}
      />

      <div className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
        <p className="text-sm leading-6 text-slate-400">
          {result.disclaimer}
        </p>
      </div>
    </section>
  )
}

export default AnalysisResult
