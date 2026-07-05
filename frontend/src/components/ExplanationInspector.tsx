import {
  AlertTriangle,
  BrainCircuit,
  Eye,
  Info,
  ScanSearch,
  ShieldAlert,
  Target,
} from 'lucide-react'
import type { AnalysisResponse } from '../types/api'

interface ExplanationInspectorProps {
  result: AnalysisResponse
  originalUrl: string
  overlayUrl: string | null
}

function percentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatMode(
  mode: AnalysisResponse['explanation']['mode'],
): string {
  return mode === 'positive_gradcam'
    ? 'Positive Grad-CAM'
    : 'Absolute attribution'
}

function formatStatus(
  status:
    AnalysisResponse['explanation_quality']['quality_status'],
): string {
  return status
    .split('_')
    .map((word) => (
      word.charAt(0) + word.slice(1).toLowerCase()
    ))
    .join(' ')
}

function ExplanationInspector({
  result,
  originalUrl,
  overlayUrl,
}: ExplanationInspectorProps) {
  const quality = result.explanation_quality
  const explanation = result.explanation

  const statusTone = quality.display_warning
    ? 'border-rose-500/30 bg-rose-500/10 text-rose-200'
    : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'

  return (
    <section className="overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/60">
      <div className="border-b border-slate-800 p-5 sm:p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex items-start gap-4">
            <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-violet-500/10">
              <BrainCircuit className="size-5 text-violet-400" />
            </div>

            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                Explanation inspector
              </p>

              <h2 className="mt-2 text-xl font-semibold tracking-tight text-slate-100">
                Grad-CAM attribution review
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                Compare the uploaded radiograph with the V1
                attribution overlay and inspect spatial
                reliability diagnostics.
              </p>
            </div>
          </div>

          <div
            className={`inline-flex w-fit items-center gap-2 rounded-full border px-3 py-2 text-xs font-medium ${statusTone}`}
          >
            {quality.display_warning ? (
              <ShieldAlert className="size-4" />
            ) : (
              <Target className="size-4" />
            )}
            {formatStatus(quality.quality_status)}
          </div>
        </div>
      </div>

      <div className="grid gap-px bg-slate-800 sm:grid-cols-2 xl:grid-cols-4">
        <div className="bg-slate-900/95 p-5">
          <ScanSearch className="size-5 text-violet-400" />
          <p className="mt-3 text-sm font-medium text-slate-200">
            Attribution mode
          </p>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            {formatMode(explanation.mode)}
          </p>
        </div>

        <div className="bg-slate-900/95 p-5">
          <BrainCircuit className="size-5 text-cyan-400" />
          <p className="mt-3 text-sm font-medium text-slate-200">
            Feature layer
          </p>
          <p className="mt-1 break-all text-xs leading-5 text-slate-500">
            {explanation.last_conv_layer}
          </p>
        </div>

        <div className="bg-slate-900/95 p-5">
          <Target className="size-5 text-emerald-400" />
          <p className="mt-3 text-sm font-medium text-slate-200">
            Thorax attention
          </p>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            {percentage(quality.thorax_energy_ratio)} energy ratio
          </p>
        </div>

        <div className="bg-slate-900/95 p-5">
          <ShieldAlert className="size-5 text-amber-400" />
          <p className="mt-3 text-sm font-medium text-slate-200">
            Border attention
          </p>
          <p className="mt-1 text-xs leading-5 text-slate-500">
            {percentage(quality.border_energy_ratio)} energy ratio
          </p>
        </div>
      </div>

      <div className="grid gap-px bg-slate-800 md:grid-cols-2">
        <article className="bg-slate-950/80">
          <div className="flex items-center gap-3 border-b border-slate-800 px-5 py-4">
            <Eye className="size-5 text-cyan-400" />

            <div>
              <h3 className="font-medium text-slate-100">
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

        <article className="bg-slate-950/80">
          <div className="flex items-center gap-3 border-b border-slate-800 px-5 py-4">
            <BrainCircuit className="size-5 text-violet-400" />

            <div>
              <h3 className="font-medium text-slate-100">
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

      <div className="space-y-4 border-t border-slate-800 p-5 sm:p-6">
        {quality.display_warning && (
          <div className="flex gap-4 rounded-2xl border border-rose-500/30 bg-rose-500/10 p-5">
            <AlertTriangle className="mt-0.5 size-5 shrink-0 text-rose-300" />

            <div>
              <h3 className="font-medium text-rose-200">
                Explanation reliability warning
              </h3>

              <p className="mt-1 text-sm leading-6 text-rose-100/70">
                Quality status: {quality.quality_status}.
                The visualization may emphasize
                non-anatomical regions and must not be
                interpreted as lesion segmentation.
              </p>

              {quality.attribution_note && (
                <p className="mt-2 text-xs leading-5 text-rose-100/60">
                  {quality.attribution_note}
                </p>
              )}
            </div>
          </div>
        )}

        <div className="flex gap-4 rounded-2xl border border-slate-800 bg-slate-950/50 p-5">
          <Info className="mt-0.5 size-5 shrink-0 text-cyan-400" />

          <div>
            <h3 className="font-medium text-slate-200">
              Interpretation context
            </h3>

            <p className="mt-1 text-sm leading-6 text-slate-400">
              Region definition: {quality.region_definition}.
              Heatmap resolution:{' '}
              {explanation.raw_heatmap_shape.join(' × ')}.
              Output resolution: {explanation.output_width} ×{' '}
              {explanation.output_height}.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

export default ExplanationInspector
