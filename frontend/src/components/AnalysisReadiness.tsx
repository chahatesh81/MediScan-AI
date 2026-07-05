import {
  CheckCircle2,
  FileImage,
  FlaskConical,
  LoaderCircle,
  ScanSearch,
  ShieldCheck,
} from 'lucide-react'

interface AnalysisReadinessProps {
  file: File
  isAnalyzing: boolean
  onAnalyze: () => void
}

function formatFileSize(bytes: number) {
  const megabytes = bytes / 1024 / 1024

  if (megabytes >= 0.1) {
    return `${megabytes.toFixed(2)} MB`
  }

  return `${Math.max(1, Math.round(bytes / 1024))} KB`
}

function formatFileType(type: string) {
  if (type === 'image/jpeg') {
    return 'JPEG'
  }

  if (type === 'image/png') {
    return 'PNG'
  }

  return type
}

function AnalysisReadiness({
  file,
  isAnalyzing,
  onAnalyze,
}: AnalysisReadinessProps) {
  const checks = [
    {
      label: 'Upload validated',
      detail: `${formatFileType(file.type)} · ${formatFileSize(file.size)}`,
      icon: FileImage,
    },
    {
      label: 'Primary classifier ready',
      detail: 'Validated V1 decision model',
      icon: ShieldCheck,
    },
    {
      label: 'Safety signal enabled',
      detail: 'Exploratory V3 disagreement check',
      icon: FlaskConical,
    },
    {
      label: 'Explanation enabled',
      detail: 'Grad-CAM with attention diagnostics',
      icon: ScanSearch,
    },
  ]

  return (
    <div className="mt-5 overflow-hidden rounded-3xl border border-slate-800 bg-slate-900/70">
      <div className="border-b border-slate-800 px-5 py-5 sm:px-6">
        <div className="flex items-start gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-emerald-500/10">
            <CheckCircle2 className="size-5 text-emerald-400" />
          </div>

          <div>
            <p className="font-semibold text-slate-100">
              Ready for analysis
            </p>

            <p className="mt-1 text-sm leading-6 text-slate-400">
              The image passed client-side validation and
              the analysis workflow is configured.
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-px bg-slate-800 sm:grid-cols-2 lg:grid-cols-4">
        {checks.map((check) => {
          const Icon = check.icon

          return (
            <div
              key={check.label}
              className="bg-slate-900/95 p-5"
            >
              <Icon className="size-5 text-cyan-400" />

              <p className="mt-3 text-sm font-medium text-slate-200">
                {check.label}
              </p>

              <p className="mt-1 text-xs leading-5 text-slate-500">
                {check.detail}
              </p>
            </div>
          )
        })}
      </div>

      <div className="border-t border-slate-800 p-5 sm:p-6">
        <button
          type="button"
          onClick={onAnalyze}
          disabled={isAnalyzing}
          className="flex w-full items-center justify-center gap-3 rounded-2xl bg-cyan-400 px-6 py-4 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-wait disabled:opacity-70"
        >
          {isAnalyzing ? (
            <>
              <LoaderCircle className="size-5 animate-spin" />
              Running AI analysis...
            </>
          ) : (
            <>
              <ScanSearch className="size-5" />
              Run full analysis
            </>
          )}
        </button>

        <p className="mt-3 text-center text-xs leading-5 text-slate-500">
          Decision-support output only · Human review required
        </p>
      </div>
    </div>
  )
}

export default AnalysisReadiness
