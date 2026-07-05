import {
  Check,
  FileImage,
  LoaderCircle,
  ScanSearch,
  Upload,
} from 'lucide-react'

type WorkflowStage =
  | 'upload'
  | 'ready'
  | 'analyzing'
  | 'complete'

interface WorkflowProgressProps {
  stage: WorkflowStage
}

const steps = [
  {
    key: 'upload',
    label: 'Upload',
    detail: 'Select X-ray',
    icon: Upload,
  },
  {
    key: 'ready',
    label: 'Validate',
    detail: 'File ready',
    icon: FileImage,
  },
  {
    key: 'analyzing',
    label: 'Analyze',
    detail: 'Run models',
    icon: ScanSearch,
  },
  {
    key: 'complete',
    label: 'Review',
    detail: 'Inspect result',
    icon: Check,
  },
] as const

const stageIndex: Record<WorkflowStage, number> = {
  upload: 0,
  ready: 1,
  analyzing: 2,
  complete: 3,
}

function WorkflowProgress({
  stage,
}: WorkflowProgressProps) {
  const currentIndex = stageIndex[stage]

  return (
    <section
      aria-label="Analysis workflow"
      className="rounded-3xl border border-slate-800 bg-slate-900/60 p-4 sm:p-5"
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {steps.map((step, index) => {
          const Icon =
            stage === 'analyzing'
            && step.key === 'analyzing'
              ? LoaderCircle
              : step.icon

          const isCurrent = index === currentIndex
          const isComplete = index < currentIndex

          return (
            <div
              key={step.key}
              aria-current={
                isCurrent ? 'step' : undefined
              }
              className={
                isCurrent
                  ? 'rounded-2xl border border-cyan-500/30 bg-cyan-500/[0.08] p-4'
                  : isComplete
                    ? 'rounded-2xl border border-emerald-500/20 bg-emerald-500/[0.05] p-4'
                    : 'rounded-2xl border border-slate-800 bg-slate-950/30 p-4'
              }
            >
              <div className="flex items-center gap-3">
                <div
                  className={
                    isCurrent
                      ? 'rounded-xl bg-cyan-500/10 p-2 text-cyan-300'
                      : isComplete
                        ? 'rounded-xl bg-emerald-500/10 p-2 text-emerald-300'
                        : 'rounded-xl bg-slate-900 p-2 text-slate-600'
                  }
                >
                  <Icon
                    className={
                      stage === 'analyzing'
                      && step.key === 'analyzing'
                        ? 'size-4 animate-spin'
                        : 'size-4'
                    }
                  />
                </div>

                <div>
                  <p
                    className={
                      isCurrent
                        ? 'text-sm font-semibold text-slate-100'
                        : isComplete
                          ? 'text-sm font-semibold text-emerald-200'
                          : 'text-sm font-semibold text-slate-500'
                    }
                  >
                    {step.label}
                  </p>

                  <p className="mt-0.5 text-xs text-slate-600">
                    {step.detail}
                  </p>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

export type { WorkflowStage }
export default WorkflowProgress
