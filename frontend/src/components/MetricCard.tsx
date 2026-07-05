import type { LucideIcon } from 'lucide-react'

interface MetricCardProps {
  label: string
  value: string
  detail: string
  icon: LucideIcon
  tone?: 'cyan' | 'emerald' | 'amber' | 'violet'
}

const toneClasses = {
  cyan: 'border-cyan-500/20 bg-cyan-500/[0.06] text-cyan-300',
  emerald:
    'border-emerald-500/20 bg-emerald-500/[0.06] text-emerald-300',
  amber:
    'border-amber-500/20 bg-amber-500/[0.06] text-amber-300',
  violet:
    'border-violet-500/20 bg-violet-500/[0.06] text-violet-300',
}

function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
  tone = 'cyan',
}: MetricCardProps) {
  return (
    <article
      className={`rounded-2xl border p-5 ${toneClasses[tone]}`}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {label}
          </p>

          <p className="mt-3 text-2xl font-semibold tracking-tight text-slate-100">
            {value}
          </p>
        </div>

        <div className="rounded-xl bg-slate-950/50 p-2.5">
          <Icon className="size-5" />
        </div>
      </div>

      <p className="mt-3 text-xs leading-5 text-slate-500">
        {detail}
      </p>
    </article>
  )
}

export default MetricCard
