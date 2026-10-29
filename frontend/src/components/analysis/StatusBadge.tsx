import type { AnalysisStatus } from '@/types'

const LABEL: Record<AnalysisStatus, string> = { queued: 'Queued', running: 'Analysing', done: 'Complete', failed: 'Failed' }
const TONE: Record<AnalysisStatus, string> = { queued: 'info', running: 'info', done: 'success', failed: 'danger' }

export default function StatusBadge({ status }: { status: AnalysisStatus }) {
  return (
    <span className={`badge badge-${TONE[status]}`}>
      {status === 'running' && <span className="spinner spinner-xs" aria-hidden />}
      {LABEL[status]}
    </span>
  )
}
