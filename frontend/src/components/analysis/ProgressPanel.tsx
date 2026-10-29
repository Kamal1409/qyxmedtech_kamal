import { Check } from 'lucide-react'
import type { AnalysisMeta } from '@/types'

// Must match backend/app/services/casa.py STAGES
const STAGES: [string, string][] = [
  ['frames', 'Extracting frames'],
  ['heads', 'Detecting sperm heads'],
  ['tracks', 'Following each sperm'],
  ['kinematic', 'Measuring speed and path'],
  ['classify', 'Classifying motility (WHO)'],
  ['report', 'Counting and building report'],
]

export default function ProgressPanel({ a }: { a: AnalysisMeta }) {
  const current = STAGES.findIndex(([k]) => k === a.stage)
  const pct = Math.round(a.progress * 100)
  return (
    <div className="card progress-card">
      <div className="progress-head">
        <div>
          <div className="eyebrow">{a.status === 'queued' ? 'Waiting' : 'Analysing'}</div>
          <h2>{a.status === 'queued'
            ? a.queue_position ? `Queued, position ${a.queue_position}` : 'Starting soon'
            : a.stage_label}</h2>
        </div>
        <div className="progress-pct">{pct}%</div>
      </div>
      <div className="confidence-bar" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="confidence-fill" style={{ width: `${pct}%` }} />
      </div>
      <ol className="stage-list">
        {STAGES.map(([key, label], i) => {
          const state = a.status === 'queued' || i > current ? 'todo' : i < current ? 'done' : 'now'
          return (
            <li key={key} className={`stage stage-${state}`}>
              <span className="stage-mark">
                {state === 'done' ? <Check size={14} /> : state === 'now' ? <span className="spinner spinner-xs" /> : i + 1}
              </span>
              {label}
            </li>
          )
        })}
      </ol>
      <p className="muted small">Usually under a minute per video with the GPU. You can leave this page; the analysis keeps running.</p>
    </div>
  )
}
