import { Link } from 'react-router-dom'
import { Upload } from 'lucide-react'
import { useAnalyses } from '@/hooks/useAnalyses'
import { AnalysisTable } from './Analyses'
import { fmt } from '@/utils/format'

export default function Dashboard() {
  const { items, error } = useAnalyses()
  const all = items ?? []
  const done = all.filter(a => a.status === 'done')
  const active = all.filter(a => a.status === 'queued' || a.status === 'running')
  const avg = (f: (h: NonNullable<(typeof done)[0]['headline']>) => number | null) => {
    const v = done.map(a => a.headline && f(a.headline)).filter((x): x is number => x !== null && x !== undefined)
    return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null
  }
  const belowWho = done.filter(a => (a.headline?.total_motility ?? 100) < 42 || (a.headline?.concentration ?? 100) < 16).length

  return (
    <div className="page fade-in">
      <div className="page-header row-between">
        <div>
          <h1 className="page-title">Semen analysis</h1>
          <p className="page-subtitle">Computer-aided analysis of videos from the LensHooke analyzer: concentration,
            WHO motility classes and swimming speed.</p>
        </div>
        <Link to="/new" className="btn btn-primary"><Upload size={16} /> New analysis</Link>
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      <div className="stats-grid">
        <div className="card stat"><div className="stat-label">Analyses completed</div>
          <div className="stat-value">{items ? done.length : '—'}</div>
          <div className="stat-sub">{active.length ? `${active.length} in progress` : 'None in progress'}</div></div>
        <div className="card stat"><div className="stat-label">Below a WHO reference</div>
          <div className="stat-value">{items ? belowWho : '—'}</div>
          <div className="stat-sub">Concentration &lt; 16 M/mL or motility &lt; 42%</div></div>
        <div className="card stat"><div className="stat-label">Average concentration</div>
          <div className="stat-value">{fmt(avg(h => h.concentration))}<span className="stat-unit"> M/mL</span></div>
          <div className="stat-sub">Across completed analyses</div></div>
        <div className="card stat"><div className="stat-label">Average total motility</div>
          <div className="stat-value">{fmt(avg(h => h.total_motility), 0)}<span className="stat-unit"> %</span></div>
          <div className="stat-sub">Across completed analyses</div></div>
      </div>

      <div className="card flush">
        <div className="card-head row-between">
          <h2 className="section-title">Recent analyses</h2>
          <Link to="/analyses" className="small">View all</Link>
        </div>
        {!items ? <div className="center pad"><span className="spinner" /></div>
          : all.length === 0 ? (
            <div className="empty">
              <p className="strong">No analyses yet</p>
              <p className="muted">Upload a video to get the first report. It takes about a minute.</p>
              <Link to="/new" className="btn btn-primary">Upload a video</Link>
            </div>
          ) : <AnalysisTable items={all.slice(0, 8)} compact />}
      </div>
    </div>
  )
}
