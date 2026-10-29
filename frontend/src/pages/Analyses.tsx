import { useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Search, Upload } from 'lucide-react'
import { useAnalyses } from '@/hooks/useAnalyses'
import StatusBadge from '@/components/analysis/StatusBadge'
import type { AnalysisMeta, AnalysisStatus } from '@/types'
import { fmt, fmtDate } from '@/utils/format'

export function AnalysisTable({ items, compact = false }: { items: AnalysisMeta[]; compact?: boolean }) {
  const nav = useNavigate()
  return (
    <div className="table-wrap">
      <table className="table table-click">
        <thead>
          <tr>
            <th>Sample</th>
            {!compact && <th>Patient</th>}
            <th>Uploaded</th>
            <th>Status</th>
            <th className="num">Conc. (M/mL)</th>
            <th className="num">Motility</th>
            <th className="num">Progressive</th>
            {!compact && <th className="num">Sperm followed</th>}
          </tr>
        </thead>
        <tbody>
          {items.map(a => {
            const h = a.headline
            return (
              <tr key={a.id} onClick={() => nav(`/analyses/${a.id}`)}>
                <td><Link to={`/analyses/${a.id}`} className="strong" onClick={e => e.stopPropagation()}>{a.sample_id}</Link>
                  {!compact && <div className="muted small">{a.filename}</div>}</td>
                {!compact && <td>{a.patient_ref ?? <span className="muted">—</span>}</td>}
                <td className="nowrap">{fmtDate(a.created_at)}</td>
                <td>
                  <StatusBadge status={a.status} />
                  {a.status === 'running' && <div className="muted small">{Math.round(a.progress * 100)}%</div>}
                  {h?.flags.includes('warning') && <div className="small warn-text">Check warnings</div>}
                </td>
                <td className="num">{fmt(h?.concentration)}</td>
                <td className="num">{h?.total_motility != null ? `${fmt(h.total_motility, 0)}%` : '—'}</td>
                <td className="num">{h?.progressive != null ? `${fmt(h.progressive, 0)}%` : '—'}</td>
                {!compact && <td className="num">{h?.n_sperm ?? '—'}</td>}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function Analyses() {
  const { items, error } = useAnalyses()
  const [q, setQ] = useState('')
  const [status, setStatus] = useState<AnalysisStatus | 'all'>('all')

  const shown = useMemo(() => (items ?? []).filter(a =>
    (status === 'all' || a.status === status) &&
    (!q || `${a.sample_id} ${a.patient_ref ?? ''} ${a.filename}`.toLowerCase().includes(q.toLowerCase()))), [items, q, status])

  return (
    <div className="page fade-in">
      <div className="page-header row-between">
        <div>
          <h1 className="page-title">Analyses</h1>
          <p className="page-subtitle">Every uploaded video and its result. Click a row to open the report.</p>
        </div>
        <Link to="/new" className="btn btn-primary"><Upload size={16} /> New analysis</Link>
      </div>

      <div className="toolbar">
        <label className="search">
          <Search size={16} />
          <input id="search" className="form-input" placeholder="Search sample, patient or file" value={q}
            onChange={e => setQ(e.target.value)} aria-label="Search analyses" />
        </label>
        <select id="status-filter" className="form-select" value={status} aria-label="Filter by status"
          onChange={e => setStatus(e.target.value as AnalysisStatus | 'all')}>
          <option value="all">All statuses</option>
          <option value="done">Complete</option>
          <option value="running">Analysing</option>
          <option value="queued">Queued</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      {error && <div className="alert alert-error">{error}</div>}
      {!items ? <div className="center"><span className="spinner" /></div>
        : items.length === 0 ? (
          <div className="card empty">
            <p className="strong">No analyses yet</p>
            <p className="muted">Upload a video recorded by the analyzer to get the first report.</p>
            <Link to="/new" className="btn btn-primary">Upload a video</Link>
          </div>
        ) : shown.length === 0 ? <div className="card empty"><p className="muted">No analyses match this search.</p></div>
          : <div className="card flush"><AnalysisTable items={shown} /></div>}
    </div>
  )
}
