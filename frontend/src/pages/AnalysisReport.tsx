import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { AlertTriangle, ArrowLeft, Info, Printer, Trash2, XCircle } from 'lucide-react'
import { useAnalysis } from '@/hooks/useAnalyses'
import { api, errorMessage } from '@/services/api'
import StatusBadge from '@/components/analysis/StatusBadge'
import ProgressPanel from '@/components/analysis/ProgressPanel'
import MotilityBar from '@/components/analysis/MotilityBar'
import DistributionChart from '@/components/analysis/DistributionChart'
import WhoTable from '@/components/analysis/WhoTable'
import type { AnalysisDetail, AnalysisResult } from '@/types'
import { duration, fmt, fmtDate } from '@/utils/format'

const KIN_ROWS: [string, string, string, number][] = [
  ['VCL', 'Curvilinear velocity', 'µm/s', 1],
  ['VSL', 'Straight-line velocity', 'µm/s', 1],
  ['VAP', 'Average path velocity', 'µm/s', 1],
  ['LIN', 'Linearity', '%', 0],
  ['STR', 'Straightness', '%', 0],
  ['WOB', 'Wobble', '%', 0],
  ['ALH', 'Lateral head displacement', 'µm', 2],
  ['BCF', 'Beat-cross frequency', 'Hz', 1],
]

function KeyStat({ label, value, unit, sub }: { label: string; value: string; unit?: string; sub?: string }) {
  return (
    <div className="card stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}{unit && <span className="stat-unit"> {unit}</span>}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  )
}

function Report({ a, r }: { a: AnalysisDetail; r: AnalysisResult }) {
  const m = r.motility
  const warnIcon = { error: XCircle, warning: AlertTriangle, info: Info }
  return (
    <>
      {r.quality.warnings.length > 0 && (
        <div className="warnings">
          {r.quality.warnings.map((w, i) => {
            const Icon = warnIcon[w.level]
            return <div key={i} className={`alert alert-${w.level}`}><Icon size={16} /><span>{w.text}</span></div>
          })}
        </div>
      )}

      <div className="stats-grid">
        <KeyStat label="Concentration" value={fmt(r.concentration_m_per_ml)} unit="M/mL"
          sub={r.total_sperm_number_m !== null ? `${fmt(r.total_sperm_number_m)} M in ${a.volume_ml} mL` : 'Add volume for total count'} />
        <KeyStat label="Total motility" value={fmt(m.total_motility_pct, 0)} unit="%"
          sub={r.motility_margin_pp !== null ? `±${fmt(r.motility_margin_pp, 0)} pp from ${r.n_sperm_analyzed} sperm` : undefined} />
        <KeyStat label="Progressive" value={fmt(m.progressive_pct, 0)} unit="%"
          sub={`Rapid ${fmt(m.rapid_progressive_pct, 0)}% · slow ${fmt(m.slow_progressive_pct, 0)}%`} />
        <KeyStat label="Swimming speed (VAP)" value={fmt(r.kinematics.VAP?.mean)} unit="µm/s" sub="Mean of moving sperm" />
      </div>

      <section className="card section">
        <h2 className="section-title">Test result</h2>
        <WhoTable rows={r.who} />
      </section>

      <section className="card section">
        <h2 className="section-title">Motility</h2>
        <MotilityBar r={r} />
        <p className="muted small">
          {r.counts.PR ?? 0} progressive, {r.counts.NP ?? 0} non-progressive and {r.counts.IM ?? 0} immotile sperm were
          followed. MSC {fmt(r.msc_m_per_ml)} M/mL, PMSC {fmt(r.pmsc_m_per_ml)} M/mL.
        </p>
      </section>

      <div className="two-col">
        <section className="card section">
          <h2 className="section-title">Sperm paths</h2>
          <img className="overlay-img" src={api.overlayUrl(a.id)} alt="Tracked sperm paths drawn on a video frame" />
          <p className="muted small"><span className="dot dot-pr" /> progressive <span className="dot dot-np" /> non-progressive
            <span className="dot dot-im" /> immotile</p>
        </section>
        <section className="card section">
          <h2 className="section-title">Video</h2>
          <video className="overlay-img" src={api.videoUrl(a.id)} poster={api.posterUrl(a.id)}
            controls muted playsInline preload="metadata" />
          <p className="muted small">{a.filename}{r.video.fps ? ` · ${r.video.fps} fps` : ''}
            {r.video.resolution ? ` · ${r.video.resolution[0]}×${r.video.resolution[1]}` : ''}</p>
        </section>
      </div>

      <section className="card section">
        <h2 className="section-title">Movement of moving sperm</h2>
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Parameter</th><th /><th className="num">Mean</th><th className="num">Median</th></tr></thead>
            <tbody>
              {KIN_ROWS.map(([k, label, unit, d]) => (
                <tr key={k}><td className="strong">{k}</td><td className="muted">{label} ({unit})</td>
                  <td className="num">{fmt(r.kinematics[k]?.mean, d)}</td><td className="num">{fmt(r.kinematics[k]?.median, d)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="dist-grid">
          <DistributionChart title="VAP" d={r.distributions.VAP} />
          <DistributionChart title="VSL" d={r.distributions.VSL} />
          <DistributionChart title="STR" d={r.distributions.STR} />
          <DistributionChart title="ALH" d={r.distributions.ALH} />
        </div>
      </section>

      <section className="card section">
        <h2 className="section-title">How this was measured</h2>
        <dl className="kv">
          <dt>Usable video</dt><dd>{r.quality.live_frames ?? '—'} of {r.quality.total_frames ?? '—'} frames
            ({fmt(r.quality.live_seconds)} s), after removing dark gaps and frozen stills</dd>
          <dt>Sperm followed</dt><dd>{r.n_sperm_analyzed}</dd>
          <dt>Frame-to-frame count variation</dt><dd>{fmt(r.quality.concentration_cv_pct, 0)}%</dd>
          <dt>Analysis time</dt><dd>{duration(a.started_at, a.finished_at)}</dd>
          <dt>Settings</dt><dd>{String(r.settings.pixel_to_micron)} µm/px · capture {String(r.settings.capture_fps)} fps ·
            chamber {String(r.settings.chamber_depth_um)} µm · count calibration ×{String(r.settings.concentration_calibration)} ·
            detector {String(r.settings.detector)}</dd>
        </dl>
        <p className="muted small">Research use only. Capture rate and count calibration are estimated from 21 lab-verified
          samples because the analyzer's maker does not publish them. See <Link to="/accuracy">model accuracy</Link>.</p>
      </section>
    </>
  )
}

export default function AnalysisReport() {
  const { id } = useParams()
  const nav = useNavigate()
  const { data: a, error } = useAnalysis(id)
  const [confirm, setConfirm] = useState(false)
  const [delErr, setDelErr] = useState<string | null>(null)

  if (error) return (
    <div className="page"><div className="alert alert-error">{error}</div>
      <Link to="/analyses" className="btn btn-secondary">Back to analyses</Link></div>
  )
  if (!a) return <div className="page center"><span className="spinner" /></div>

  const remove = async () => {
    try { await api.remove(a.id); nav('/analyses') } catch (e) { setDelErr(errorMessage(e)); setConfirm(false) }
  }

  return (
    <div className="page fade-in report">
      <Link to="/analyses" className="back-link no-print"><ArrowLeft size={14} /> All analyses</Link>
      <div className="report-head">
        <div>
          <div className="eyebrow">Semen analysis report</div>
          <h1 className="page-title">Sample {a.sample_id}</h1>
          <p className="page-subtitle">
            {a.patient_ref && <>Patient {a.patient_ref} · </>}Uploaded {fmtDate(a.created_at)}
            {a.volume_ml && <> · Volume {a.volume_ml} mL</>}
          </p>
          {a.notes && <p className="muted small notes">{a.notes}</p>}
        </div>
        <div className="report-actions no-print">
          <StatusBadge status={a.status} />
          {a.status === 'done' && (
            <button className="btn btn-secondary" onClick={() => window.print()}><Printer size={16} /> Print</button>
          )}
          {a.status !== 'running' && (confirm ? (
            <span className="confirm">
              <span className="small">Delete this analysis?</span>
              <button className="btn btn-danger" onClick={remove}>Delete</button>
              <button className="btn btn-ghost" onClick={() => setConfirm(false)}>Cancel</button>
            </span>
          ) : (
            <button className="btn btn-ghost" onClick={() => setConfirm(true)} aria-label="Delete analysis"><Trash2 size={16} /></button>
          ))}
        </div>
      </div>
      {delErr && <div className="alert alert-error">{delErr}</div>}

      {(a.status === 'queued' || a.status === 'running') && <ProgressPanel a={a} />}
      {a.status === 'failed' && (
        <div className="card section">
          <div className="alert alert-error"><XCircle size={16} /><span>{a.error ?? 'The analysis failed.'}</span></div>
          <Link to="/new" className="btn btn-primary">Upload again</Link>
        </div>
      )}
      {a.status === 'done' && a.result && <Report a={a} r={a.result} />}
    </div>
  )
}
