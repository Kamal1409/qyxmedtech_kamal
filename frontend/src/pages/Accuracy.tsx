import { useEffect, useState } from 'react'
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { api, errorMessage } from '@/services/api'
import type { MetricAccuracy, ModelInfo } from '@/types'
import { fmt } from '@/utils/format'

const ROWS: [string, string, string][] = [
  ['concentration', 'Sperm concentration', 'M/mL'],
  ['total_motility', 'Total motility', 'pp'],
  ['progressive', 'Progressive motility', 'pp'],
  ['rapid_progressive', 'Rapid progressive', 'pp'],
  ['slow_progressive', 'Slow progressive', 'pp'],
  ['non_progressive', 'Non-progressive', 'pp'],
  ['VAP_mean', 'Swimming speed (VAP)', 'µm/s'],
  ['VSL_mean', 'Straight-line speed (VSL)', 'µm/s'],
  ['VCL_mean', 'Curvilinear speed (VCL)', 'µm/s'],
]

function agreement(icc: number): [string, string] {
  if (icc >= 0.9) return ['Excellent', 'success']
  if (icc >= 0.75) return ['Good', 'success']
  if (icc >= 0.5) return ['Moderate', 'warning']
  return ['Poor', 'danger']
}

function LabVsOurs({ title, unit, m, test }: { title: string; unit: string; m: MetricAccuracy; test: Set<string> }) {
  const pts = m.per_video.map(p => ({ lab: p.actual, ours: p.predicted, video: p.video, held: test.has(p.video) }))
  const top = Math.max(...pts.flatMap(p => [p.lab, p.ours]))
  const step = top > 150 ? 50 : top > 60 ? 25 : 10
  const max = Math.ceil(top / step) * step
  const ticks = Array.from({ length: max / step + 1 }, (_, i) => i * step)
  const tick = { fill: 'var(--color-text-secondary)', fontSize: 12 }
  return (
    <figure className="card section chart-card">
      <figcaption className="section-title">{title}: lab vs ours ({unit})</figcaption>
      <div className="chart">
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart margin={{ top: 8, right: 12, bottom: 24, left: 4 }}>
            <CartesianGrid stroke="var(--color-border)" />
            <XAxis type="number" dataKey="lab" domain={[0, max]} ticks={ticks} tick={tick} stroke="var(--color-border)"
              label={{ value: 'Lab analyzer', position: 'insideBottom', offset: -12, fill: 'var(--color-text-secondary)', fontSize: 12 }} />
            <YAxis type="number" dataKey="ours" domain={[0, max]} ticks={ticks} tick={tick} stroke="var(--color-border)" width={60}
              label={{ value: 'This software', angle: -90, position: 'insideLeft', offset: 0, dy: 40, fill: 'var(--color-text-secondary)', fontSize: 12 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="var(--color-text-muted)" strokeDasharray="4 4" />
            <Tooltip cursor={false} contentStyle={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', borderRadius: 8 }}
              formatter={(v: number) => fmt(v)} labelFormatter={() => ''}
              content={({ payload }) => {
                const p = payload?.[0]?.payload as (typeof pts)[0] | undefined
                return p ? <div className="tooltip">Sample {p.video}{p.held ? ' (held out)' : ''}<br />Lab {fmt(p.lab)} · ours {fmt(p.ours)}</div> : null
              }} />
            <Scatter data={pts.filter(p => !p.held)} fill="#3b9eff" name="Tuning samples" isAnimationActive={false} />
            <Scatter data={pts.filter(p => p.held)} fill="#fbbf24" name="Held-out samples" isAnimationActive={false} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <p className="muted small"><span className="dot dot-blue" /> used for tuning <span className="dot dot-amber" /> held out
        (never used for tuning). Points on the dashed line match the lab exactly.</p>
    </figure>
  )
}

export default function Accuracy() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { api.model().then(setInfo).catch(e => setError(errorMessage(e))) }, [])

  if (error) return <div className="page"><div className="alert alert-error">{error}</div></div>
  if (!info) return <div className="page center"><span className="spinner" /></div>
  const v = info.validation, b = info.baseline
  const test = new Set(info.test_videos)
  const s = info.settings

  return (
    <div className="page fade-in">
      <div className="page-header">
        <h1 className="page-title">Model accuracy</h1>
        <p className="page-subtitle">How closely this software matches the LensHooke lab analyzer's reports
          {info.n_videos ? ` on ${info.n_videos} samples` : ''}. Samples {info.excluded_videos.join(' and ') || '—'} are
          excluded until they are hand-annotated.</p>
      </div>

      {!v ? <div className="card"><p className="muted">No validation report found. Run <code>python -m src.validate</code> in training/.</p></div> : (
        <>
          <section className="card section flush">
            <div className="card-head"><h2 className="section-title">Average difference from the lab</h2></div>
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Measurement</th><th className="num">Before update</th><th className="num">Now</th>
                  <th className="num">Bias now</th><th>Agreement</th></tr></thead>
                <tbody>
                  {ROWS.filter(([k]) => v[k]).map(([k, label, unit]) => {
                    const [ag, tone] = agreement(v[k].icc)
                    const better = b?.[k] && v[k].mae < b[k].mae
                    return (
                      <tr key={k}>
                        <td>{label}</td>
                        <td className="num muted">{b?.[k] ? `${fmt(b[k].mae)} ${unit}` : '—'}</td>
                        <td className={`num strong ${b?.[k] ? (better ? 'good-text' : 'warn-text') : ''}`}>{fmt(v[k].mae)} {unit}</td>
                        <td className="num">{v[k].bias > 0 ? '+' : ''}{fmt(v[k].bias)}</td>
                        <td><span className={`badge badge-${tone}`}>{ag}</span> <span className="muted small">ICC {fmt(v[k].icc, 2)}</span></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="muted small pad">"pp" = percentage points. Bias: positive means we read higher than the lab on average.
              The lab analyzer itself varies about 8–10% between repeat runs of one sample.</p>
          </section>

          <div className="two-col">
            {v.concentration && <LabVsOurs title="Concentration" unit="M/mL" m={v.concentration} test={test} />}
            {v.total_motility && <LabVsOurs title="Total motility" unit="%" m={v.total_motility} test={test} />}
          </div>
        </>
      )}

      <div className="two-col">
        <section className="card section">
          <h2 className="section-title">Settings in use</h2>
          <dl className="kv">
            <dt>Head detector</dt><dd>{String(s.detector)} (confidence ≥ {String(s.detector_confidence)})</dd>
            <dt>Image scale</dt><dd>{String(s.pixel_to_micron)} µm per pixel (stage micrometer)</dd>
            <dt>Capture rate</dt><dd>{String(s.capture_fps)} fps (estimated)</dd>
            <dt>Count calibration</dt><dd>×{String(s.concentration_calibration)} (estimated), chamber {String(s.chamber_depth_um)} µm</dd>
            <dt>Tracking radius</dt><dd>{String(s.tracker_search_radius_px)} px per frame</dd>
            <dt>Rapid progressive</dt><dd>VAP ≥ {String(s.rapid_min_vap)} µm/s</dd>
          </dl>
        </section>
        <section className="card section">
          <h2 className="section-title">What was changed</h2>
          {info.changes.length ? <ul className="bullets">{info.changes.map((c, i) => <li key={i}>{c}</li>)}</ul>
            : <p className="muted">No change notes found.</p>}
        </section>
      </div>
    </div>
  )
}
