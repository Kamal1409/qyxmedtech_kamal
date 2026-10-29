import type { Distribution } from '@/types'

/** Horizontal percentage bars per bin, top bin first, as on the lab report. */
export default function DistributionChart({ title, d }: { title: string; d: Distribution }) {
  const rows = d.labels.map((l, i) => ({ label: l, pct: d.percent[i] })).reverse()
  return (
    <figure className="dist">
      <figcaption>{title} <span className="muted">({d.unit})</span></figcaption>
      {d.n === 0 ? <p className="muted small">No moving sperm to measure.</p> : (
        <div className="dist-rows">
          {rows.map(r => (
            <div key={r.label} className="dist-row">
              <span className="dist-label">{r.label}</span>
              <span className="dist-track"><span className="dist-fill" style={{ width: `${r.pct}%` }} /></span>
              <span className="dist-val">{r.pct.toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </figure>
  )
}
