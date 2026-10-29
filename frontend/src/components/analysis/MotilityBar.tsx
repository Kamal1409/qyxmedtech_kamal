import type { AnalysisResult } from '@/types'
import { fmt } from '@/utils/format'

/** One bar split into rapid PR / slow PR / NP / IM, like the WHO categories on the lab report. */
export default function MotilityBar({ r }: { r: AnalysisResult }) {
  const m = r.motility
  const parts = [
    { key: 'rapid', label: 'Rapid progressive', value: m.rapid_progressive_pct, cls: 'seg-rapid' },
    { key: 'slow', label: 'Slow progressive', value: m.slow_progressive_pct, cls: 'seg-slow' },
    { key: 'np', label: 'Non-progressive', value: m.non_progressive_pct, cls: 'seg-np' },
    { key: 'im', label: 'Immotile', value: m.immotile_pct, cls: 'seg-im' },
  ]
  return (
    <div className="motility">
      <div className="motility-bar" role="img"
        aria-label={parts.map(p => `${p.label} ${fmt(p.value, 0)}%`).join(', ')}>
        {parts.map(p => p.value > 0 && (
          <div key={p.key} className={`motility-seg ${p.cls}`} style={{ width: `${p.value}%` }}
            title={`${p.label}: ${fmt(p.value)}%`}>
            {p.value >= 8 ? `${fmt(p.value, 0)}%` : ''}
          </div>
        ))}
      </div>
      <ul className="legend">
        {parts.map(p => (
          <li key={p.key}><span className={`legend-swatch ${p.cls}`} />{p.label} <b>{fmt(p.value)}%</b></li>
        ))}
      </ul>
    </div>
  )
}
