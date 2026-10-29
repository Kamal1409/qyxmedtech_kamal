import type { WhoRow } from '@/types'
import { fmt } from '@/utils/format'

const STATUS: Record<WhoRow['status'], [string, string]> = {
  normal: ['Within reference', 'success'],
  low: ['Below reference', 'warning'],
  high: ['Above reference', 'warning'],
  na: ['Not measured', 'muted'],
}

export default function WhoTable({ rows }: { rows: WhoRow[] }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr><th>Parameter</th><th className="num">Result</th><th className="num">WHO 6th ed. reference</th><th>Status</th></tr>
        </thead>
        <tbody>
          {rows.map(r => {
            const [label, tone] = STATUS[r.status]
            return (
              <tr key={r.key}>
                <td>{r.label}</td>
                <td className="num strong">{r.value === null ? '—' : `${fmt(r.value, r.unit === '%' ? 0 : 1)} ${r.unit}`}</td>
                <td className="num muted">{r.reference}</td>
                <td>{tone === 'muted' ? <span className="muted small">{label}</span>
                  : <span className={`badge badge-${tone}`}>{label}</span>}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
