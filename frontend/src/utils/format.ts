export function fmt(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(digits)
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString(undefined, {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

export function duration(start: string | null, end: string | null): string {
  if (!start || !end) return '—'
  const s = Math.round((new Date(end).getTime() - new Date(start).getTime()) / 1000)
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`
}
