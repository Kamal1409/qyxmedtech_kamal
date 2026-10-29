import { useCallback, useEffect, useRef, useState } from 'react'
import { api, errorMessage } from '@/services/api'
import type { AnalysisDetail, AnalysisMeta } from '@/types'

const ACTIVE = new Set(['queued', 'running'])

/** All analyses; refreshes every few seconds while any is queued or running. */
export function useAnalyses() {
  const [items, setItems] = useState<AnalysisMeta[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setItems(await api.list())
      setError(null)
    } catch (e) {
      setError(errorMessage(e))
    }
  }, [])

  useEffect(() => { load() }, [load])

  const active = items?.some(a => ACTIVE.has(a.status))
  useEffect(() => {
    if (!active) return
    const t = setInterval(load, 3000)
    return () => clearInterval(t)
  }, [active, load])

  return { items, error, reload: load }
}

/** One analysis; polls while it is queued or running. */
export function useAnalysis(id: string | undefined) {
  const [data, setData] = useState<AnalysisDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout>>()

  const load = useCallback(async () => {
    if (!id) return
    try {
      const d = await api.get(id)
      setData(d)
      setError(null)
      if (ACTIVE.has(d.status)) timer.current = setTimeout(load, 1500)
    } catch (e) {
      setError(errorMessage(e))
    }
  }, [id])

  useEffect(() => {
    setData(null)
    load()
    return () => clearTimeout(timer.current)
  }, [load])

  return { data, error }
}
