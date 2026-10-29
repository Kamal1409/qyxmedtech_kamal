import axios, { AxiosError } from 'axios'
import type { AnalysisDetail, AnalysisMeta, ModelInfo } from '@/types'

const http = axios.create({ baseURL: '/api/v1' })

/** The server's own message when it sent one, otherwise a plain description. */
export function errorMessage(err: unknown): string {
  const e = err as AxiosError<{ detail?: unknown }>
  const detail = e.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    return detail.map((d: { loc?: string[]; msg?: string }) => `${d.loc?.slice(-1)[0] ?? 'field'}: ${d.msg}`).join('; ')
  }
  if (e.code === 'ERR_NETWORK') return 'Cannot reach the analysis server. Check that it is running.'
  return e.message || 'Something went wrong.'
}

export interface NewAnalysis {
  file: File
  sample_id: string
  patient_ref?: string
  volume_ml?: string
  notes?: string
}

export const api = {
  list: () => http.get<AnalysisMeta[]>('/analyses').then(r => r.data),
  get: (id: string) => http.get<AnalysisDetail>(`/analyses/${id}`).then(r => r.data),
  remove: (id: string) => http.delete(`/analyses/${id}`),
  model: () => http.get<ModelInfo>('/model').then(r => r.data),
  health: () => http.get<{ status: string; checks: Record<string, boolean> }>('/health/ready').then(r => r.data),
  create: (a: NewAnalysis, onProgress?: (fraction: number) => void) => {
    const form = new FormData()
    form.append('file', a.file)
    form.append('sample_id', a.sample_id)
    if (a.patient_ref) form.append('patient_ref', a.patient_ref)
    if (a.volume_ml) form.append('volume_ml', a.volume_ml)
    if (a.notes) form.append('notes', a.notes)
    return http.post<AnalysisMeta>('/analyses', form, {
      onUploadProgress: e => onProgress?.(e.total ? e.loaded / e.total : 0),
    }).then(r => r.data)
  },
  overlayUrl: (id: string) => `/api/v1/analyses/${id}/overlay`,
  videoUrl: (id: string) => `/api/v1/analyses/${id}/video`,
  posterUrl: (id: string) => `/api/v1/analyses/${id}/poster`,
}
