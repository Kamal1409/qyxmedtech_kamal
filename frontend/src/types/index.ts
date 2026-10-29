// Types for the QYX MedTech CASA frontend (mirror backend/app/models/analysis.py)

export type AnalysisStatus = 'queued' | 'running' | 'done' | 'failed'

export interface Headline {
  concentration: number | null
  total_motility: number | null
  progressive: number | null
  n_sperm: number | null
  flags: string[]
}

export interface AnalysisMeta {
  id: string
  sample_id: string
  patient_ref: string | null
  volume_ml: number | null
  notes: string | null
  filename: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  status: AnalysisStatus
  stage: string | null
  stage_label: string | null
  progress: number
  error: string | null
  queue_position: number | null
  headline: Headline | null
}

export interface WhoRow {
  key: string
  label: string
  value: number | null
  unit: string
  reference: string
  status: 'normal' | 'low' | 'high' | 'na'
}

export interface Distribution {
  labels: string[]
  unit: string
  n: number
  percent: number[]
}

export interface Warning {
  level: 'info' | 'warning' | 'error'
  text: string
}

export interface Motility {
  total_motility_pct: number
  progressive_pct: number
  rapid_progressive_pct: number
  slow_progressive_pct: number
  non_progressive_pct: number
  immotile_pct: number
}

export interface AnalysisResult {
  concentration_m_per_ml: number | null
  total_sperm_number_m: number | null
  msc_m_per_ml: number | null
  pmsc_m_per_ml: number | null
  motility: Motility
  counts: { PR?: number; NP?: number; IM?: number }
  n_sperm_analyzed: number
  motility_margin_pp: number | null
  kinematics: Record<string, { mean: number | null; median: number | null }>
  who: WhoRow[]
  distributions: Record<'VAP' | 'VSL' | 'STR' | 'ALH', Distribution>
  quality: {
    detection_tier: string | null
    live_frames: number | null
    total_frames: number | null
    live_seconds: number | null
    concentration_cv_pct: number | null
    warnings: Warning[]
  }
  video: { fps: number | null; resolution: [number, number] | null; frames: number | null }
  settings: Record<string, number | string | null>
}

export interface AnalysisDetail extends AnalysisMeta {
  result: AnalysisResult | null
}

export interface MetricAccuracy {
  mae: number
  bias: number
  icc: number
  pearson_r: number
  n: number
  per_video: { video: string; predicted: number; actual: number; diff: number }[]
}

export interface ModelInfo {
  settings: Record<string, number | string | null>
  validation: Record<string, MetricAccuracy> | null
  baseline: Record<string, MetricAccuracy> | null
  n_videos: number | null
  test_videos: string[]
  excluded_videos: string[]
  changes: string[]
}
