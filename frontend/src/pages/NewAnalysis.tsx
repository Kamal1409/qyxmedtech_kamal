import { useRef, useState, type DragEvent, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileVideo, UploadCloud, X } from 'lucide-react'
import { api, errorMessage } from '@/services/api'

const ACCEPT = ['.mp4', '.avi', '.mov', '.mkv']
const MAX_MB = 500

export default function NewAnalysis() {
  const nav = useNavigate()
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [sampleId, setSampleId] = useState('')
  const [patientRef, setPatientRef] = useState('')
  const [volume, setVolume] = useState('')
  const [notes, setNotes] = useState('')
  const [drag, setDrag] = useState(false)
  const [upload, setUpload] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  const pick = (f: File | undefined) => {
    if (!f) return
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPT.includes(ext)) return setError(`"${f.name}" is not a video. Choose an ${ACCEPT.join(', ')} file.`)
    if (f.size > MAX_MB * 1024 * 1024) return setError(`The video is larger than ${MAX_MB} MB.`)
    setError(null)
    setFile(f)
    if (!sampleId) setSampleId(f.name.replace(/\.[^.]+$/, ''))
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDrag(false)
    pick(e.dataTransfer.files[0])
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!file) return setError('Choose a video to analyse.')
    if (!sampleId.trim()) return setError('Enter a sample ID.')
    const v = volume.trim()
    if (v && !(Number(v) > 0 && Number(v) <= 20)) return setError('Semen volume must be between 0 and 20 mL.')
    setError(null)
    setUpload(0)
    try {
      const a = await api.create({ file, sample_id: sampleId.trim(), patient_ref: patientRef.trim(), volume_ml: v, notes: notes.trim() },
        setUpload)
      nav(`/analyses/${a.id}`)
    } catch (err) {
      setError(errorMessage(err))
      setUpload(null)
    }
  }

  const busy = upload !== null
  return (
    <div className="page narrow fade-in">
      <div className="page-header">
        <h1 className="page-title">New analysis</h1>
        <p className="page-subtitle">Upload the video recorded by the analyzer. We measure concentration, motility and
          swimming speed, and compare them with WHO reference values.</p>
      </div>

      <form className="card form-card" onSubmit={submit} noValidate>
        <div
          className={`dropzone${drag ? ' drag' : ''}${file ? ' has-file' : ''}`}
          onDragOver={e => { e.preventDefault(); setDrag(true) }}
          onDragLeave={() => setDrag(false)}
          onDrop={onDrop}
          onClick={() => !busy && input.current?.click()}
          onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && !busy && input.current?.click()}
          role="button" tabIndex={0} aria-label="Choose a video file"
        >
          <input ref={input} id="video-file" type="file" accept={ACCEPT.join(',')} hidden
            onChange={e => pick(e.target.files?.[0])} />
          {file ? (
            <div className="file-chip">
              <FileVideo size={22} />
              <div>
                <div className="strong">{file.name}</div>
                <div className="muted small">{(file.size / 1024 / 1024).toFixed(1)} MB</div>
              </div>
              {!busy && (
                <button type="button" className="btn btn-ghost icon-btn" aria-label="Remove file"
                  onClick={e => { e.stopPropagation(); setFile(null) }}><X size={16} /></button>
              )}
            </div>
          ) : (
            <>
              <UploadCloud size={36} className="dz-icon" />
              <div className="strong">Drop a video here or click to choose</div>
              <div className="muted small">{ACCEPT.join(', ')} up to {MAX_MB} MB</div>
            </>
          )}
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label" htmlFor="sample-id">Sample ID *</label>
            <input id="sample-id" className="form-input" value={sampleId} maxLength={64} required
              onChange={e => setSampleId(e.target.value)} placeholder="e.g. S-2026-0412" disabled={busy} />
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="patient-ref">Patient reference</label>
            <input id="patient-ref" className="form-input" value={patientRef} maxLength={64}
              onChange={e => setPatientRef(e.target.value)} placeholder="Optional, avoid full names" disabled={busy} />
          </div>
          <div className="form-group">
            <label className="form-label" htmlFor="volume">Semen volume (mL)</label>
            <input id="volume" className="form-input" inputMode="decimal" value={volume}
              onChange={e => setVolume(e.target.value)} placeholder="Optional, e.g. 2.5" disabled={busy} />
            <span className="muted small">Needed for total sperm number.</span>
          </div>
          <div className="form-group span-2">
            <label className="form-label" htmlFor="notes">Notes</label>
            <textarea id="notes" className="form-textarea" value={notes} maxLength={1000}
              onChange={e => setNotes(e.target.value)} placeholder="Optional" disabled={busy} />
          </div>
        </div>

        {error && <div className="alert alert-error" role="alert">{error}</div>}

        <div className="form-actions">
          {busy && (
            <div className="upload-progress">
              <div className="confidence-bar"><div className="confidence-fill" style={{ width: `${Math.round((upload ?? 0) * 100)}%` }} /></div>
              <span className="muted small">{upload! < 1 ? `Uploading ${Math.round(upload! * 100)}%` : 'Starting analysis…'}</span>
            </div>
          )}
          <button id="start-analysis" className="btn btn-primary" type="submit" disabled={busy}>
            {busy ? <span className="spinner spinner-xs" /> : <UploadCloud size={16} />}
            Analyse video
          </button>
        </div>
      </form>
    </div>
  )
}
