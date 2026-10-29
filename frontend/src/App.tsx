import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import Dashboard from '@/pages/Dashboard'
import NewAnalysis from '@/pages/NewAnalysis'
import Analyses from '@/pages/Analyses'
import AnalysisReport from '@/pages/AnalysisReport'

// Only this page uses the charting library; load it on demand.
const Accuracy = lazy(() => import('@/pages/Accuracy'))

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="new" element={<NewAnalysis />} />
          <Route path="analyses" element={<Analyses />} />
          <Route path="analyses/:id" element={<AnalysisReport />} />
          <Route path="accuracy" element={
            <Suspense fallback={<div className="center"><span className="spinner" /></div>}><Accuracy /></Suspense>
          } />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
