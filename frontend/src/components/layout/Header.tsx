import { useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { api } from '@/services/api'
import { Brand, navItems } from './Sidebar'

function title(pathname: string): string {
  if (pathname === '/') return 'Dashboard'
  if (pathname === '/new') return 'New analysis'
  if (pathname.startsWith('/analyses/')) return 'Analysis report'
  if (pathname === '/analyses') return 'Analyses'
  if (pathname === '/accuracy') return 'Model accuracy'
  return 'QYX MedTech'
}

type Health = 'checking' | 'ready' | 'not_ready' | 'offline'

export default function Header() {
  const { pathname } = useLocation()
  const [health, setHealth] = useState<Health>('checking')
  const [gpu, setGpu] = useState<boolean | null>(null)

  useEffect(() => {
    let alive = true
    const check = () => api.health()
      .then(h => { if (alive) { setHealth(h.status === 'ready' ? 'ready' : 'not_ready'); setGpu(h.checks.gpu) } })
      .catch(() => alive && setHealth('offline'))
    check()
    const t = setInterval(check, 15000)
    return () => { alive = false; clearInterval(t) }
  }, [])

  const label = {
    checking: 'Checking server…',
    ready: gpu === false ? 'Ready (CPU only, slower)' : 'Analyzer ready',
    not_ready: 'Server missing model files',
    offline: 'Server offline',
  }[health]
  const tone = health === 'ready' ? 'ok' : health === 'checking' ? 'muted' : 'bad'

  return (
    <header className="app-header">
      <div className="header-mobile-brand"><Brand /></div>
      <h1 className="header-title">{title(pathname)}</h1>
      <div className={`status-pill status-${tone}`} role="status">
        <span className="status-dot" />{label}
      </div>
      <nav className="mobile-nav" aria-label="Main">
        {navItems.map(({ label: l, path, icon: Icon, end }) => (
          <NavLink key={path} to={path} end={end} className={({ isActive }) => `mobile-nav-item${isActive ? ' active' : ''}`}>
            <Icon size={16} />{l}
          </NavLink>
        ))}
      </nav>
    </header>
  )
}
