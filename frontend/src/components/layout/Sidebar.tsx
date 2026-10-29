import { NavLink } from 'react-router-dom'
import { LayoutDashboard, Upload, ListChecks, Target, Microscope } from 'lucide-react'

export const navItems = [
  { label: 'Dashboard', path: '/', icon: LayoutDashboard, end: true },
  { label: 'New analysis', path: '/new', icon: Upload, end: true },
  { label: 'Analyses', path: '/analyses', icon: ListChecks, end: false },
  { label: 'Model accuracy', path: '/accuracy', icon: Target, end: true },
]

export function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark"><Microscope size={20} color="#fff" /></div>
      <div>
        <div className="brand-name">QYX</div>
        <div className="brand-sub">SEMEN ANALYSIS</div>
      </div>
    </div>
  )
}

export default function Sidebar() {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand"><Brand /></div>
      <nav className="sidebar-nav" aria-label="Main">
        {navItems.map(({ label, path, icon: Icon, end }) => (
          <NavLink key={path} to={path} end={end} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-foot">
        Research use only. Not a diagnostic device.
      </div>
    </aside>
  )
}
