import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'

export default function Layout() {
  return (
    <div className="app-layout">
      <Sidebar />
      <div className="app-content">
        <Header />
        <main className="app-main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
