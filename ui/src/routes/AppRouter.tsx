import { Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom'

import { AppShell } from '../app/layout/AppShell'
import { CamerasPage } from '../features/cameras/CamerasPage'
import { ClipDetailPage } from '../features/clips/ClipDetailPage'
import { ClipsPage } from '../features/clips/ClipsPage'
import { LivePage } from '../features/live/LivePage'
import { NotFoundPage } from '../features/not-found/NotFoundPage'
import { SettingsPage } from '../features/settings/SettingsPage'
import { SetupPage } from '../features/setup/SetupPage'
import { SystemPage } from '../features/system/SystemPage'

function RedirectWithSearch({ to }: { to: string }) {
  const location = useLocation()
  return <Navigate to={`${to}${location.search}`} replace />
}

function RedirectClipDetailToEvent() {
  const { clipId } = useParams<{ clipId: string }>()
  const location = useLocation()
  return <Navigate to={`/events/${encodeURIComponent(clipId ?? '')}${location.search}`} replace />
}

export function AppRouter() {
  return (
    <Routes>
      <Route path="/setup" element={<SetupPage />} />
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to="/live" replace />} />
        <Route path="/live" element={<LivePage />} />
        <Route path="/events" element={<ClipsPage />} />
        <Route path="/events/:clipId" element={<ClipDetailPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/settings/cameras" element={<CamerasPage />} />
        <Route path="/system" element={<SystemPage />} />
        <Route path="/cameras" element={<RedirectWithSearch to="/settings/cameras" />} />
        <Route path="/clips" element={<RedirectWithSearch to="/events" />} />
        <Route path="/clips/:clipId" element={<RedirectClipDetailToEvent />} />
        <Route path="/dashboard" element={<RedirectWithSearch to="/system" />} />
        <Route path="/home" element={<Navigate to="/live" replace />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
