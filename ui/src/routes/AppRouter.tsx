import { Navigate, Route, Routes } from 'react-router-dom'

import { AppShell } from '../app/layout/AppShell'
import { CamerasPage } from '../features/cameras/CamerasPage'
import { ClipDetailPage } from '../features/clips/ClipDetailPage'
import { ClipsPage } from '../features/clips/ClipsPage'
import { DashboardPage } from '../features/dashboard/DashboardPage'
import { NotFoundPage } from '../features/not-found/NotFoundPage'
import { SetupPage } from '../features/setup/SetupPage'
import { StoragePage } from '../features/storage/StoragePage'

export function AppRouter() {
  return (
    <Routes>
      <Route path="/setup" element={<SetupPage />} />
      <Route element={<AppShell />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/cameras" element={<CamerasPage />} />
        <Route path="/storage" element={<StoragePage />} />
        <Route path="/clips" element={<ClipsPage />} />
        <Route path="/clips/:clipId" element={<ClipDetailPage />} />
        <Route path="/home" element={<Navigate to="/" replace />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  )
}
