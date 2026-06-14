import { Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom'

import { isRuntimeAuthSessionReady, runtimeServerBaseUrlProvider } from '../api/client'
import { AppShell } from '../app/layout/AppShell'
import { isIOSNativeApp } from '../runtime/nativeRuntime'
import { useNativePushRegistration } from '../runtime/nativePushRegistration'
import { CamerasPage } from '../features/cameras/CamerasPage'
import { ClipDetailPage } from '../features/clips/ClipDetailPage'
import { ClipsPage } from '../features/clips/ClipsPage'
import { LivePage } from '../features/live/LivePage'
import { NativeSetupPage } from '../features/native-setup/NativeSetupPage'
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

function NativeSetupGuard() {
  const location = useLocation()
  const serverBaseUrl = runtimeServerBaseUrlProvider.getBaseUrlSync()
  const nativeSetupRequired = isIOSNativeApp() && (!serverBaseUrl || !isRuntimeAuthSessionReady())

  useNativePushRegistration({
    enabled: isIOSNativeApp() && !nativeSetupRequired,
    registrationKey: serverBaseUrl ?? 'ios-native',
  })

  if (nativeSetupRequired) {
    return (
      <Navigate
        to="/native-setup"
        replace
        state={{ nativeSetupReturnTo: `${location.pathname}${location.search}${location.hash}` }}
      />
    )
  }

  return <AppShell />
}

export function AppRouter() {
  return (
    <Routes>
      <Route path="/setup" element={<SetupPage />} />
      <Route path="/native-setup" element={<NativeSetupPage />} />
      <Route element={<NativeSetupGuard />}>
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
