import { useEffect, useRef, useState, type RefObject } from 'react'

import type { PreviewSessionSnapshot } from '../../../api/client'
import { WebRTCPreviewPlayer, type WebRTCPlayerState } from '../preview/WebRTCPreviewPlayer'

const CLOSED_STATE: WebRTCPlayerState = { state: 'closed', error: null }

export function useWebRTCPreview(
  session: PreviewSessionSnapshot | null,
  videoRef: RefObject<HTMLVideoElement | null>,
  attempt = 0,
): WebRTCPlayerState {
  const playerRef = useRef<WebRTCPreviewPlayer | null>(null)
  const sessionRef = useRef(session)
  const [state, setState] = useState<WebRTCPlayerState>(CLOSED_STATE)
  const cameraName = session?.camera_name
  const transport = session?.transport
  const signalingUrl = session?.signaling_url

  useEffect(() => {
    sessionRef.current = session
    if (session) {
      playerRef.current?.updateSession(session)
    }
  }, [session])

  useEffect(() => {
    if (transport !== 'webrtc' || !signalingUrl || !videoRef.current) {
      return
    }
    const video = videoRef.current
    let mounted = true
    const start = (): void => {
      const currentSession = sessionRef.current
      if (!currentSession || document.visibilityState === 'hidden' || playerRef.current) {
        return
      }
      const player = new WebRTCPreviewPlayer(currentSession, video, (nextState) => {
        if (mounted) {
          setState(nextState)
        }
      })
      playerRef.current = player
      player.start()
    }
    const close = (): void => {
      playerRef.current?.close()
      playerRef.current = null
    }
    const onVisibility = (): void => {
      if (document.visibilityState === 'hidden') {
        close()
      } else {
        start()
      }
    }
    start()
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('pagehide', close)
    window.addEventListener('pageshow', start)
    return () => {
      mounted = false
      document.removeEventListener('visibilitychange', onVisibility)
      window.removeEventListener('pagehide', close)
      window.removeEventListener('pageshow', start)
      close()
    }
  }, [cameraName, transport, signalingUrl, videoRef, attempt])

  return transport === 'webrtc' ? state : CLOSED_STATE
}
