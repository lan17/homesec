import { useMemo, useReducer } from 'react'

import type {
  CameraCreate,
  DiscoveredCameraResponse,
  ProbeResponse,
} from '../../../api/generated/types'
import { Card } from '../../../components/ui/Card'
import { describeUnknownError } from '../../shared/errorPresentation'
import { discoverOnvifCameras, probeOnvifCamera } from '../onvifApi'
import { ONVIF_DEFAULT_PORT, ONVIF_DEFAULT_PROBE_TIMEOUT_S } from '../onvifDefaults'
import {
  deriveOnvifCameraName,
  deriveOnvifProbePortFromXaddr,
  injectCredentialsIntoRtspUri,
} from '../presentationOnvif'
import { OnvifDiscoverStep } from './OnvifDiscoverStep'
import { OnvifProbeStep, type OnvifProbeCredentials } from './OnvifProbeStep'
import { OnvifStreamSelectStep } from './OnvifStreamSelectStep'
import type { CameraCreateActionResult } from '../actions'
import { describeCameraCreateError } from '../add-flow/validation'

interface OnvifDiscoveryWizardProps {
  applyChangesImmediately: boolean
  createPending: boolean
  isMutating: boolean
  onCreateCamera: (payload: CameraCreate) => Promise<CameraCreateActionResult>
  submitLabel?: string
  showApplyChangesSummary?: boolean
  onClose: () => void
}

type WizardStep = 'discover' | 'probe' | 'select-stream'

const DEFAULT_PROBE_CREDENTIALS: OnvifProbeCredentials = {
  username: '',
  password: '',
  port: ONVIF_DEFAULT_PORT,
}

interface OnvifWizardState {
  step: WizardStep
  discoveredCameras: DiscoveredCameraResponse[]
  selectedCamera: DiscoveredCameraResponse | null
  probeResult: ProbeResponse | null
  probeCredentials: OnvifProbeCredentials
  selectedProfileToken: string | null
  cameraName: string
  hasScanned: boolean
  discoverPending: boolean
  probePending: boolean
  discoverError: string | null
  probeError: string | null
  createError: string | null
}

type OnvifWizardAction =
  | { type: 'reset' }
  | { type: 'discover_started' }
  | { type: 'discover_succeeded'; cameras: DiscoveredCameraResponse[] }
  | { type: 'discover_failed'; error: string }
  | { type: 'camera_selected'; camera: DiscoveredCameraResponse; inferredPort: number }
  | { type: 'probe_credentials_changed'; credentials: OnvifProbeCredentials }
  | { type: 'probe_started' }
  | {
      type: 'probe_succeeded'
      result: ProbeResponse
      selectedProfileToken: string | null
      cameraName: string
    }
  | { type: 'probe_failed'; error: string }
  | { type: 'stream_selected'; token: string }
  | { type: 'camera_name_changed'; cameraName: string }
  | { type: 'create_error_set'; error: string }
  | { type: 'create_error_cleared' }
  | { type: 'back_to_discover' }
  | { type: 'back_to_probe' }

function createInitialWizardState(): OnvifWizardState {
  return {
    step: 'discover',
    discoveredCameras: [],
    selectedCamera: null,
    probeResult: null,
    probeCredentials: DEFAULT_PROBE_CREDENTIALS,
    selectedProfileToken: null,
    cameraName: '',
    hasScanned: false,
    discoverPending: false,
    probePending: false,
    discoverError: null,
    probeError: null,
    createError: null,
  }
}

function onvifWizardReducer(state: OnvifWizardState, action: OnvifWizardAction): OnvifWizardState {
  switch (action.type) {
    case 'reset':
      return createInitialWizardState()
    case 'discover_started':
      return {
        ...state,
        hasScanned: true,
        discoverPending: true,
        discoverError: null,
      }
    case 'discover_succeeded':
      return {
        ...state,
        discoverPending: false,
        discoveredCameras: action.cameras,
      }
    case 'discover_failed':
      return {
        ...state,
        discoverPending: false,
        discoverError: action.error,
        discoveredCameras: [],
      }
    case 'camera_selected':
      return {
        ...state,
        step: 'probe',
        selectedCamera: action.camera,
        probeResult: null,
        probeCredentials: {
          username: '',
          password: '',
          port: action.inferredPort,
        },
        selectedProfileToken: null,
        createError: null,
      }
    case 'probe_credentials_changed':
      return {
        ...state,
        probeCredentials: action.credentials,
      }
    case 'probe_started':
      return {
        ...state,
        probePending: true,
        probeError: null,
        createError: null,
      }
    case 'probe_succeeded':
      return {
        ...state,
        step: 'select-stream',
        probePending: false,
        probeResult: action.result,
        selectedProfileToken: action.selectedProfileToken,
        cameraName: action.cameraName,
      }
    case 'probe_failed':
      return {
        ...state,
        probePending: false,
        probeError: action.error,
      }
    case 'stream_selected':
      return {
        ...state,
        selectedProfileToken: action.token,
      }
    case 'camera_name_changed':
      return {
        ...state,
        cameraName: action.cameraName,
      }
    case 'create_error_set':
      return {
        ...state,
        createError: action.error,
      }
    case 'create_error_cleared':
      return {
        ...state,
        createError: null,
      }
    case 'back_to_discover':
      return {
        ...state,
        step: 'discover',
        probeError: null,
      }
    case 'back_to_probe':
      return {
        ...state,
        step: 'probe',
        createError: null,
      }
  }
}

export function OnvifDiscoveryWizard({
  applyChangesImmediately,
  createPending,
  isMutating,
  onCreateCamera,
  submitLabel,
  showApplyChangesSummary,
  onClose,
}: OnvifDiscoveryWizardProps) {
  const [state, dispatch] = useReducer(onvifWizardReducer, undefined, createInitialWizardState)

  function closeWizard(): void {
    dispatch({ type: 'reset' })
    onClose()
  }

  async function handleDiscover(): Promise<void> {
    dispatch({ type: 'discover_started' })
    try {
      const cameras = await discoverOnvifCameras()
      dispatch({ type: 'discover_succeeded', cameras })
    } catch (error) {
      dispatch({
        type: 'discover_failed',
        error: describeUnknownError(error),
      })
    }
  }

  function handleSelectCamera(camera: DiscoveredCameraResponse): void {
    dispatch({
      type: 'camera_selected',
      camera,
      inferredPort: deriveOnvifProbePortFromXaddr(camera.xaddr),
    })
  }

  async function handleProbe(): Promise<void> {
    if (!state.selectedCamera) {
      return
    }

    if (
      state.probeCredentials.username.trim().length === 0
      || state.probeCredentials.password.trim().length === 0
    ) {
      dispatch({
        type: 'probe_failed',
        error: 'Username and password are required.',
      })
      return
    }
    if (state.probeCredentials.port < 1 || state.probeCredentials.port > 65535) {
      dispatch({
        type: 'probe_failed',
        error: 'Port must be between 1 and 65535.',
      })
      return
    }

    dispatch({ type: 'probe_started' })
    try {
      const result = await probeOnvifCamera({
        host: state.selectedCamera.ip,
        port: state.probeCredentials.port,
        timeout_s: ONVIF_DEFAULT_PROBE_TIMEOUT_S,
        username: state.probeCredentials.username,
        password: state.probeCredentials.password,
      })
      const defaultProfile = result.profiles.find((profile) => profile.stream_uri !== null) ?? null
      dispatch({
        type: 'probe_succeeded',
        result,
        selectedProfileToken: defaultProfile?.token ?? null,
        cameraName: deriveOnvifCameraName({
          discoveredCamera: state.selectedCamera,
          deviceInfo: result.device,
        }),
      })
    } catch (error) {
      dispatch({
        type: 'probe_failed',
        error: describeUnknownError(error),
      })
    }
  }

  const selectedProfile = useMemo(() => {
    if (!state.probeResult || !state.selectedProfileToken) {
      return null
    }
    return (
      state.probeResult.profiles.find((profile) => profile.token === state.selectedProfileToken)
      ?? null
    )
  }, [state.probeResult, state.selectedProfileToken])

  async function handleCreateCamera(): Promise<void> {
    if (!selectedProfile?.stream_uri) {
      dispatch({
        type: 'create_error_set',
        error: 'Select a stream profile with a valid RTSP URI.',
      })
      return
    }
    const trimmedName = state.cameraName.trim()
    if (!trimmedName) {
      dispatch({
        type: 'create_error_set',
        error: 'Camera name is required.',
      })
      return
    }

    dispatch({ type: 'create_error_cleared' })
    const rtspUrl = injectCredentialsIntoRtspUri({
      streamUri: selectedProfile.stream_uri,
      username: state.probeCredentials.username,
      password: state.probeCredentials.password,
    })
    const createResult = await onCreateCamera({
      name: trimmedName,
      enabled: true,
      source_backend: 'rtsp',
      source_config: {
        rtsp_url: rtspUrl,
      },
    })
    if (createResult.ok) {
      closeWizard()
      return
    }
    dispatch({
      type: 'create_error_set',
      error: describeCameraCreateError(createResult.error),
    })
  }

  return (
    <Card
      title="ONVIF Discovery Wizard"
      subtitle="Discover, probe, and create RTSP camera configs from ONVIF metadata."
    >
      <div className="onvif-wizard">
        {state.step === 'discover' ? (
          <OnvifDiscoverStep
            cameras={state.discoveredCameras}
            hasScanned={state.hasScanned}
            isScanning={state.discoverPending}
            error={state.discoverError}
            onScan={() => {
              void handleDiscover()
            }}
            onSelect={handleSelectCamera}
            onCancel={closeWizard}
          />
        ) : null}

        {state.step === 'probe' && state.selectedCamera ? (
          <OnvifProbeStep
            camera={state.selectedCamera}
            credentials={state.probeCredentials}
            isProbing={state.probePending}
            error={state.probeError}
            onCredentialsChange={(credentials) => {
              dispatch({
                type: 'probe_credentials_changed',
                credentials,
              })
            }}
            onProbe={() => {
              void handleProbe()
            }}
            onBack={() => {
              dispatch({ type: 'back_to_discover' })
            }}
            onCancel={closeWizard}
          />
        ) : null}

        {state.step === 'select-stream' && state.selectedCamera && state.probeResult ? (
          <OnvifStreamSelectStep
            camera={state.selectedCamera}
            probeResult={state.probeResult}
            selectedProfileToken={state.selectedProfileToken}
            cameraName={state.cameraName}
            createPending={createPending}
            isMutating={isMutating}
            applyChangesImmediately={applyChangesImmediately}
            submitLabel={submitLabel}
            showApplyChangesSummary={showApplyChangesSummary}
            error={state.createError}
            onSelectProfile={(token) => {
              dispatch({ type: 'stream_selected', token })
            }}
            onCameraNameChange={(cameraName) => {
              dispatch({
                type: 'camera_name_changed',
                cameraName,
              })
            }}
            onCreate={() => {
              void handleCreateCamera()
            }}
            onBack={() => {
              dispatch({ type: 'back_to_probe' })
            }}
            onCancel={closeWizard}
          />
        ) : null}
      </div>
    </Card>
  )
}
