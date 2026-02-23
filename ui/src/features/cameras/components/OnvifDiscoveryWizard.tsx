import { useMemo, useState } from 'react'

import { isAPIError } from '../../../api/client'
import type {
  CameraCreate,
  DiscoveredCameraResponse,
  ProbeResponse,
} from '../../../api/generated/types'
import { Card } from '../../../components/ui/Card'
import { describeUnknownError } from '../../shared/errorPresentation'
import { discoverOnvifCameras, probeOnvifCamera } from '../onvifApi'
import { deriveOnvifCameraName, injectCredentialsIntoRtspUri } from '../presentationOnvif'
import { OnvifDiscoverStep } from './OnvifDiscoverStep'
import { OnvifProbeStep, type OnvifProbeCredentials } from './OnvifProbeStep'
import { OnvifStreamSelectStep } from './OnvifStreamSelectStep'
import type { CameraCreateActionResult } from '../hooks/useCameraActions'

interface OnvifDiscoveryWizardProps {
  applyChangesImmediately: boolean
  createPending: boolean
  isMutating: boolean
  onCreateCamera: (
    payload: CameraCreate,
    applyChanges: boolean,
  ) => Promise<CameraCreateActionResult>
  onClose: () => void
}

type WizardStep = 'discover' | 'probe' | 'select-stream'

const DEFAULT_PROBE_CREDENTIALS: OnvifProbeCredentials = {
  username: '',
  password: '',
  port: 80,
}

export function OnvifDiscoveryWizard({
  applyChangesImmediately,
  createPending,
  isMutating,
  onCreateCamera,
  onClose,
}: OnvifDiscoveryWizardProps) {
  const [step, setStep] = useState<WizardStep>('discover')
  const [discoveredCameras, setDiscoveredCameras] = useState<DiscoveredCameraResponse[]>([])
  const [selectedCamera, setSelectedCamera] = useState<DiscoveredCameraResponse | null>(null)
  const [probeResult, setProbeResult] = useState<ProbeResponse | null>(null)
  const [probeCredentials, setProbeCredentials] = useState<OnvifProbeCredentials>(
    DEFAULT_PROBE_CREDENTIALS,
  )
  const [selectedProfileToken, setSelectedProfileToken] = useState<string | null>(null)
  const [cameraName, setCameraName] = useState('')
  const [hasScanned, setHasScanned] = useState(false)

  const [discoverPending, setDiscoverPending] = useState(false)
  const [probePending, setProbePending] = useState(false)
  const [discoverError, setDiscoverError] = useState<string | null>(null)
  const [probeError, setProbeError] = useState<string | null>(null)
  const [createError, setCreateError] = useState<string | null>(null)

  function resetWizard(): void {
    setStep('discover')
    setDiscoveredCameras([])
    setSelectedCamera(null)
    setProbeResult(null)
    setProbeCredentials(DEFAULT_PROBE_CREDENTIALS)
    setSelectedProfileToken(null)
    setCameraName('')
    setHasScanned(false)
    setDiscoverPending(false)
    setProbePending(false)
    setDiscoverError(null)
    setProbeError(null)
    setCreateError(null)
  }

  function closeWizard(): void {
    resetWizard()
    onClose()
  }

  async function handleDiscover(): Promise<void> {
    setDiscoverPending(true)
    setDiscoverError(null)
    setHasScanned(true)
    try {
      const cameras = await discoverOnvifCameras()
      setDiscoveredCameras(cameras)
    } catch (error) {
      setDiscoverError(describeUnknownError(error))
      setDiscoveredCameras([])
    } finally {
      setDiscoverPending(false)
    }
  }

  function handleSelectCamera(camera: DiscoveredCameraResponse): void {
    setSelectedCamera(camera)
    setStep('probe')
    setProbeResult(null)
    setSelectedProfileToken(null)
    setCreateError(null)
  }

  async function handleProbe(): Promise<void> {
    if (!selectedCamera) {
      return
    }

    if (probeCredentials.username.trim().length === 0 || probeCredentials.password.trim().length === 0) {
      setProbeError('Username and password are required.')
      return
    }
    if (probeCredentials.port < 1 || probeCredentials.port > 65535) {
      setProbeError('Port must be between 1 and 65535.')
      return
    }

    setProbePending(true)
    setProbeError(null)
    setCreateError(null)
    try {
      const result = await probeOnvifCamera({
        host: selectedCamera.ip,
        port: probeCredentials.port,
        username: probeCredentials.username,
        password: probeCredentials.password,
      })
      setProbeResult(result)
      const defaultProfile = result.profiles.find((profile) => profile.stream_uri !== null) ?? null
      setSelectedProfileToken(defaultProfile?.token ?? null)
      setCameraName(
        deriveOnvifCameraName({
          discoveredCamera: selectedCamera,
          deviceInfo: result.device,
        }),
      )
      setStep('select-stream')
    } catch (error) {
      setProbeError(describeUnknownError(error))
    } finally {
      setProbePending(false)
    }
  }

  const selectedProfile = useMemo(() => {
    if (!probeResult || !selectedProfileToken) {
      return null
    }
    return probeResult.profiles.find((profile) => profile.token === selectedProfileToken) ?? null
  }, [probeResult, selectedProfileToken])

  function describeCreateError(error: unknown): string {
    if (isAPIError(error) && error.errorCode === 'CAMERA_ALREADY_EXISTS') {
      return 'Camera name already exists. Choose a different name and retry.'
    }
    if (error instanceof Error && error.message.trim().length > 0) {
      return `Create camera failed: ${error.message}`
    }
    return 'Create camera failed. Review the page error details and retry.'
  }

  async function handleCreateCamera(): Promise<void> {
    if (!selectedProfile?.stream_uri) {
      setCreateError('Select a stream profile with a valid RTSP URI.')
      return
    }
    const trimmedName = cameraName.trim()
    if (!trimmedName) {
      setCreateError('Camera name is required.')
      return
    }

    setCreateError(null)
    const rtspUrl = injectCredentialsIntoRtspUri({
      streamUri: selectedProfile.stream_uri,
      username: probeCredentials.username,
      password: probeCredentials.password,
    })
    const createResult = await onCreateCamera(
      {
        name: trimmedName,
        enabled: true,
        source_backend: 'rtsp',
        source_config: {
          rtsp_url: rtspUrl,
        },
      },
      applyChangesImmediately,
    )
    if (createResult.ok) {
      closeWizard()
      return
    }
    setCreateError(describeCreateError(createResult.error))
  }

  return (
    <Card
      title="ONVIF Discovery Wizard"
      subtitle="Discover, probe, and create RTSP camera configs from ONVIF metadata."
    >
      <div className="onvif-wizard">
        {step === 'discover' ? (
          <OnvifDiscoverStep
            cameras={discoveredCameras}
            hasScanned={hasScanned}
            isScanning={discoverPending}
            error={discoverError}
            onScan={() => {
              void handleDiscover()
            }}
            onSelect={handleSelectCamera}
            onCancel={closeWizard}
          />
        ) : null}

        {step === 'probe' && selectedCamera ? (
          <OnvifProbeStep
            camera={selectedCamera}
            credentials={probeCredentials}
            isProbing={probePending}
            error={probeError}
            onCredentialsChange={setProbeCredentials}
            onProbe={() => {
              void handleProbe()
            }}
            onBack={() => {
              setStep('discover')
              setProbeError(null)
            }}
            onCancel={closeWizard}
          />
        ) : null}

        {step === 'select-stream' && selectedCamera && probeResult ? (
          <OnvifStreamSelectStep
            camera={selectedCamera}
            probeResult={probeResult}
            selectedProfileToken={selectedProfileToken}
            cameraName={cameraName}
            createPending={createPending}
            isMutating={isMutating}
            applyChangesImmediately={applyChangesImmediately}
            error={createError}
            onSelectProfile={setSelectedProfileToken}
            onCameraNameChange={setCameraName}
            onCreate={() => {
              void handleCreateCamera()
            }}
            onBack={() => {
              setStep('probe')
              setCreateError(null)
            }}
            onCancel={closeWizard}
          />
        ) : null}
      </div>
    </Card>
  )
}
