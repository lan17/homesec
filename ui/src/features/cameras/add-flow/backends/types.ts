import type { JSX } from 'react'

import type { CameraCreate, TestConnectionRequest } from '../../../../api/generated/types'
import type { CameraBackend } from '../../forms'
import type { CameraAddBackend } from '../types'

export interface BackendFormStepProps {
  config: Record<string, unknown>
  onChange: (config: Record<string, unknown>) => void
  stepIndex: number
  onSuggestedNameChange: (cameraName: string) => void
}

export interface BackendStepDef {
  title: string
  component: (props: BackendFormStepProps) => JSX.Element
}

export interface BackendCameraSource {
  source_backend: CameraBackend
  source_config: CameraCreate['source_config']
}

export interface BackendFormDef {
  id: CameraAddBackend
  label: string
  description: string
  steps: readonly BackendStepDef[]
  defaultConfig: Record<string, unknown>
  suggestNamePrefix: string
  validateStep: (stepIndex: number, config: Record<string, unknown>) => string | null
  buildCameraSource: (config: Record<string, unknown>) => BackendCameraSource
  buildTestRequest: (config: Record<string, unknown>) => TestConnectionRequest
}
