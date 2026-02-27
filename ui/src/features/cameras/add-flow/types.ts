import type { TestConnectionResponse } from '../../../api/generated/types'
import type { CameraBackend } from '../forms'

export type CameraAddBackend = CameraBackend | 'onvif'

export type CameraAddStage = 'pick-backend' | 'configure' | 'test' | 'confirm'

export interface CameraAddState {
  step: CameraAddStage
  backend: CameraAddBackend | null
  backendStepIndex: number
  config: Record<string, unknown>
  cameraName: string
  testResult: TestConnectionResponse | null
}

