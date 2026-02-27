import type { CameraCreate } from '../../api/generated/types'

export type CameraCreateActionResult =
  | { ok: true }
  | { ok: false; error: unknown }

export interface CameraAddFlowCompleteOptions {
  applyChangesImmediately: boolean
}

export type CameraAddFlowOnComplete = (
  payload: CameraCreate,
  options: CameraAddFlowCompleteOptions,
) => Promise<CameraCreateActionResult>
