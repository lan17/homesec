import { isAPIError } from '../../../api/client'

const CAMERA_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/

export function describeCameraCreateError(error: unknown): string {
  if (isAPIError(error) && error.errorCode === 'CAMERA_ALREADY_EXISTS') {
    return 'Camera name already exists. Choose a different name and retry.'
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return `Create camera failed: ${error.message}`
  }
  return 'Create camera failed. Review the page error details and retry.'
}

export function validateCameraName(cameraName: string): string | null {
  const trimmed = cameraName.trim()
  if (!trimmed) {
    return 'Camera name is required.'
  }
  if (!CAMERA_NAME_PATTERN.test(trimmed)) {
    return 'Camera name may contain only letters, numbers, underscores, and hyphens.'
  }
  return null
}

