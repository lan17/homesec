import { describe, expect, it } from 'vitest'

import { describeCameraCreateError, validateCameraName } from './validation'
import { APIError } from '../../../api/errors'

describe('camera add-flow validation utilities', () => {
  it('validates camera name constraints', () => {
    // Given: Candidate camera names across valid and invalid formats
    const invalidEmpty = '   '
    const invalidSpecial = 'front door!'
    const valid = 'front_door-1'

    // When: Validation runs on each candidate
    const emptyError = validateCameraName(invalidEmpty)
    const specialError = validateCameraName(invalidSpecial)
    const validError = validateCameraName(valid)

    // Then: Empty and special chars fail, valid slug passes
    expect(emptyError).toBe('Camera name is required.')
    expect(specialError).toBe(
      'Camera name may contain only letters, numbers, underscores, and hyphens.',
    )
    expect(validError).toBeNull()
  })

  it('maps known API duplicate-name errors to user-friendly message', () => {
    // Given: API layer reports duplicate camera via canonical error code
    const error = new APIError(
      'Camera already exists.',
      409,
      { detail: 'Camera already exists.', error_code: 'CAMERA_ALREADY_EXISTS' },
      'CAMERA_ALREADY_EXISTS',
    )

    // When: UI formats create-camera error for display
    const description = describeCameraCreateError(error)

    // Then: Helper emits stable duplicate-name guidance
    expect(description).toBe('Camera name already exists. Choose a different name and retry.')
  })
})
