import type { CameraCreate } from '../../../api/generated/types'
import { CameraAddFlow } from '../../cameras/add-flow/CameraAddFlow'

interface CameraStepProps {
  existingCameraNames: readonly string[]
  onComplete: () => void
  onUpdateData: (data: CameraCreate) => void
  onSkip: () => void
}

export function CameraStep({
  existingCameraNames,
  onComplete,
  onUpdateData,
  onSkip,
}: CameraStepProps) {
  return (
    <CameraAddFlow
      existingCameraNames={existingCameraNames}
      defaultApplyChangesImmediately={false}
      onComplete={async (payload) => {
        onUpdateData(payload)
        return { ok: true }
      }}
      onDone={onComplete}
      onCancel={onSkip}
    />
  )
}
