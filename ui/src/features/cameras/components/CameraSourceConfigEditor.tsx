import { useMemo, useState, type FormEvent } from 'react'

import type { CameraCreate, CameraResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import {
  defaultSourceConfigPatchForCamera,
  parseSourceConfigPatchJson,
} from '../forms'

interface CameraSourceConfigEditorProps {
  camera: CameraResponse
  isMutating: boolean
  updatePending: boolean
  applyChangesImmediately: boolean
  onSubmitPatch: (
    cameraName: string,
    sourceConfigPatch: CameraCreate['source_config'],
  ) => Promise<boolean>
}

export function CameraSourceConfigEditor({
  camera,
  isMutating,
  updatePending,
  applyChangesImmediately,
  onSubmitPatch,
}: CameraSourceConfigEditorProps) {
  const defaultPatchRaw = useMemo(
    () => defaultSourceConfigPatchForCamera(camera.source_config),
    [camera.source_config],
  )
  const [isOpen, setIsOpen] = useState(false)
  const [patchRaw, setPatchRaw] = useState<string | null>(null)
  const [formError, setFormError] = useState<string | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    const activePatchRaw = patchRaw ?? defaultPatchRaw

    const parsedPatch = parseSourceConfigPatchJson(activePatchRaw)
    if (!parsedPatch.ok) {
      setFormError(parsedPatch.message)
      return
    }

    if (Object.keys(parsedPatch.value).length === 0) {
      setFormError('Source config patch cannot be empty.')
      return
    }

    setFormError(null)
    const updated = await onSubmitPatch(camera.name, parsedPatch.value)
    if (updated) {
      setIsOpen(false)
      setPatchRaw(null)
      setFormError(null)
      return
    }
    setFormError('Source config update failed. Review the page error details and retry.')
  }

  if (!isOpen) {
    return (
      <Button
        variant="ghost"
        onClick={() => {
          setIsOpen(true)
        }}
        disabled={isMutating}
      >
        Edit source config
      </Button>
    )
  }

  return (
    <form className="camera-config-editor" onSubmit={(event) => void handleSubmit(event)}>
      <p className="muted">
        Patch only changed keys. Use <code>null</code> to clear optional fields. Omit redacted values
        unless replacing them.
      </p>
      <label className="field-label" htmlFor={`camera-patch-config-${camera.name}`}>
        Source config patch (JSON)
        <textarea
          id={`camera-patch-config-${camera.name}`}
          className="input camera-source-config"
          value={patchRaw ?? defaultPatchRaw}
          onChange={(event) => setPatchRaw(event.target.value)}
          disabled={isMutating}
        />
      </label>

      {formError ? <p className="error-text">{formError}</p> : null}

      <div className="inline-form__actions">
        <Button type="submit" disabled={isMutating}>
          {updatePending ? 'Applying patch...' : 'Apply source patch'}
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            setPatchRaw(defaultPatchRaw)
            setFormError(null)
          }}
          disabled={isMutating}
        >
          Load current non-secret values
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            setIsOpen(false)
            setPatchRaw(null)
            setFormError(null)
          }}
          disabled={isMutating}
        >
          Cancel
        </Button>
      </div>

      <p className="subtle">
        Apply mode: {applyChangesImmediately ? 'immediate runtime reload' : 'config only (reload later)'}
      </p>
    </form>
  )
}
