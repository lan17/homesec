import type { ConfigResponse } from '../../api/generated/types'
import type { CredentialDraft, CredentialFields } from './CredentialField'

export function updateCredentialDraft(draft: CredentialDraft, path: string, value: string | null | undefined): CredentialDraft {
  const updated = { ...draft }
  if (value === undefined) {
    delete updated[path]
  } else {
    updated[path] = value
  }
  return updated
}

export function credentialsBlockProbe(credentials: CredentialFields, applyRequired: ConfigResponse['apply_required']): boolean {
  return Object.keys(credentials.draft).some((path) => path.startsWith(`${credentials.prefix}.`))
    || (applyRequired !== 'none' && Object.entries(credentials.statuses).some(
      ([path, status]) => path.startsWith(`${credentials.prefix}.`) && status.source === 'managed',
    ))
}
