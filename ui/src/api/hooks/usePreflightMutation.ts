import { useMutation } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { PreflightSnapshot } from '../client'

export function usePreflightMutation() {
  return useMutation<PreflightSnapshot, Error>({
    mutationFn: () => apiClient.runSetupPreflight(),
  })
}
