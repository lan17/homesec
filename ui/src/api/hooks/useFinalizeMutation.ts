import { useMutation } from '@tanstack/react-query'

import { apiClient, type FinalizeSnapshot } from '../client'
import type { FinalizeRequest } from '../generated/types'

export function useFinalizeMutation() {
  return useMutation<FinalizeSnapshot, Error, FinalizeRequest>({
    mutationFn: (payload) => apiClient.finalizeSetup(payload),
  })
}

