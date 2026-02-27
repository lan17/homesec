import { useMutation } from '@tanstack/react-query'

import { apiClient, type FinalizeSnapshot } from '../client'
import type { FinalizeRequest } from '../generated/types'

interface FinalizeMutationInput {
  payload: FinalizeRequest
  signal?: AbortSignal
}

export function useFinalizeMutation() {
  return useMutation<FinalizeSnapshot, Error, FinalizeMutationInput>({
    mutationFn: ({ payload, signal }) => apiClient.finalizeSetup(payload, { signal }),
  })
}
