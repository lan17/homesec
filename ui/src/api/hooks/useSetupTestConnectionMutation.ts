import { useMutation } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { TestConnectionSnapshot } from '../client'
import type { TestConnectionRequest } from '../generated/types'

export function useSetupTestConnectionMutation() {
  return useMutation<TestConnectionSnapshot, Error, { request: TestConnectionRequest; signal?: AbortSignal }>({
    mutationFn: ({ request, signal }) => apiClient.runSetupTestConnection(request, { signal }),
  })
}
