import { useMutation, useQueryClient } from '@tanstack/react-query'

import { apiClient, type PostgresBackupRunSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'

export function usePostgresBackupRunMutation() {
  const queryClient = useQueryClient()
  return useMutation<PostgresBackupRunSnapshot>({
    mutationFn: () => apiClient.runPostgresBackupNow(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.postgresBackupStatus })
    },
  })
}
