import { useQuery } from '@tanstack/react-query'

import { apiClient, type PostgresBackupStatusSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'

const POSTGRES_BACKUP_REFRESH_MS = 10_000

export function usePostgresBackupStatusQuery() {
  return useQuery<PostgresBackupStatusSnapshot>({
    queryKey: QUERY_KEYS.postgresBackupStatus,
    queryFn: ({ signal }) => apiClient.getPostgresBackupStatus({ signal }),
    staleTime: POSTGRES_BACKUP_REFRESH_MS,
    refetchInterval: POSTGRES_BACKUP_REFRESH_MS,
  })
}
