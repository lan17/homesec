import { useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { apiClient, type ConfigPatch } from '../client'
import { QUERY_KEYS } from './queryKeys'

export function useConfigMutation() {
  const queryClient = useQueryClient()
  const [isPending, setPending] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const pending = useRef(false)

  async function mutateAsync(patch: ConfigPatch) {
    if (pending.current) {
      throw new Error('A settings save is already in progress.')
    }
    pending.current = true
    setPending(true)
    setError(null)
    try {
      // Keep write-only credentials out of React Query's mutation variables/cache.
      const config = await apiClient.patchConfig(patch)
      await queryClient.cancelQueries({ queryKey: QUERY_KEYS.config })
      queryClient.setQueryData(QUERY_KEYS.config, config)
      return config
    } catch (saveError) {
      setError(saveError)
      throw saveError
    } finally {
      pending.current = false
      setPending(false)
    }
  }

  return { mutateAsync, isPending, error, reset: () => { setError(null) } }
}
