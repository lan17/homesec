import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { apiClient, type ConfigPatch, type ConfigSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'

export function useConfigMutation() {
  const queryClient = useQueryClient()
  const [isPending, setPending] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const pending = useRef(false)
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => () => { controllerRef.current?.abort() }, [])

  async function mutateAsync(patch: ConfigPatch) {
    if (pending.current) {
      throw new Error('A settings save is already in progress.')
    }
    pending.current = true
    const controller = new AbortController()
    controllerRef.current = controller
    setPending(true)
    setError(null)
    try {
      // Keep write-only credentials out of React Query's mutation variables/cache.
      const config = await apiClient.patchConfig(patch, { signal: controller.signal })
      if (controller.signal.aborted) { throw new DOMException('Settings save cancelled', 'AbortError') }
      await queryClient.cancelQueries({ queryKey: QUERY_KEYS.config })
      if (controller.signal.aborted) { throw new DOMException('Settings save cancelled', 'AbortError') }
      const observed = queryClient.getQueryData<ConfigSnapshot>(QUERY_KEYS.config)
      const interveningRevision = observed && observed.saved_config_version !== patch.expected_config_version
        && observed.saved_config_version !== config.saved_config_version
      const interveningActivation = observed?.saved_config_version === config.saved_config_version
        && (observed.active_config_version !== config.active_config_version || observed.apply_required !== config.apply_required)
      if (interveningRevision || interveningActivation) {
        // Another saved or active snapshot was observed while this response was in flight.
        await queryClient.refetchQueries({ queryKey: QUERY_KEYS.config })
        if (controller.signal.aborted) { throw new DOMException('Settings save cancelled', 'AbortError') }
        return queryClient.getQueryData<ConfigSnapshot>(QUERY_KEYS.config) ?? config
      }
      queryClient.setQueryData(QUERY_KEYS.config, config)
      return config
    } catch (saveError) {
      if (!controller.signal.aborted) { setError(saveError) }
      throw saveError
    } finally {
      pending.current = false
      if (!controller.signal.aborted) { setPending(false) }
    }
  }

  return { mutateAsync, isPending, error, reset: () => { setError(null) } }
}
