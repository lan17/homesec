import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { apiClient, isAPIError, type ConfigPatch } from '../../api/client'
import { useConfigMutation } from '../../api/hooks/useConfigMutation'
import { useConfigQuery } from '../../api/hooks/useConfigQuery'
import { QUERY_KEYS } from '../../api/hooks/queryKeys'
import { CONFIG_APPLY_TIMEOUT_MS, waitForConfigApply } from './configApply'

export function useConfigSettings() {
  const configQuery = useConfigQuery()
  const saveMutation = useConfigMutation()
  const queryClient = useQueryClient()
  const [applyPending, setApplyPending] = useState(false)
  const [applyError, setApplyError] = useState<unknown>(null)
  const [applyMessage, setApplyMessage] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  useEffect(() => () => { controllerRef.current?.abort() }, [])

  async function saveConfig(patch: ConfigPatch) {
    setApplyMessage(null)
    setApplyError(null)
    const config = await saveMutation.mutateAsync(patch)
    setApplyMessage('Settings saved. Apply the saved changes to activate them.')
    return config
  }

  async function refreshConfig(): Promise<void> {
    saveMutation.reset()
    setApplyError(null)
    setApplyMessage(null)
    await configQuery.refetch()
  }

  async function applyConfig(): Promise<void> {
    const config = configQuery.data
    if (!config || applyPending) {
      return
    }
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setApplyPending(true)
    setApplyError(null)
    setApplyMessage(null)
    let timedOut = false
    const timer = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, CONFIG_APPLY_TIMEOUT_MS)
    try {
      const response = await apiClient.applyConfig(
        { expected_config_version: config.saved_config_version },
        { signal: controller.signal },
      )
      if (!response.accepted) {
        throw new Error(response.message)
      }
      setApplyMessage(response.action === 'restart'
        ? 'Settings saved. Waiting for HomeSec to restart and activate them…'
        : 'Settings saved. Waiting for runtime activation…')
      const activated = await waitForConfigApply(response, controller.signal)
      queryClient.setQueryData(QUERY_KEYS.config, activated)
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runtimeStatus }),
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameras }),
      ])
      if (!controller.signal.aborted) {
        setApplyMessage('Saved settings are active.')
      }
    } catch (error) {
      if (timedOut) {
        setApplyError(new Error('Activation could not be confirmed before the timeout. Refresh status before retrying Apply.'))
        setApplyMessage('Settings remain saved. Activation has not been confirmed.')
      } else if (!controller.signal.aborted) {
        setApplyError(error)
        setApplyMessage('Settings remain saved. Activation has not been confirmed.')
      }
    } finally {
      clearTimeout(timer)
      if (timedOut || !controller.signal.aborted) {
        setApplyPending(false)
      }
    }
  }

  const conflict = [saveMutation.error, applyError].some(
    (error) => isAPIError(error) && error.status === 409
      && error.errorCode === 'CONFIG_VERSION_CONFLICT',
  )
  return {
    configQuery, saveConfig, refreshConfig, savePending: saveMutation.isPending,
    saveError: saveMutation.error, conflict,
    applyConfig, applyPending, applyError, applyMessage,
  }
}

export type ConfigSettings = ReturnType<typeof useConfigSettings>
