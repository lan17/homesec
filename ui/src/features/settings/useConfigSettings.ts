import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { apiClient, isAPIError, type ConfigPatch, type ConfigSnapshot } from '../../api/client'
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
    setApplyMessage(config.apply_required === 'none' ? 'Saved settings are active.'
      : 'Settings saved. Apply the saved changes to activate them.')
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
      if (controller.signal.aborted) { return }
      await queryClient.cancelQueries({ queryKey: QUERY_KEYS.config })
      if (controller.signal.aborted) { return }
      const observed = queryClient.getQueryData<ConfigSnapshot>(QUERY_KEYS.config)
      const interveningSave = observed && observed.saved_config_version !== config.saved_config_version
        && observed.saved_config_version !== activated.saved_config_version
      if (interveningSave) {
        await queryClient.refetchQueries({ queryKey: QUERY_KEYS.config })
        if (controller.signal.aborted) { return }
      } else {
        queryClient.setQueryData(QUERY_KEYS.config, activated)
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runtimeStatus }),
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameras }),
      ])
      if (!controller.signal.aborted) {
        const latest = queryClient.getQueryData<ConfigSnapshot>(QUERY_KEYS.config)
        const targetIsActive = latest?.saved_config_version === activated.saved_config_version
          && latest.active_config_version === activated.active_config_version && latest.apply_required === 'none'
        setApplyMessage(targetIsActive ? 'Saved settings are active.'
          : 'Activation was confirmed, but saved settings changed. Apply the latest saved revision separately.')
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
