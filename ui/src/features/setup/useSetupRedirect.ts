import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'

import { useSetupStatusQuery } from '../../api/hooks/useSetupStatusQuery'
import { WIZARD_STATE_STORAGE_KEY } from './useWizardState'

interface PersistedWizardStateForRedirect {
  completedSteps?: unknown
}

export interface SetupRedirectState {
  shouldRedirect: boolean
  isChecking: boolean
}

function hasPersistedWizardProgress(): boolean {
  if (typeof window === 'undefined') {
    return false
  }
  const raw = window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)
  if (!raw) {
    return false
  }

  try {
    const parsed = JSON.parse(raw) as PersistedWizardStateForRedirect
    return Array.isArray(parsed.completedSteps) && parsed.completedSteps.length > 0
  } catch {
    return false
  }
}

export function useSetupRedirect(): SetupRedirectState {
  const navigate = useNavigate()
  const setupStatusQuery = useSetupStatusQuery()
  const isChecking = setupStatusQuery.isPending
  const shouldRedirect =
    setupStatusQuery.data?.state === 'fresh'
    && setupStatusQuery.error == null
    && !hasPersistedWizardProgress()

  useEffect(() => {
    if (shouldRedirect) {
      navigate('/setup', { replace: true })
    }
  }, [navigate, shouldRedirect])

  return {
    shouldRedirect,
    isChecking,
  }
}
