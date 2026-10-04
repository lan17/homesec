import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useLocation, useNavigate } from 'react-router-dom'

import {
  HomeSecApiClient,
  isAPIError,
  isUnauthorizedAPIError,
  persistRuntimeAuthSessionReady,
  runtimeAuthTokenProvider,
  runtimeServerBaseUrlProvider,
} from '../../api/client'
import type { ClientServerBaseUrlProvider } from '../../api/serverBaseUrlProvider'
import {
  clearPersistedRuntimeAuthSessionReady,
  isRuntimeAuthSessionReady,
  type AuthTokenProvider,
} from '../../api/tokenProvider'
import { Button } from '../../components/ui/Button'
import { validateNativeSetupServerUrl } from './nativeSetup'
import './nativeSetup.css'

type NativeSetupStep = 'server' | 'token'
type NativeSetupOperation = 'checking-server' | 'validating-token' | 'persisting'

export interface NativeSetupPageProps {
  authTokenProvider?: AuthTokenProvider
  createClient?: (baseUrl: string) => HomeSecApiClient
  serverBaseUrlProvider?: ClientServerBaseUrlProvider
}

function isAuthFailure(error: unknown): boolean {
  return isUnauthorizedAPIError(error) || (isAPIError(error) && error.status === 403)
}

function describeServerCheckError(error: unknown): string {
  if (isAPIError(error)) {
    return `Server responded with HTTP ${error.status}. Check the URL and try again.`
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return `Unable to reach HomeSec: ${error.message}`
  }
  return 'Unable to reach HomeSec. Check the server URL and network connection.'
}

function describeTokenError(error: unknown): string {
  if (isAuthFailure(error)) {
    return 'API token was rejected. Paste the HomeSec API token and try again.'
  }
  if (isAPIError(error)) {
    return `Token validation failed with HTTP ${error.status}. Check server status and try again.`
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return `Unable to validate token: ${error.message}`
  }
  return 'Unable to validate the API token. Try again.'
}

function nativeSetupReturnTo(state: unknown): string {
  if (!state || typeof state !== 'object') {
    return '/live'
  }

  const returnTo = (state as { nativeSetupReturnTo?: unknown }).nativeSetupReturnTo
  if (
    typeof returnTo !== 'string' ||
    !returnTo.startsWith('/') ||
    returnTo.startsWith('//') ||
    returnTo === '/native-setup'
  ) {
    return '/live'
  }

  return returnTo
}

export function NativeSetupPage({
  authTokenProvider = runtimeAuthTokenProvider,
  createClient = (baseUrl: string) => new HomeSecApiClient(baseUrl),
  serverBaseUrlProvider = runtimeServerBaseUrlProvider,
}: NativeSetupPageProps = {}) {
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const [serverUrl, setServerUrl] = useState('')
  const [apiToken, setApiToken] = useState('')
  const [validatedServerUrl, setValidatedServerUrl] = useState<string | null>(null)
  const [isPlainHttp, setIsPlainHttp] = useState(false)
  const [authDisabled, setAuthDisabled] = useState(false)
  const [serverError, setServerError] = useState<string | null>(null)
  const [tokenError, setTokenError] = useState<string | null>(null)
  const [step, setStep] = useState<NativeSetupStep>('server')
  const [operation, setOperation] = useState<NativeSetupOperation | null>(null)
  const operationControllerRef = useRef<{
    controller: AbortController
    phase: NativeSetupOperation
  } | null>(null)
  const isCheckingServer = operation === 'checking-server'
  const isSaving = operation === 'validating-token' || operation === 'persisting'
  const canReturnToApp = Boolean(serverBaseUrlProvider.getBaseUrlSync()) && isRuntimeAuthSessionReady()

  useEffect(() => () => {
    operationControllerRef.current?.controller.abort()
    operationControllerRef.current = null
  }, [])

  function cancelSetup(): void {
    const currentOperation = operationControllerRef.current
    // Enforce the persistence boundary even before React updates the button.
    if (currentOperation?.phase === 'persisting') return
    currentOperation?.controller.abort()
    operationControllerRef.current = null
    setOperation(null)
    if (canReturnToApp) {
      navigate(nativeSetupReturnTo(location.state), { replace: true })
    }
  }

  function handleServerUrlChange(value: string): void {
    setServerUrl(value)
    setApiToken('')
    setValidatedServerUrl(null)
    setIsPlainHttp(false)
    setAuthDisabled(false)
    setServerError(null)
    setTokenError(null)
    setStep('server')
  }

  async function checkServer(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    const validation = validateNativeSetupServerUrl(serverUrl)
    if (!validation.ok) {
      setServerError(validation.message)
      setValidatedServerUrl(null)
      setStep('server')
      return
    }

    const controller = new AbortController()
    operationControllerRef.current?.controller.abort()
    operationControllerRef.current = { controller, phase: 'checking-server' }
    setOperation('checking-server')
    setServerError(null)
    setTokenError(null)
    setIsPlainHttp(false)
    setAuthDisabled(false)
    try {
      const client = createClient(validation.value.serverBaseUrl)
      await client.getHealth({ apiKey: null, signal: controller.signal })
      if (controller.signal.aborted) return

      let acceptsUnauthenticatedRequests = false
      try {
        await client.getSetupStatus({ apiKey: null, signal: controller.signal })
        acceptsUnauthenticatedRequests = true
      } catch (error) {
        if (!isAuthFailure(error)) {
          acceptsUnauthenticatedRequests = false
        }
      }
      if (controller.signal.aborted) return

      setValidatedServerUrl(validation.value.serverBaseUrl)
      setServerUrl(validation.value.serverBaseUrl)
      setIsPlainHttp(validation.value.isPlainHttp)
      setAuthDisabled(acceptsUnauthenticatedRequests)
      if (acceptsUnauthenticatedRequests) {
        setApiToken('')
      }
      setStep('token')
    } catch (error) {
      if (controller.signal.aborted) return
      setValidatedServerUrl(null)
      setIsPlainHttp(false)
      setStep('server')
      setServerError(describeServerCheckError(error))
    } finally {
      if (operationControllerRef.current?.controller === controller) {
        operationControllerRef.current = null
        setOperation(null)
      }
    }
  }

  async function saveAndContinue(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    const apiKey = apiToken.trim()
    if (!validatedServerUrl) {
      setTokenError('Check the server URL before continuing.')
      return
    }
    if (!authDisabled && !apiKey) {
      setTokenError('API token is required.')
      return
    }

    const controller = new AbortController()
    operationControllerRef.current?.controller.abort()
    operationControllerRef.current = { controller, phase: 'validating-token' }
    setOperation('validating-token')
    setTokenError(null)
    try {
      if (!authDisabled && apiKey) {
        await createClient(validatedServerUrl).getSetupStatus({ apiKey, signal: controller.signal })
      }
      if (controller.signal.aborted) return
      const currentOperation = operationControllerRef.current
      if (currentOperation?.controller !== controller) return
      currentOperation.phase = 'persisting'
      setOperation('persisting')
      await queryClient.cancelQueries()
      queryClient.clear()
      await authTokenProvider.clearToken()
      await clearPersistedRuntimeAuthSessionReady()
      await serverBaseUrlProvider.setBaseUrl(validatedServerUrl)
      await authTokenProvider.setToken(authDisabled ? null : apiKey || null)
      await persistRuntimeAuthSessionReady({ persistAuthDisabled: authDisabled })
      queryClient.clear()
      if (!controller.signal.aborted) {
        navigate(nativeSetupReturnTo(location.state), { replace: true })
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        setTokenError(describeTokenError(error))
      }
    } finally {
      if (operationControllerRef.current?.controller === controller) {
        operationControllerRef.current = null
        setOperation(null)
      }
    }
  }

  const tokenInputDisabled = step !== 'token' || authDisabled || isSaving || isCheckingServer
  const canSave = step === 'token' && !isCheckingServer && !isSaving
  const showCancel = canReturnToApp || operation !== null

  return (
    <main className="native-setup-page">
      <section className="native-setup-panel" aria-labelledby="native-setup-title">
        <header className="native-setup-panel__header">
          <p className="subtle">iOS setup</p>
          <h1 id="native-setup-title" className="native-setup-panel__title">
            Connect to HomeSec
          </h1>
        </header>

        <form className="native-setup-form" onSubmit={checkServer} noValidate>
          <div className="native-setup-form__field">
            <label className="field-label" htmlFor="native-server-url">
              Server URL
            </label>
            <input
              id="native-server-url"
              className="input"
              type="url"
              inputMode="url"
              autoCapitalize="none"
              autoCorrect="off"
              placeholder="https://homesec.example.com"
              value={serverUrl}
              onChange={(event) => handleServerUrlChange(event.target.value)}
              disabled={isCheckingServer || isSaving}
            />
          </div>
          {serverError ? <p className="error-text">{serverError}</p> : null}
          <Button type="submit" disabled={isCheckingServer || isSaving}>
            {isCheckingServer ? 'Checking...' : 'Check server'}
          </Button>
        </form>

        {step === 'token' ? (
          <div className="native-setup-status" role="status">
            Server reachable
          </div>
        ) : null}

        {isPlainHttp ? (
          <div className="native-setup-warning" role="alert">
            Plain HTTP is visible on the network. Prefer HTTPS or VPN for iOS access.
          </div>
        ) : null}

        {authDisabled ? (
          <div className="native-setup-warning native-setup-warning--strong" role="alert">
            This server accepted setup requests without an API token. Authentication appears
            disabled, so the token cannot be verified.
          </div>
        ) : null}

        <form className="native-setup-form" onSubmit={saveAndContinue}>
          <div className="native-setup-form__field">
            <label className="field-label" htmlFor="native-api-token">
              API token
            </label>
            <input
              id="native-api-token"
              className="input"
              type="password"
              autoComplete="off"
              placeholder={authDisabled ? 'Optional while auth is disabled' : 'Paste API token'}
              value={apiToken}
              onChange={(event) => setApiToken(event.target.value)}
              disabled={tokenInputDisabled}
            />
          </div>
          {tokenError ? <p className="error-text">{tokenError}</p> : null}
          <div className="inline-form__actions">
            <Button type="submit" disabled={!canSave}>
              {isSaving ? 'Saving...' : 'Save and continue'}
            </Button>
            {showCancel ? (
              <Button
                type="button"
                variant="ghost"
                disabled={operation === 'persisting'}
                onClick={cancelSetup}
              >
                Cancel
              </Button>
            ) : null}
          </div>
        </form>
      </section>
    </main>
  )
}
