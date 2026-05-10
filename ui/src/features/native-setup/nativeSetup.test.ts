import { describe, expect, it } from 'vitest'

import { validateNativeSetupServerUrl } from './nativeSetup'

describe('validateNativeSetupServerUrl', () => {
  it('normalizes HTTPS and LAN URLs while rejecting unsupported input', () => {
    // Given: Server URL candidates from the native setup form
    const httpsUrl = ' https://homesec.example.com/// '
    const lanUrl = 'http://192.168.1.10:8081/'
    const missingScheme = 'homesec.local:8081'

    // When: Validating each value
    const httpsResult = validateNativeSetupServerUrl(httpsUrl)
    const lanResult = validateNativeSetupServerUrl(lanUrl)
    const missingSchemeResult = validateNativeSetupServerUrl(missingScheme)

    // Then: Supported URLs normalize and invalid input returns an actionable message
    expect(httpsResult).toEqual({
      ok: true,
      value: {
        serverBaseUrl: 'https://homesec.example.com',
        isPlainHttp: false,
      },
    })
    expect(lanResult).toEqual({
      ok: true,
      value: {
        serverBaseUrl: 'http://192.168.1.10:8081',
        isPlainHttp: true,
      },
    })
    expect(missingSchemeResult).toEqual({
      ok: false,
      message: 'Only http:// and https:// server URLs are supported.',
    })
  })

  it('rejects blank server URLs', () => {
    // Given: A blank server URL
    const input = '   '

    // When: Validating setup input
    const result = validateNativeSetupServerUrl(input)

    // Then: The setup flow asks for a server URL
    expect(result).toEqual({
      ok: false,
      message: 'Enter the HomeSec server URL.',
    })
  })
})
