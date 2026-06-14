import { describe, expect, it } from 'vitest'

import { validateNativeSetupServerUrl } from './nativeSetup'

describe('validateNativeSetupServerUrl', () => {
  it('normalizes HTTPS and LAN URLs while rejecting unsupported input', () => {
    // Given: Server URL candidates from the native setup form
    const httpsUrl = ' https://homesec.example.com/// '
    const pathUrl = 'https://homesec.example.com/homesec///'
    const lanUrl = 'http://192.168.1.10:8081/'
    const localHostUrl = 'http://homesec.local:8081/'
    const singleLabelUrl = 'http://homesec:8081/'
    const missingScheme = 'homesec.local:8081'

    // When: Validating each value
    const httpsResult = validateNativeSetupServerUrl(httpsUrl)
    const pathResult = validateNativeSetupServerUrl(pathUrl)
    const lanResult = validateNativeSetupServerUrl(lanUrl)
    const localHostResult = validateNativeSetupServerUrl(localHostUrl)
    const singleLabelResult = validateNativeSetupServerUrl(singleLabelUrl)
    const missingSchemeResult = validateNativeSetupServerUrl(missingScheme)

    // Then: Supported URLs normalize and invalid input returns an actionable message
    expect(httpsResult).toEqual({
      ok: true,
      value: {
        serverBaseUrl: 'https://homesec.example.com',
        isPlainHttp: false,
      },
    })
    expect(pathResult).toEqual({
      ok: true,
      value: {
        serverBaseUrl: 'https://homesec.example.com/homesec',
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
    expect(localHostResult).toEqual({
      ok: true,
      value: {
        serverBaseUrl: 'http://homesec.local:8081',
        isPlainHttp: true,
      },
    })
    expect(singleLabelResult).toEqual({
      ok: true,
      value: {
        serverBaseUrl: 'http://homesec:8081',
        isPlainHttp: true,
      },
    })
    expect(missingSchemeResult).toEqual({
      ok: false,
      message: 'Only http:// and https:// server URLs are supported.',
    })
  })

  it('rejects public plain-HTTP hosts', () => {
    // Given: Plain-HTTP URLs outside the local network
    const publicHostname = 'http://homesec.example.com'
    const publicIp = 'http://8.8.8.8:8081'

    // When: Validating setup input
    const hostnameResult = validateNativeSetupServerUrl(publicHostname)
    const ipResult = validateNativeSetupServerUrl(publicIp)

    // Then: The setup flow requires HTTPS for public hosts
    expect(hostnameResult).toEqual({
      ok: false,
      message: 'Plain HTTP is only supported for local network HomeSec servers. Use HTTPS for public hosts.',
    })
    expect(ipResult).toEqual({
      ok: false,
      message: 'Plain HTTP is only supported for local network HomeSec servers. Use HTTPS for public hosts.',
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
