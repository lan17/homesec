import { describe, expect, it } from 'vitest'

import { isEnvReference, validateEnvReferences } from './envReferences'

describe('environment references', () => {
  it.each(['KEY\n', 'KEY\r', ' KEY', 'KEY ', 'KEY\t', 'KEY\u2028', 'KEY\u2029'])('rejects the entire malformed reference %j', (value) => {
    // Given: A plausible env name contains whitespace or a trailing line terminator
    // When/Then: The full input must be a reference, and errors do not reflect the submitted value
    expect(isEnvReference(value)).toBe(false)
    expect(() => validateEnvReferences({ auth: { password_env: value } })).toThrow('environment variable names')
  })

  it('accepts env names and absent optional auth references', () => {
    // Given: Valid and explicitly cleared optional credential references
    // When/Then: The same shared validation accepts both at the provider config boundary
    expect(isEnvReference('_KEY_1')).toBe(true)
    expect(() => validateEnvReferences({ api_key_env: 'KEY_1', auth: { username_env: null, password_env: '' } })).not.toThrow()
  })
})
