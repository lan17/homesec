export function isEnvReference(value: string): boolean {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.exec(value)?.[0] === value
}

/** Check a provider config or merge patch before it enters a request/cache. */
export function validateEnvReferences(config: Record<string, unknown>): void {
  for (const [key, value] of Object.entries(config)) {
    if (key.endsWith('_env') && value !== undefined && value !== null && value !== '') {
      if (typeof value !== 'string' || !isEnvReference(value)) {
        throw new Error('Credentials must reference environment variable names.')
      }
    }
    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
      validateEnvReferences(value as Record<string, unknown>)
    }
  }
}
