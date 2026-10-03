const REDACTED_PLACEHOLDER = '***redacted***'

export function isConfigObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function expectConfigObject(value: unknown, name: string): Record<string, unknown> {
  if (!isConfigObject(value)) {
    throw new Error(`${name} must be an object.`)
  }
  return value
}

export function expectConfigString(value: unknown, name: string): string {
  if (typeof value !== 'string') {
    throw new Error(`${name} must be a string.`)
  }
  return value
}

export function expectConfigBoolean(value: unknown, name: string): boolean {
  if (typeof value !== 'boolean') {
    throw new Error(`${name} must be a boolean.`)
  }
  return value
}

export function expectConfigStringList(value: unknown, name: string): string[] {
  if (!Array.isArray(value) || !value.every((item): item is string => typeof item === 'string')) {
    throw new Error(`${name} must be a list of strings.`)
  }
  return [...value]
}

function containsPlaceholder(value: unknown): boolean {
  if (typeof value === 'string') {
    return value.includes(REDACTED_PLACEHOLDER)
  }
  if (Array.isArray(value)) {
    return value.some(containsPlaceholder)
  }
  return isConfigObject(value) && Object.values(value).some(containsPlaceholder)
}

/** Build a merge patch, preserving unchanged redacted and advanced values. */
export function diffConfig(
  original: Record<string, unknown>,
  edited: Record<string, unknown>,
): Record<string, unknown> {
  const patch: Record<string, unknown> = {}
  for (const key of new Set([...Object.keys(original), ...Object.keys(edited)])) {
    if (!(key in edited)) {
      patch[key] = null
      continue
    }
    const before = original[key]
    const after = edited[key]
    if (JSON.stringify(before) === JSON.stringify(after)) {
      continue
    }
    if (isConfigObject(before) && isConfigObject(after)) {
      const nested = diffConfig(before, after)
      if (Object.keys(nested).length > 0) {
        patch[key] = nested
      }
      continue
    }
    if (containsPlaceholder(after)) {
      throw new Error('Redacted values cannot be changed here. Use the YAML configuration.');
    }
    patch[key] = after
  }
  return patch
}
