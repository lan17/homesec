export function readString(
  config: Record<string, unknown>,
  key: string,
  fallback: string,
): string {
  const value = config[key]
  return typeof value === 'string' ? value : fallback
}

export function readNumber(
  config: Record<string, unknown>,
  key: string,
  fallback: number,
): number {
  const value = config[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

export function readBoolean(
  config: Record<string, unknown>,
  key: string,
  fallback: boolean,
): boolean {
  const value = config[key]
  return typeof value === 'boolean' ? value : fallback
}
