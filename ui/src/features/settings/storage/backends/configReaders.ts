export function readString(
  config: Record<string, unknown>,
  key: string,
  fallback: string,
): string {
  const value = config[key]
  return typeof value === 'string' ? value : fallback
}
