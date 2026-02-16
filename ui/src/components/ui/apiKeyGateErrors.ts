export function toApiKeyGateActionErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message
  }
  return 'Unable to apply API key. Please try again.'
}
