export interface CursorHistoryState {
  signature: string
  history: string[]
}

export function historyForSignature(
  state: CursorHistoryState,
  signature: string,
): string[] {
  if (state.signature !== signature) {
    return []
  }
  return state.history
}

export function resetCursorHistory(signature: string): CursorHistoryState {
  return {
    signature,
    history: [],
  }
}

export function pushCursorHistory(
  state: CursorHistoryState,
  signature: string,
  currentCursor: string | null | undefined,
): CursorHistoryState {
  const history = historyForSignature(state, signature)
  return {
    signature,
    history: [...history, currentCursor ?? ''],
  }
}

export function popCursorHistory(
  state: CursorHistoryState,
  signature: string,
): { state: CursorHistoryState; previousCursor: string | undefined } {
  const history = historyForSignature(state, signature)
  if (history.length === 0) {
    return {
      state: {
        signature,
        history: [],
      },
      previousCursor: undefined,
    }
  }

  const previousCursor = history[history.length - 1] ?? ''
  return {
    state: {
      signature,
      history: history.slice(0, -1),
    },
    previousCursor: previousCursor || undefined,
  }
}
