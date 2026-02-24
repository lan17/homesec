import { describe, expect, it } from 'vitest'

import {
  historyForSignature,
  popCursorHistory,
  pushCursorHistory,
  resetCursorHistory,
} from './cursorState'

describe('cursorState', () => {
  it('keeps filter signature scoped while moving across keyset pages', () => {
    // Given: Active filters represented by one signature and empty cursor history
    const filterSignature = 'camera=front_door&status=done&limit=25'
    const initialState = resetCursorHistory(filterSignature)

    // When: Advancing from first page to second and then third page
    const stateAfterSecondPage = pushCursorHistory(initialState, filterSignature, undefined)
    const stateAfterThirdPage = pushCursorHistory(stateAfterSecondPage, filterSignature, 'cursor-2')

    // Then: History should track prior cursors in order for deterministic previous navigation
    expect(historyForSignature(stateAfterThirdPage, filterSignature)).toEqual(['', 'cursor-2'])
  })

  it('rewinds previous cursor and keeps filters intact', () => {
    // Given: Cursor history built under a specific filter signature
    const filterSignature = 'camera=front_door&status=done&limit=25'
    const initialState = resetCursorHistory(filterSignature)
    const advanced = pushCursorHistory(initialState, filterSignature, undefined)

    // When: Navigating back to previous cursor page
    const popped = popCursorHistory(advanced, filterSignature)

    // Then: Previous cursor should be empty (return to first page), and history should shrink
    expect(popped.previousCursor).toBeUndefined()
    expect(historyForSignature(popped.state, filterSignature)).toEqual([])
  })

  it('drops stale cursor history when filter signature changes', () => {
    // Given: History created under one filter set
    const oldSignature = 'camera=front_door&status=done&limit=25'
    const newSignature = 'camera=garage&status=done&limit=25'
    const oldState = pushCursorHistory(resetCursorHistory(oldSignature), oldSignature, undefined)

    // When: Reading and popping cursor state under a different signature
    const history = historyForSignature(oldState, newSignature)
    const popped = popCursorHistory(oldState, newSignature)

    // Then: Stale history should not leak across filter changes
    expect(history).toEqual([])
    expect(popped.previousCursor).toBeUndefined()
    expect(historyForSignature(popped.state, newSignature)).toEqual([])
  })
})
