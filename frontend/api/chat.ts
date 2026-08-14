/**
 * API client for the chat assistant (/api/analytics/chat/*).
 * Deliberately not using the shared axios client for streaming — axios
 * doesn't stream cleanly in-browser — so this uses native fetch + ReadableStream,
 * the first streaming consumer in the codebase.
 */
import axios from 'axios'

import { apiBase } from '@/api/client'

export type ChatEventType =
  | 'text_delta'
  | 'tool_call'
  | 'tool_result'
  | 'ui_action'
  | 'confirm_action'
  | 'done'
  | 'error'

export interface ChatEvent {
  type: ChatEventType
  text?: string
  tool_name?: string
  tool_input?: Record<string, unknown>
  route?: string
  label?: string
  message?: string
  /** confirm_action only — pass back to confirmAction() to resolve this proposal. */
  confirmation_id?: string
}

const statusClient = axios.create({ baseURL: apiBase('/api/analytics') })

export interface ChatStatus {
  enabled: boolean
  /** Which ChatAgentBackend implementation is running, e.g. "tool_loop". */
  backend?: string
  /** Which model provider is active, e.g. "anthropic" or "openai_compatible". */
  provider?: string
  /** The actual model identifier in use, e.g. "claude-sonnet-5". */
  model?: string
}

export async function getChatStatus(): Promise<ChatStatus> {
  try {
    const { data } = await statusClient.get<ChatStatus>('/chat/status')
    return data
  } catch {
    return { enabled: false }
  }
}

export interface ChatConfirmResult {
  found: boolean
  approved?: boolean
  result?: string
  is_error?: boolean
}

/**
 * Resolves a confirm_action proposal (see ChatEvent.confirmation_id) —
 * approved:true actually executes the action outside the model loop;
 * approved:false discards it. found:false means it was already resolved or
 * the session no longer has it (not an error to surface to the user as one —
 * just stop showing the card as pending).
 */
export async function confirmAction(
  sessionId: string,
  confirmationId: string,
  approved: boolean,
): Promise<ChatConfirmResult> {
  const { data } = await statusClient.post<ChatConfirmResult>('/chat/confirm', {
    session_id: sessionId,
    confirmation_id: confirmationId,
    approved,
  })
  return data
}

export interface ChatPageContext {
  /** e.g. "/dashboard/builds/_" — usePathname()'s value, no query string. */
  pathname: string
  /** e.g. "?id=abc123" — searchParams.toString() with a leading "?", or "". */
  search: string
}

/**
 * Streams one chat turn. Yields each NormalizedEvent as the server sends it.
 * Parses newline-delimited SSE frames of the form `data: {...}\n\n`.
 *
 * pageContext is passive browser-awareness info (which dashboard route the
 * user is currently viewing) — never treated as part of the message itself;
 * the backend prepends it to the model's context separately. Omit it and the
 * agent simply has no page awareness for that turn.
 */
export async function* streamChat(
  sessionId: string,
  message: string,
  pageContext?: ChatPageContext,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(apiBase('/api/analytics/chat/stream'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      page_pathname: pageContext?.pathname,
      page_search: pageContext?.search,
    }),
    signal,
  })

  if (!res.ok || !res.body) {
    throw new Error(`Chat stream request failed: ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let separatorIndex: number
    // SSE frames are separated by a blank line ("\n\n")
    while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, separatorIndex)
      buffer = buffer.slice(separatorIndex + 2)

      for (const line of frame.split('\n')) {
        if (!line.startsWith('data: ')) continue
        const payload = line.slice(6)
        try {
          yield JSON.parse(payload) as ChatEvent
        } catch {
          // Ignore malformed frames rather than killing the whole stream.
        }
      }
    }
  }
}
