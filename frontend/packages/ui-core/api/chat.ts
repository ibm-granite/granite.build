/**
 * API client for the chat assistant (/api/analytics/chat/*).
 * Deliberately not using the shared axios client for streaming — axios
 * doesn't stream cleanly in-browser — so this uses native fetch + ReadableStream.
 */
import axios from 'axios'

import { apiBase } from './client'
import { parseSSEStream } from '../lib/sse'

export type ChatEventType =
  | 'text_delta'
  | 'tool_call'
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

/**
 * Extra request headers to attach to every chat call, supplied by the host app.
 *
 * Chat is the one API client here that a host cannot reach with its own axios
 * interceptor. Deployments whose per-user identity does not come from gbserver's
 * AuthMiddleware identify the caller with request headers instead (gb-ui's
 * sidecar, for one, sends an Authorization bearer token plus X-User-Email) and
 * normally attach them by interceptor on their own axios instance. This module
 * uses neither that instance nor, for streaming, axios at all: /chat/status,
 * /chat/stop and /chat/confirm go through the module-private `statusClient`
 * below, and /chat/stream uses native fetch because axios does not stream
 * cleanly in-browser.
 *
 * So without a hook here, every chat request arrives unidentified no matter what
 * the host configures, and the backend's resolve_identity() falls back to one
 * shared "standalone" identity for everybody — collapsing the per-identity rate
 * limit into a single bucket and, more seriously, collapsing the session scoping
 * that stops one user resolving another user's pending confirm_action.
 *
 * Call setChatHeaderProvider once during app startup; pass null to clear it. The
 * provider runs per request rather than once, so it always sees current
 * credentials, and it may be async for hosts that refresh a token first.
 */
export type ChatHeaderProvider = () =>
  | Record<string, string>
  | Promise<Record<string, string>>

let chatHeaderProvider: ChatHeaderProvider | null = null

export function setChatHeaderProvider(provider: ChatHeaderProvider | null): void {
  chatHeaderProvider = provider
}

/**
 * Resolves the host-supplied headers for a single request.
 *
 * A provider that throws or rejects contributes no headers rather than failing
 * the call: that degrades chat to exactly the unidentified behaviour it has when
 * no provider is installed, whereas propagating the error would take the whole
 * widget down over something like a transient token read.
 */
async function resolveChatHeaders(): Promise<Record<string, string>> {
  if (!chatHeaderProvider) return {}
  try {
    return (await chatHeaderProvider()) ?? {}
  } catch {
    return {}
  }
}

const statusClient = axios.create({ baseURL: apiBase('/api/analytics') })

// One interceptor covers /chat/status, /chat/stop and /chat/confirm. axios awaits
// async request interceptors, so a provider that refreshes a token still works.
statusClient.interceptors.request.use(async (config) => {
  const extra = await resolveChatHeaders()
  for (const [name, value] of Object.entries(extra)) {
    config.headers.set(name, value)
  }
  return config
})

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

export interface ChatStopResult {
  interrupted: boolean
}

/**
 * Requests server-side interruption of whatever turn is currently running
 * for this session (a model call or a tool call, e.g. a long-running
 * wait_for_build) — complements aborting the client's own fetch/read loop,
 * since that alone only stops the browser from reading further SSE frames,
 * not the backend work producing them.
 */
export async function stopChat(sessionId: string): Promise<ChatStopResult> {
  const { data } = await statusClient.post<ChatStopResult>('/chat/stop', {
    session_id: sessionId,
  })
  return data
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
 * Streams one chat turn. Yields each NormalizedEvent as the server sends it
 * (see lib/sse.ts's parseSSEStream for the actual frame parsing).
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
    headers: {
      ...(await resolveChatHeaders()),
      // Set last on purpose: the body below is always JSON, so a host provider
      // must not be able to change how the server reads it.
      'Content-Type': 'application/json',
    },
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

  yield* parseSSEStream<ChatEvent>(res.body)
}
