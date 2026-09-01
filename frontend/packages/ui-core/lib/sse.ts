/**
 * Parses a fetch Response body as newline-delimited SSE frames of the form
 * `data: {...}\n\n`, yielding each frame's JSON-decoded payload.
 *
 * Extracted out of api/chat.ts's streamChat() — the first (and, as of this
 * writing, only) streaming consumer in the codebase — so a second one
 * doesn't have to copy-paste or fork this parsing loop.
 */
export async function* parseSSEStream<T>(body: ReadableStream<Uint8Array>): AsyncGenerator<T> {
  const reader = body.getReader()
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
          yield JSON.parse(payload) as T
        } catch {
          // Ignore malformed frames rather than killing the whole stream.
        }
      }
    }
  }
}
