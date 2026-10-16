"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";

import { Button, InlineNotification, Tag, TextInput, Tile } from "@carbon/react";
import { ChatBot, Close, Send, StopFilledAlt } from "@carbon/icons-react";

import { ChatStatus, confirmAction, getChatStatus, stopChat, streamChat } from "../api/chat";
import styles from "./ChatWidget.module.scss";

interface UiActionState {
  id: string;
  route: string;
  label: string;
  status: "pending" | "confirmed" | "navigated" | "dismissed";
}

interface ConfirmActionState {
  id: string;
  confirmationId: string;
  actionName: string;
  toolInput: Record<string, unknown>;
  label: string;
  status: "pending" | "submitting" | "approved" | "declined";
  result?: string;
  isError?: boolean;
}

const MIN_WIDTH_PX = 320;
const DEFAULT_WIDTH_PX = 448; // 28rem

function normalizeRoutePath(path: string): string {
  const withoutQuery = path.split("?")[0];
  return withoutQuery.length > 1 && withoutQuery.endsWith("/") ? withoutQuery.slice(0, -1) : withoutQuery;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  uiActions: UiActionState[];
  confirmActions: ConfirmActionState[];
  isStreaming?: boolean;
  /** Name of the tool currently running for this message, if any — surfaced
   * so a tool-only turn (no text_delta in between calls) shows what's
   * happening instead of a bare, uninformative spinner. */
  activeToolCall?: string;
}

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
}

function LoadingDots() {
  return (
    <span className={styles.loadingDots}>
      <span className={styles.loadingDot} />
      <span className={styles.loadingDot} />
      <span className={styles.loadingDot} />
    </span>
  );
}

export function ChatWidget() {
  const router = useRouter();
  const pathname = usePathname();
  const [status, setStatus] = useState<ChatStatus | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [width, setWidth] = useState(DEFAULT_WIDTH_PX);
  const sessionIdRef = useRef<string>(newId());
  const scrollRef = useRef<HTMLDivElement>(null);
  const dragStateRef = useRef<{ startX: number; startWidth: number } | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    getChatStatus().then(setStatus);
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, isOpen]);

  // The panel is anchored to the bottom-right corner, so dragging its left
  // edge leftward should grow it — width tracks the drag on a window-level
  // listener rather than the CSS `resize` property, since that property's
  // native grip renders in the bottom-right corner, directly on top of the
  // Send button.
  useEffect(() => {
    function handlePointerMove(e: PointerEvent) {
      const drag = dragStateRef.current;
      if (!drag) return;
      const next = drag.startWidth + (drag.startX - e.clientX);
      const maxWidth = window.innerWidth - 48;
      setWidth(Math.min(Math.max(next, MIN_WIDTH_PX), maxWidth));
    }
    function handlePointerUp() {
      dragStateRef.current = null;
    }
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, []);

  // Once the URL actually lands on a proposed page, flip its card from the
  // transient "Navigating…" state to a final "Navigated" one.
  useEffect(() => {
    const currentPath = normalizeRoutePath(pathname ?? "");
    if (!currentPath) return;
    setMessages((prev) =>
      prev.map((m) => ({
        ...m,
        uiActions: m.uiActions.map((a) =>
          a.status === "confirmed" && normalizeRoutePath(a.route) === currentPath ? { ...a, status: "navigated" } : a
        ),
      }))
    );
  }, [pathname]);

  // Gate visibility: silently disappear when no model provider is configured
  // (neither Anthropic credentials nor an OpenAI-compatible endpoint). Also
  // render nothing while the status check is still in flight, to avoid a
  // flash of the launcher.
  if (!status?.enabled) return null;

  function updateMessage(id: string, updater: (m: ChatMessage) => ChatMessage) {
    setMessages((prev) => prev.map((m) => (m.id === id ? updater(m) : m)));
  }

  function setActionStatus(messageId: string, actionId: string, status: UiActionState["status"]) {
    updateMessage(messageId, (m) => ({
      ...m,
      uiActions: m.uiActions.map((a) => (a.id === actionId ? { ...a, status } : a)),
    }));
  }

  function updateConfirmAction(messageId: string, confirmId: string, updater: (a: ConfirmActionState) => ConfirmActionState) {
    updateMessage(messageId, (m) => ({
      ...m,
      confirmActions: m.confirmActions.map((a) => (a.id === confirmId ? updater(a) : a)),
    }));
  }

  async function handleApprove(messageId: string, action: ConfirmActionState) {
    updateConfirmAction(messageId, action.id, (a) => ({ ...a, status: "submitting" }));
    try {
      const outcome = await confirmAction(sessionIdRef.current, action.confirmationId, true);
      updateConfirmAction(messageId, action.id, (a) => ({
        ...a,
        status: "approved",
        result: outcome.found ? outcome.result : "This proposal is no longer available.",
        isError: outcome.found ? outcome.is_error : true,
      }));
    } catch (err) {
      updateConfirmAction(messageId, action.id, (a) => ({
        ...a,
        status: "approved",
        result: err instanceof Error ? err.message : "Failed to submit the confirmation.",
        isError: true,
      }));
    }
  }

  async function handleDecline(messageId: string, action: ConfirmActionState) {
    updateConfirmAction(messageId, action.id, (a) => ({ ...a, status: "submitting" }));
    try {
      await confirmAction(sessionIdRef.current, action.confirmationId, false);
    } catch {
      // Declining is best-effort — the pending proposal simply expires with
      // the session if this fails, so there's nothing the user needs to see.
    }
    updateConfirmAction(messageId, action.id, (a) => ({ ...a, status: "declined" }));
  }

  // Drops the trailing placeholder if it never received anything, otherwise
  // just stops its loading dots. Safe to call more than once for the same id.
  function finalizePlaceholder(id: string) {
    setMessages((prev) => {
      const withoutEmpty = prev.filter(
        (m) => !(m.id === id && m.text.length === 0 && m.uiActions.length === 0 && m.confirmActions.length === 0)
      );
      return withoutEmpty.map((m) => (m.id === id ? { ...m, isStreaming: false } : m));
    });
  }

  async function handleStop() {
    // Aborting the client's own fetch/read loop only stops this browser
    // from reading further SSE frames — it doesn't stop the backend work
    // producing them (a model call, or a tool call like wait_for_build that
    // can run for up to 30 minutes). /chat/stop asks the server to
    // interrupt that work too; best-effort, since the abort() above already
    // unblocks the composer regardless of whether this call succeeds.
    abortControllerRef.current?.abort();
    try {
      await stopChat(sessionIdRef.current);
    } catch {
      // Nothing actionable for the user here — the client-side stream is
      // already stopped either way.
    }
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    setError(null);

    const userMessage: ChatMessage = { id: newId(), role: "user", text, uiActions: [], confirmActions: [] };
    let activeId = newId();
    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: activeId, role: "assistant", text: "", uiActions: [], confirmActions: [], isStreaming: true },
    ]);
    setIsStreaming(true);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      // Read window.location directly rather than next/navigation's
      // useSearchParams() — that hook requires a <Suspense> boundary on
      // statically-exported pages, which we'd otherwise need to add
      // everywhere ChatWidget is mounted (i.e. every page, via ClientShell).
      // Reading it here, inside an event handler, is plain client-only
      // browser API access with none of that ceremony, and gives the
      // freshest value anyway (exactly the page the user was on when they
      // hit send, not a stale render).
      const pageContext = { pathname: pathname ?? "", search: window.location.search };

      // The SDK yields complete text blocks, not token deltas, and a turn
      // can contain several of them around tool calls (e.g. a preamble
      // before a tool call, then the real answer after). Each text_delta is
      // already a finished block, so finalize it as its own message right
      // away and open a fresh placeholder for whatever comes next — a
      // preamble and the real answer show up as two distinct messages
      // instead of one bubble that appears to change after the fact.
      for await (const event of streamChat(sessionIdRef.current, text, pageContext, abortController.signal)) {
        if (event.type === "text_delta" && event.text) {
          const finishedId = activeId;
          const finishedText = event.text;
          updateMessage(finishedId, (m) => ({ ...m, text: finishedText, isStreaming: false }));

          activeId = newId();
          setMessages((prev) => [
            ...prev,
            { id: activeId, role: "assistant", text: "", uiActions: [], confirmActions: [], isStreaming: true },
          ]);
        } else if (event.type === "tool_call" && event.tool_name) {
          updateMessage(activeId, (m) => ({ ...m, activeToolCall: event.tool_name }));
        } else if (event.type === "ui_action" && event.route && event.label) {
          updateMessage(activeId, (m) => ({
            ...m,
            uiActions: [...m.uiActions, { id: newId(), route: event.route!, label: event.label!, status: "pending" }],
          }));
        } else if (event.type === "confirm_action" && event.confirmation_id && event.tool_name) {
          updateMessage(activeId, (m) => ({
            ...m,
            confirmActions: [
              ...m.confirmActions,
              {
                id: newId(),
                confirmationId: event.confirmation_id!,
                actionName: event.tool_name!,
                toolInput: event.tool_input ?? {},
                label: event.label ?? `Proposed action: ${event.tool_name}`,
                status: "pending",
              },
            ],
          }));
        } else if (event.type === "error") {
          setError(event.message ?? "The chat assistant hit an error.");
          finalizePlaceholder(activeId);
        } else if (event.type === "done") {
          finalizePlaceholder(activeId);
        }
      }
    } catch (err) {
      // The user's own Stop click aborts the fetch — that's success, not a
      // failure to surface as a scary error banner.
      const wasUserInitiatedStop = err instanceof DOMException && err.name === "AbortError";
      if (!wasUserInitiatedStop) {
        setError(err instanceof Error ? err.message : "Chat request failed.");
      }
      finalizePlaceholder(activeId);
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
      // Safety net: if the stream ended without ever finalizing the trailing
      // placeholder (e.g. the connection dropped with no "done"/error
      // event), don't leave its loading dots stuck on screen forever.
      finalizePlaceholder(activeId);
    }
  }

  return (
    <div style={{ position: "fixed", bottom: "1.5rem", right: "1.5rem", zIndex: 8000 }}>
      {isOpen ? (
        <div
          style={{
            position: "relative",
            width: `${width}px`,
            height: "32rem",
            maxWidth: "calc(100vw - 3rem)",
            display: "flex",
            flexDirection: "column",
            background: "var(--cds-layer)",
            border: "1px solid var(--cds-border-subtle-01)",
            borderRadius: "8px",
            boxShadow: "0 4px 16px rgba(0, 0, 0, 0.25)",
            overflow: "hidden",
          }}
        >
          <div
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize chat window"
            onPointerDown={(e) => {
              dragStateRef.current = { startX: e.clientX, startWidth: width };
            }}
            className={styles.resizeHandle}
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "0.75rem 1rem",
              borderBottom: "1px solid var(--cds-border-subtle-01)",
            }}
          >
            <span style={{ fontSize: "0.875rem", fontWeight: 600 }}>granite.build assistant</span>
            <Button
              kind="ghost"
              size="sm"
              hasIconOnly
              iconDescription="Close chat"
              renderIcon={Close}
              onClick={() => setIsOpen(false)}
            />
          </div>

          <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
            {messages.length === 0 && (
              <>
                <p style={{ fontSize: "0.875rem", color: "var(--cds-text-secondary)" }}>
                  Ask about builds, artifacts, or data pipelines — I can look things up and point you to the
                  right page, but I can&apos;t change anything without your confirmation.
                </p>
                {status?.model && (
                  <p
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--cds-text-secondary)",
                      fontFamily: "var(--cds-code-01-font-family, monospace)",
                    }}
                  >
                    Model: {status.model} · Harness: {status.backend} ({status.provider})
                  </p>
                )}
              </>
            )}
            {messages.map((message) => {
              const hasContent =
                message.text.length > 0 || message.uiActions.length > 0 || message.confirmActions.length > 0;
              // Only show the animated-dots placeholder bubble while we're
              // genuinely waiting on the first bit of a response — once either
              // real text or a ui_action has arrived, drop it instead of
              // leaving an empty bubble stuck on screen for turns that are
              // tool-only.
              const showLoadingBubble = message.role === "assistant" && message.isStreaming && !hasContent;
              return (
                <div key={message.id} style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {(message.text || showLoadingBubble) && (
                    <div
                      style={{
                        alignSelf: message.role === "user" ? "flex-end" : "flex-start",
                        maxWidth: message.role === "user" ? "85%" : "95%",
                        minWidth: 0,
                        background: message.role === "user" ? "#d0e2ff" : "var(--cds-layer-02)",
                        color: message.role === "user" ? "#161616" : undefined,
                        border: message.role === "user" ? "none" : "1px solid var(--cds-border-subtle-01)",
                        borderRadius: "8px",
                        padding: "0.5rem 0.75rem",
                        fontSize: "0.875rem",
                        whiteSpace: message.role === "user" ? "pre-wrap" : undefined,
                      }}
                    >
                      {message.text ? (
                        message.role === "assistant" ? (
                          <div className={styles.markdown}>
                            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{message.text}</ReactMarkdown>
                          </div>
                        ) : (
                          message.text
                        )
                      ) : message.activeToolCall ? (
                        <span style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--cds-text-secondary)" }}>
                          Calling <code>{message.activeToolCall}</code>
                          <LoadingDots />
                        </span>
                      ) : (
                        <LoadingDots />
                      )}
                    </div>
                  )}
                  {message.uiActions.map((action) => (
                    <Tile
                      key={action.id}
                      className={styles.actionTile}
                      style={{
                        maxWidth: "85%",
                        border: "1px solid var(--cds-border-subtle-01)",
                      }}
                    >
                      <p className={styles.actionLabel}>{action.label}</p>
                      {action.status === "pending" && (
                        <div className={styles.actionButtonRow}>
                          <Button
                            kind="primary"
                            size="sm"
                            onClick={() => {
                              // If we're already on the target page (e.g. a
                              // second navigation card for the same page),
                              // there's no pathname change to detect later —
                              // mark it navigated immediately.
                              const alreadyThere = normalizeRoutePath(action.route) === normalizeRoutePath(pathname ?? "");
                              setActionStatus(message.id, action.id, alreadyThere ? "navigated" : "confirmed");
                              router.push(action.route);
                            }}
                          >
                            Take me there
                          </Button>
                          <Button
                            kind="ghost"
                            size="sm"
                            onClick={() => setActionStatus(message.id, action.id, "dismissed")}
                          >
                            No thanks
                          </Button>
                        </div>
                      )}
                      {action.status === "confirmed" && <Tag type="green" size="sm">Navigating…</Tag>}
                      {action.status === "navigated" && <Tag type="gray" size="sm">Navigated</Tag>}
                      {action.status === "dismissed" && <Tag type="gray" size="sm">Dismissed</Tag>}
                    </Tile>
                  ))}
                  {message.confirmActions.map((action) => (
                    <Tile
                      key={action.id}
                      className={styles.actionTile}
                      style={{
                        maxWidth: "95%",
                        minWidth: 0,
                        border: "1px solid var(--cds-support-warning, #f1c21b)",
                      }}
                    >
                      <p className={styles.actionLabel} style={{ fontWeight: 600 }}>
                        {action.label}
                      </p>
                      {/* Show the real args (e.g. build_start's file_content is actual
                          YAML) rather than asking the user to approve blind. */}
                      <pre
                        style={{
                          fontSize: "0.75rem",
                          background: "var(--cds-layer-02)",
                          padding: "0.5rem",
                          borderRadius: "4px",
                          maxHeight: "10rem",
                          overflow: "auto",
                          margin: "0 0 0.5rem",
                          whiteSpace: "pre-wrap",
                          wordBreak: "break-word",
                        }}
                      >
                        {JSON.stringify(action.toolInput, null, 2)}
                      </pre>
                      {action.status === "pending" && (
                        <div className={styles.actionButtonRow}>
                          <Button kind="danger" size="sm" onClick={() => handleApprove(message.id, action)}>
                            Approve
                          </Button>
                          <Button kind="ghost" size="sm" onClick={() => handleDecline(message.id, action)}>
                            Decline
                          </Button>
                        </div>
                      )}
                      {action.status === "submitting" && <LoadingDots />}
                      {action.status === "declined" && <Tag type="gray" size="sm">Declined</Tag>}
                      {action.status === "approved" && (
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
                          <Tag type={action.isError ? "red" : "green"} size="sm">
                            {action.isError ? "Failed" : "Executed"}
                          </Tag>
                          {action.result && (
                            <p style={{ fontSize: "0.75rem", color: "var(--cds-text-secondary)" }}>
                              {action.result}
                            </p>
                          )}
                        </div>
                      )}
                    </Tile>
                  ))}
                </div>
              );
            })}
          </div>

          {error && (
            <InlineNotification
              kind="error"
              title="Chat error"
              subtitle={error}
              onCloseButtonClick={() => setError(null)}
              style={{ margin: "0 0.5rem" }}
              lowContrast
            />
          )}

          <div style={{ display: "flex", gap: "0.5rem", padding: "0.75rem", borderTop: "1px solid var(--cds-border-subtle-01)" }}>
            <TextInput
              id="chat-widget-input"
              labelText="Message"
              hideLabel
              placeholder="Ask about a build, artifact, or pipeline…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              size="md"
            />
            {isStreaming ? (
              <Button
                kind="danger"
                size="md"
                hasIconOnly
                iconDescription="Stop"
                renderIcon={StopFilledAlt}
                onClick={handleStop}
              />
            ) : (
              <Button
                kind="primary"
                size="md"
                hasIconOnly
                iconDescription="Send"
                renderIcon={Send}
                disabled={!input.trim()}
                onClick={handleSend}
              />
            )}
          </div>
        </div>
      ) : (
        <Button
          kind="primary"
          size="lg"
          hasIconOnly
          iconDescription="Open chat assistant"
          renderIcon={ChatBot}
          onClick={() => setIsOpen(true)}
          style={{ borderRadius: "50%", width: "3.5rem", height: "3.5rem" }}
        />
      )}
    </div>
  );
}
