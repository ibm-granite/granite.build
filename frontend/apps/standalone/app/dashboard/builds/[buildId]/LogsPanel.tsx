"use client";

import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import {
  Button,
  TextInput,
  Checkbox,
  Toggle,
  InlineLoading,
  InlineNotification,
  Tag,
  SkeletonText,
  AILabel,
  AILabelContent,
  Layer,
} from "@carbon/react";
import { Download, Copy, Renew, Restart } from "@carbon/icons-react";
import { useQuery } from "@tanstack/react-query";
import { getBuildStepLog, getBuildLiveLogs } from "@granite-build/ui-core/api/gbserver";
import { getAIAnalysis, analyzeLogsContent } from "@granite-build/ui-core/api/analytics";
import type { AIAnalysis, BuildStatusDetail, BuildStepRun } from "@granite-build/ui-core/types";

interface Props {
  buildId: string;
  status?: BuildStatusDetail;
}

function stepLabel(step: BuildStepRun): string {
  const name = step.step_name ?? "";
  const parts = name.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || name;
}

function Toolbar({
  onRefetch,
  onDownload,
  onCopy,
  stream,
  onStreamChange,
  wrap,
  onWrapChange,
  filter,
  onFilterChange,
  linesInfo,
  disabled,
}: {
  onRefetch: () => void;
  onDownload: () => void;
  onCopy: () => void;
  stream: boolean;
  onStreamChange: (v: boolean) => void;
  wrap: boolean;
  onWrapChange: (v: boolean) => void;
  filter: string;
  onFilterChange: (v: string) => void;
  linesInfo: string;
  disabled: boolean;
}) {
  return (
    <>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.5rem",
          alignItems: "center",
          marginBottom: "1rem",
        }}
      >
        <TextInput
          id="log-filter"
          size="sm"
          labelText=""
          placeholder="Filter…"
          value={filter}
          onChange={(e) => onFilterChange(e.target.value)}
          style={{ width: "50%" }}
        />
        <Button
          kind="ghost"
          size="sm"
          renderIcon={Renew}
          iconDescription="Refresh"
          hasIconOnly
          onClick={onRefetch}
        />
        <Button
          kind="ghost"
          size="sm"
          renderIcon={Copy}
          iconDescription="Copy"
          hasIconOnly
          onClick={onCopy}
          disabled={disabled}
        />
        <Button
          kind="ghost"
          size="sm"
          renderIcon={Download}
          iconDescription="Download"
          hasIconOnly
          onClick={onDownload}
          disabled={disabled}
          tooltipPosition="left"
        />
      </div>
      <div
        style={{
          display: "flex",
          gap: "0.5rem",
          alignItems: "baseline",
          marginBottom: "0.25rem",
        }}
      >
        <div
          style={{
            display: "flex",
            gap: "1rem",
            alignItems: "baseline",
          }}
        >
          <Checkbox
            id="log-stream"
            labelText="Stream"
            checked={stream}
            onChange={(_: unknown, { checked }: { checked: boolean }) =>
              onStreamChange(checked)
            }
          />
          <Checkbox
            id="log-wrap"
            labelText="Wrap lines"
            checked={wrap}
            onChange={(_: unknown, { checked }: { checked: boolean }) =>
              onWrapChange(checked)
            }
          />
        </div>
        <div
          style={{
            marginLeft: "auto",
            fontSize: "0.875rem",
            color: "var(--cds-text-secondary)",
            whiteSpace: "nowrap",
          }}
        >
          {linesInfo}
        </div>
      </div>
    </>
  );
}

function LogPre({
  content,
  wrap,
  filter,
}: {
  content: string;
  wrap: boolean;
  filter: string;
}) {
  const lines = content.split("\n");
  const displayed = filter
    ? lines.filter((l) => l.toLowerCase().includes(filter.toLowerCase()))
    : lines;
  return (
    <pre
      style={{
        background: "var(--cds-layer)",
        border: "1px solid var(--cds-border-subtle-01)",
        padding: "0.75rem 1rem",
        margin: 0,
        fontSize: "0.75rem",
        fontFamily: "IBM Plex Mono, monospace",
        lineHeight: 1.6,
        overflowX: "auto",
        overflowY: "auto",
        flex: 1,
        minHeight: 0,
        whiteSpace: wrap ? "pre-wrap" : "pre",
        wordBreak: wrap ? "break-all" : "normal",
      }}
    >
      {displayed.length > 0 ? (
        displayed.join("\n")
      ) : (
        <span style={{ color: "var(--cds-text-secondary)" }}>
          No log lines{filter ? " matching filter" : ""}.
        </span>
      )}
    </pre>
  );
}

// ── Step-log view (completed steps with log_path) ─────────────────────────────

function StepLogsView({
  steps,
  onFirstContent,
}: {
  steps: BuildStepRun[];
  onFirstContent?: (c: string) => void;
}) {
  const [selectedPath, setSelectedPath] = useState<string>(
    steps[0]?.log_path ?? "",
  );
  const [stream, setStream] = useState(false);
  const [wrap, setWrap] = useState(false);
  const [filter, setFilter] = useState("");

  const activePath = selectedPath || steps[0]?.log_path || "";

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["logs-panel-step", activePath],
    queryFn: () => getBuildStepLog(activePath),
    enabled: Boolean(activePath),
    refetchInterval: stream ? 5000 : false,
    staleTime: 0,
  });

  const content = data ?? "";
  const lineCount = content ? content.split("\n").length : 0;
  const filterCount = filter
    ? content
        .split("\n")
        .filter((l) => l.toLowerCase().includes(filter.toLowerCase())).length
    : lineCount;

  const firedRef = useRef(false);
  useEffect(() => {
    if (content && !firedRef.current) {
      firedRef.current = true;
      onFirstContent?.(content);
    }
  }, [content, onFirstContent]);

  function download() {
    if (!content) return;
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activePath.split("/").pop() ?? "log"}.log`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        flex: 1,
        minHeight: 0,
        gap: "0.5rem",
      }}
    >
      {steps.length > 1 && (
        <div
          style={{
            display: "flex",
            gap: 0,
            flexShrink: 0,
            borderBottom: "1px solid var(--cds-border-subtle-01)",
          }}
        >
          {steps.map((s) => (
            <button
              key={s.log_path}
              onClick={() => setSelectedPath(s.log_path!)}
              style={{
                padding: "0.5rem 1rem",
                fontSize: "0.875rem",
                background: "none",
                border: "none",
                borderBottom:
                  activePath === s.log_path
                    ? "2px solid var(--cds-interactive)"
                    : "2px solid transparent",
                color:
                  activePath === s.log_path
                    ? "var(--cds-text-primary)"
                    : "var(--cds-text-secondary)",
                cursor: "pointer",
                marginBottom: "-1px",
              }}
            >
              {stepLabel(s)}
            </button>
          ))}
        </div>
      )}
      <Layer>
        <Toolbar
          onRefetch={() => refetch()}
          onDownload={download}
          onCopy={() => navigator.clipboard.writeText(content)}
          stream={stream}
          onStreamChange={setStream}
          wrap={wrap}
          onWrapChange={setWrap}
          filter={filter}
          onFilterChange={setFilter}
          linesInfo={
            filter
              ? `${filterCount} of ${lineCount} lines`
              : `${lineCount} lines`
          }
          disabled={!content}
        />
      </Layer>
      {error && (
        <InlineNotification
          kind="error"
          title="Error"
          subtitle={(error as Error).message}
          lowContrast
        />
      )}
      {isLoading ? (
        <InlineLoading description="Loading logs…" />
      ) : (
        <LogPre content={content} wrap={wrap} filter={filter} />
      )}
    </div>
  );
}

// ── Live log view (running steps, no log_path yet) ─────────────────────────────
//
// Reads directly from gbserver's /logs/logquery endpoint, which in standalone
// mode serves MESSAGE_EVENT rows straight from the local event store (see
// getBuildLiveLogs in api/gbserver.ts) — no cloud logs service required.

function LiveLogsView({
  buildId,
  onFirstContent,
}: {
  buildId: string;
  onFirstContent?: (c: string) => void;
}) {
  const [stream, setStream] = useState(false);
  const [wrap, setWrap] = useState(false);
  const [filter, setFilter] = useState("");

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["logs-panel-live", buildId],
    queryFn: () => getBuildLiveLogs(buildId, 500),
    refetchInterval: stream ? 5000 : false,
    staleTime: 0,
  });

  const lines = data?.lines ?? [];
  const total = data?.total ?? 0;
  const filterCount = filter
    ? lines.filter((l) => l.toLowerCase().includes(filter.toLowerCase())).length
    : lines.length;
  const content = lines.join("\n");

  const firedRef = useRef(false);
  useEffect(() => {
    if (content && !firedRef.current) {
      firedRef.current = true;
      onFirstContent?.(content);
    }
  }, [content, onFirstContent]);

  function download() {
    if (!content) return;
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${buildId}.log`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        flex: 1,
        minHeight: 0,
        gap: "0.5rem",
      }}
    >
      <Layer>
        <Toolbar
          onRefetch={() => refetch()}
          onDownload={download}
          onCopy={() => navigator.clipboard.writeText(content)}
          stream={stream}
          onStreamChange={setStream}
          wrap={wrap}
          onWrapChange={setWrap}
          filter={filter}
          onFilterChange={setFilter}
          linesInfo={
            filter
              ? `${filterCount} of ${lines.length} lines`
              : `${lines.length} of ${total} lines`
          }
          disabled={lines.length === 0}
        />
      </Layer>
      {error && (
        <InlineNotification
          kind="error"
          title="Error"
          subtitle={(error as Error).message}
          lowContrast
        />
      )}
      {isLoading ? (
        <InlineLoading description="Loading logs…" />
      ) : (
        <LogPre content={content} wrap={wrap} filter={filter} />
      )}
    </div>
  );
}

// ── AI summary banner ──────────────────────────────────────────────────────────

function AIBanner({
  buildId,
  logContent,
}: {
  buildId: string;
  logContent: string;
}) {
  const [liveAnalysis, setLiveAnalysis] = useState<AIAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const firedRef = useRef(false);

  const { data: cachedAnalyses } = useQuery({
    queryKey: ["ai-analysis", buildId],
    queryFn: () => getAIAnalysis(buildId),
    staleTime: 60_000,
  });

  const runAnalysis = useCallback(
    (content: string) => {
      setIsAnalyzing(true);
      setLiveAnalysis(null);
      analyzeLogsContent(buildId, content)
        .then((result) => {
          if (result) setLiveAnalysis(result);
        })
        .finally(() => setIsAnalyzing(false));
    },
    [buildId],
  );

  useEffect(() => {
    if (!logContent || firedRef.current) return;
    firedRef.current = true;
    runAnalysis(logContent);
  }, [logContent, runAnalysis]);

  const primary =
    liveAnalysis ??
    cachedAnalyses?.find((a) => a.source === "llm_phase1") ??
    cachedAnalyses?.[0];

  if (!primary?.summary && !isAnalyzing) return null;

  const pct = primary ? Math.round(primary.confidence * 100) : 0;
  const confidenceColor = pct >= 70 ? "green" : pct >= 40 ? "warm-gray" : "red";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.25rem",
        padding: "0.75rem 0.75rem 1rem 0.75rem",
        background: "var(--cds-layer)",
        border: "1px solid var(--cds-border-subtle-01)",
        fontSize: "0.875rem",
      }}
    >
      <div
        style={{
          display: "flex",
          gap: "0.5rem",
          alignItems: "center",
          flexWrap: "wrap",
          marginBottom: "0.25rem"
        }}
      >
        <AILabel size="xs">
          <AILabelContent>AI was used to generate this content</AILabelContent>
        </AILabel>
        {isAnalyzing ? (
          <SkeletonText width="200px" />
        ) : (
          <>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                flexWrap: "wrap",
                flexGrow: "1",
                justifyContent: "space-between",
              }}
            >
              <Tag type={confidenceColor} size="md">
                Confidence: {pct}%
              </Tag>
              {primary?.error_category_1 && (
                <Tag type="warm-gray" size="sm">
                  {primary.error_category_1}
                </Tag>
              )}
              {primary?.error_category_2 && (
                <Tag type="warm-gray" size="sm">
                  {primary.error_category_2}
                </Tag>
              )}

              {logContent && (
                <Button
                  kind="ghost"
                  size="sm"
                  renderIcon={Restart}
                  iconDescription="Refresh"
                  hasIconOnly
                  onClick={() => {
                    firedRef.current = false;
                    runAnalysis(logContent);
                  }}
                  style={{ marginLeft: "auto" }}
                />
              )}
            </div>
          </>
        )}
      </div>
      {!isAnalyzing && primary?.summary && (
        <span
          style={{
            color: "var(--cds-text-primary)",
            fontSize: "0.875rem",
            marginBottom: "0.25rem",
          }}
        >
          {primary.summary}
        </span>
      )}
      {!isAnalyzing && primary?.root_cause && (
        <span
          style={{
            color: "var(--cds-text-secondary)",
            fontSize: "0.875rem",
            marginBottom: "0.25rem",
          }}
        >
          <strong>Root cause: </strong>
          {primary.root_cause}
        </span>
      )}
      {!isAnalyzing && primary?.suggested_action && (
        <span
          style={{
            color: "var(--cds-text-secondary)",
            fontSize: "0.875rem",
            marginBottom: "0.25rem",
          }}
        >
          <strong>Suggested action: </strong>
          {primary.suggested_action}
        </span>
      )}
    </div>
  );
}

// ── Main panel ─────────────────────────────────────────────────────────────────

export function LogsPanel({ buildId, status }: Props) {
  const [logContent, setLogContent] = useState("");

  const stepsWithLogs: BuildStepRun[] = useMemo(() => {
    const seen = new Set<string>();
    const out: BuildStepRun[] = [];
    for (const target of Object.values(status?.targets ?? {})) {
      for (const step of target.steps ?? []) {
        if (step.log_path && !seen.has(step.log_path)) {
          seen.add(step.log_path);
          out.push(step);
        }
      }
    }
    return out;
  }, [status]);

  return (
    <div
      style={{
        padding: "2rem",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        boxSizing: "border-box",
        gap: "0.75rem",
      }}
    >
      <AIBanner buildId={buildId} logContent={logContent} />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          flex: 1,
          minHeight: 0,
        }}
      >
        {stepsWithLogs.length > 0 ? (
          <StepLogsView steps={stepsWithLogs} onFirstContent={setLogContent} />
        ) : (
          <LiveLogsView buildId={buildId} onFirstContent={setLogContent} />
        )}
      </div>
    </div>
  );
}
