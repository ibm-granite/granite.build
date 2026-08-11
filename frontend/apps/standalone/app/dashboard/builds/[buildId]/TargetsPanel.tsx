"use client";

import { useState } from "react";
import type { CSSProperties } from "react";
import { useQuery, useQueries } from "@tanstack/react-query";
import Link from "next/link";
import { Button, InlineNotification, Modal, SkeletonText } from "@carbon/react";
import { Document } from "@carbon/icons-react";
import { BuildStatusBadge } from "@granite-build/ui-core/components/BuildStatusBadge";
import { getArtifact, getBuildStepLog } from "@granite-build/ui-core/api/gbserver";
import type { Artifact, BuildTargetRun, BuildStatus } from "@granite-build/ui-core/types";

interface Props {
  targets?: Record<string, BuildTargetRun> | BuildTargetRun[];
}

interface LogTarget {
  path: string;
  name: string;
}

function isUUID(s: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
    s,
  );
}

const thStyle: CSSProperties = {
  padding: "0.25rem 1rem 0.25rem 0",
  fontSize: "0.875rem",
  fontWeight: 400,
  color: "var(--cds-text-secondary)",
  textAlign: "left",
  whiteSpace: "nowrap",
};

const tdStyle: CSSProperties = {
  padding: "0.375rem 1rem 0.375rem 0",
  fontSize: "0.875rem",
  verticalAlign: "middle",
};

function StepLogModal({
  logTarget,
  onClose,
}: {
  logTarget: LogTarget | null;
  onClose: () => void;
}) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["step-log", logTarget?.path],
    queryFn: () => getBuildStepLog(logTarget!.path),
    enabled: !!logTarget,
  });

  return (
    <Modal
      open={!!logTarget}
      modalHeading={logTarget?.name ?? ""}
      passiveModal
      onRequestClose={onClose}
      size="lg"
    >
      {isLoading && <SkeletonText paragraph lineCount={12} />}
      {error && (
        <InlineNotification
          kind="error"
          title="Failed to load log"
          subtitle={(error as Error).message}
          lowContrast
        />
      )}
      {data != null && (
        <pre
          style={{
            background: "var(--cds-layer, #f4f4f4)",
            padding: "1rem",
            overflowX: "auto",
            margin: 0,
            fontFamily: "IBM Plex Mono, monospace",
            fontSize: "0.75rem",
            lineHeight: "1.5",
            whiteSpace: "pre-wrap",
            wordBreak: "break-all",
          }}
        >
          {data}
        </pre>
      )}
    </Modal>
  );
}

function ArtifactTable({
  title,
  entries,
  artifactMap,
}: {
  title: string;
  entries: [string, string][];
  artifactMap: Map<string, Artifact | undefined>;
}) {
  return (
    <div style={{ marginBottom: "1.25rem" }}>
      <strong
        style={{
          display: "block",
          marginBottom: "0.5rem",
          fontSize: "0.875rem",
        }}
      >
        {title}
      </strong>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ ...thStyle, width: "45%" }}>Artifact ID</th>
            <th style={thStyle}>URI</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([param, artifactId]) => {
            const linked = isUUID(artifactId);
            const artifact = linked ? artifactMap.get(artifactId) : undefined;
            return (
              <tr key={param}>
                <td style={{ ...tdStyle, wordBreak: "break-all" }}>
                  {linked ? (
                    <Link
                      href={`/dashboard/artifacts/_/?id=${artifactId}`}
                      style={{
                        color: "var(--cds-link-primary)",
                        fontSize: "0.75rem",
                      }}
                    >
                      {artifactId}
                    </Link>
                  ) : (
                    <span
                      style={{
                        color: "var(--cds-text-secondary)",
                        fontSize: "0.75rem",
                      }}
                    >
                      {artifactId || "N/A"}
                    </span>
                  )}
                </td>
                <td
                  style={{
                    ...tdStyle,
                    fontSize: "0.75rem",
                    color: "var(--cds-text-secondary)",
                    wordBreak: "break-all",
                  }}
                >
                  {artifact?.uri ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function TargetsPanel({ targets }: Props) {
  const [openLog, setOpenLog] = useState<LogTarget | null>(null);

  const entries: [string, BuildTargetRun][] = !targets
    ? []
    : Array.isArray(targets)
      ? targets.map((t) => [t.target_name, t])
      : Object.entries(targets);

  const allArtifactIds = [
    ...new Set(
      entries
        .flatMap(([, t]) => [
          ...Object.values(t.inputs ?? {}),
          ...Object.values(t.outputs ?? {}),
        ])
        .filter(isUUID),
    ),
  ];

  const artifactQueries = useQueries({
    queries: allArtifactIds.map((id) => ({
      queryKey: ["artifact", id],
      queryFn: () => getArtifact(id),
      staleTime: 5 * 60 * 1000,
      retry: false,
    })),
  });

  const artifactMap = new Map<string, Artifact | undefined>(
    allArtifactIds.map((id, i) => [id, artifactQueries[i]?.data]),
  );

  if (!targets) {
    return (
      <p style={{ padding: "1rem", color: "var(--cds-text-secondary)" }}>
        No target data available.
      </p>
    );
  }

  if (entries.length === 0) {
    return (
      <p style={{ padding: "1rem", color: "var(--cds-text-secondary)" }}>
        No targets.
      </p>
    );
  }

  return (
    <>
      <StepLogModal logTarget={openLog} onClose={() => setOpenLog(null)} />
      <div style={{ padding: "0.5rem 0 1rem 1rem" }}>
        {entries.map(([name, target], idx) => {
          const inputEntries = Object.entries(target.inputs ?? {});
          const outputEntries = Object.entries(target.outputs ?? {});

          return (
            <div key={name}>
              {idx > 0 && (
                <div
                  style={{
                    borderTop: "1px solid var(--cds-border-subtle-01)",
                    margin: "1.5rem 0",
                  }}
                />
              )}

              {/* Target header */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  alignItems: "baseline",
                  marginBottom: "1rem",
                }}
              >
                <p
                  style={{
                    fontSize: "0.875rem",
                    color: "var(--cds-text-secondary)",
                    marginBottom: "1rem",
                    marginTop: 0,
                    marginRight: "4rem",
                  }}
                >
                  Target #{idx + 1} {name}
                </p>
                <BuildStatusBadge
                  status={target.status as BuildStatus}
                  showLabel
                />
              </div>

              {/* Steps */}
              {target.steps && target.steps.length > 0 && (
                <div style={{ marginBottom: "1.25rem" }}>
                  <strong
                    style={{
                      display: "block",
                      marginBottom: "0.5rem",
                      fontSize: "0.875rem",
                    }}
                  >
                    Steps
                  </strong>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th style={{ ...thStyle, width: "15rem" }}>Name</th>
                        <th style={{ ...thStyle, width: "15rem" }}>Status</th>
                        <th style={thStyle}>URI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {target.steps.map((step) => (
                        <tr key={step.step_name}>
                          <td style={{ ...tdStyle, whiteSpace: "nowrap" }}>
                            {step.step_name}
                          </td>
                          <td style={{ ...tdStyle, whiteSpace: "nowrap" }}>
                            <BuildStatusBadge
                              status={step.status as BuildStatus}
                              showLabel
                            />
                          </td>
                          <td
                            style={{
                              ...tdStyle,
                              fontSize: "0.75rem",
                              color: "var(--cds-text-secondary)",
                              wordBreak: "break-all",
                            }}
                          >
                            {step.uri ?? "—"}
                          </td>
                          <td style={{ ...tdStyle, width: "2rem", padding: 0 }}>
                            {step.log_path && (
                              <Button
                                kind="ghost"
                                size="sm"
                                hasIconOnly
                                renderIcon={Document}
                                iconDescription="View logs"
                                tooltipPosition="left"
                                onClick={() =>
                                  setOpenLog({
                                    path: step.log_path!,
                                    name: step.step_name,
                                  })
                                }
                              />
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Input Artifacts */}
              {inputEntries.length > 0 && (
                <ArtifactTable
                  title="Input Artifacts"
                  entries={inputEntries}
                  artifactMap={artifactMap}
                />
              )}

              {/* Output Artifacts */}
              {outputEntries.length > 0 && (
                <ArtifactTable
                  title="Output Artifacts"
                  entries={outputEntries}
                  artifactMap={artifactMap}
                />
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}
