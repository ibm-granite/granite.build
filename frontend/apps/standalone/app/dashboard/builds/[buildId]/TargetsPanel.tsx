"use client";

import { useState } from "react";
import { useQuery, useQueries } from "@tanstack/react-query";
import NextLink from "next/link";
import {
  Button,
  CopyButton,
  InlineNotification,
  Link as CarbonLink,
  Modal,
  SkeletonText,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@carbon/react";
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

/** Middle-truncate long IDs/URIs so they stay on one line; the full value is shown on hover and copied intact. */
function middleEllipsis(value: string, max = 48) {
  if (value.length <= max) return value;
  const head = Math.ceil((max - 1) / 2);
  const tail = Math.floor((max - 1) / 2);
  return `${value.slice(0, head)}…${value.slice(value.length - tail)}`;
}

/** A monospace ID/URI value: middle-truncated with a hover title, optional Carbon link, and a Carbon copy button. */
function ValueWithCopy({
  value,
  href,
  copyLabel,
}: {
  value: string;
  href?: string;
  copyLabel: string;
}) {
  const display = middleEllipsis(value);
  const textStyle = {
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.75rem",
  } as const;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.25rem",
        maxWidth: "100%",
      }}
    >
      {href ? (
        <CarbonLink as={NextLink} href={href} title={value} style={textStyle}>
          {display}
        </CarbonLink>
      ) : (
        <span title={value} style={{ ...textStyle, color: "var(--cds-text-secondary)" }}>
          {display}
        </span>
      )}
      <CopyButton
        feedback="Copied!"
        iconDescription={copyLabel}
        size="sm"
        onClick={() => navigator.clipboard.writeText(value)}
      />
    </span>
  );
}

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
      <Table size="sm" useZebraStyles={false}>
        <TableHead>
          <TableRow>
            <TableHeader>Artifact ID</TableHeader>
            <TableHeader>URI</TableHeader>
          </TableRow>
        </TableHead>
        <TableBody>
          {entries.map(([param, artifactId]) => {
            const linked = isUUID(artifactId);
            const artifact = linked ? artifactMap.get(artifactId) : undefined;
            return (
              <TableRow key={param}>
                <TableCell>
                  {linked ? (
                    <ValueWithCopy
                      value={artifactId}
                      href={`/dashboard/artifacts/_/?id=${artifactId}`}
                      copyLabel="Copy artifact ID"
                    />
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
                </TableCell>
                <TableCell>
                  {artifact?.uri ? (
                    <ValueWithCopy value={artifact.uri} copyLabel="Copy URI" />
                  ) : (
                    "—"
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
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

              {/* Target header — status sits directly beside its target name */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.75rem",
                  flexWrap: "wrap",
                  marginBottom: "1rem",
                }}
              >
                <p
                  style={{
                    fontSize: "0.875rem",
                    fontWeight: 600,
                    color: "var(--cds-text-primary)",
                    margin: 0,
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
                  <Table size="sm" useZebraStyles={false}>
                    <TableHead>
                      <TableRow>
                        <TableHeader>Name</TableHeader>
                        <TableHeader>Status</TableHeader>
                        <TableHeader>URI</TableHeader>
                        <TableHeader aria-label="Actions" />
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {target.steps.map((step) => (
                        <TableRow key={step.step_name}>
                          <TableCell style={{ whiteSpace: "nowrap" }}>
                            {step.step_name}
                          </TableCell>
                          <TableCell style={{ whiteSpace: "nowrap" }}>
                            <BuildStatusBadge
                              status={step.status as BuildStatus}
                              showLabel
                            />
                          </TableCell>
                          <TableCell>
                            {step.uri ? (
                              <ValueWithCopy value={step.uri} copyLabel="Copy URI" />
                            ) : (
                              "—"
                            )}
                          </TableCell>
                          <TableCell style={{ width: "2.5rem" }}>
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
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
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
