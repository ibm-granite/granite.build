"use client";

import * as React from "react";
import { CopyButton, Link as CarbonLink, SkeletonText, Tag } from "@carbon/react";
import styles from "./DetailsPanel.module.scss";
import type { Build, BuildStatusDetail } from "@granite-build/ui-core/types";
import { BuildStatusBadge } from "@granite-build/ui-core/components/BuildStatusBadge";


interface DetailFieldProps {
  label: string;
  column: 1 | 2;
  row: number;
  children: React.ReactNode;
}

function DetailField({ label, column, row, children }: DetailFieldProps) {
  return (
    <div className={column === 1 ? styles.col1 : styles.col2} style={{ gridRow: row }}>
      <div className={styles.fieldLabel}>{label}</div>
      <div className={styles.fieldValue}>{children}</div>
    </div>
  );
}

interface DetailsPanelProps {
  build: Build | undefined;
  status: BuildStatusDetail | undefined;
  loading: boolean;
}

export function DetailsPanel({ build, status, loading }: DetailsPanelProps) {
  if (loading) {
    return <SkeletonText paragraph lineCount={6} />;
  }

  if (!build) return null;

  const fields: { label: string; content: React.ReactNode }[] = [
    {
      label: "Build ID",
      content: (
        <span style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}>
          <code className={styles.wordBreakAll} style={{ fontSize: "0.875rem" }}>
            {build.uuid}
          </code>
          <CopyButton
            feedback="Copied!"
            iconDescription="Copy build UUID"
            onClick={() => navigator.clipboard.writeText(build.uuid)}
            size="sm"
          />
        </span>
      ),
    },
    { label: "Name", content: <span className={styles.wordBreakAll}>{build.name}</span> },
    { label: "Space", content: build.space_name },
    { label: "Username", content: build.username },
    { label: "Started", content: new Date(build.created_time).toLocaleString() },
    { label: "Updated", content: new Date(build.updated_time).toLocaleString() },
  ];

  if (build.finished_at) {
    fields.push({ label: "Finished", content: new Date(build.finished_at).toLocaleString() });
  }
  if (build.source_uri) {
    fields.push({
      label: "Source URI",
      content: (
        <CarbonLink href={build.source_uri} target="_blank" rel="noreferrer" className={styles.sourceLink}>
          {build.source_uri}
        </CarbonLink>
      ),
    });
  }
  if (build.description) {
    fields.push({ label: "Description", content: build.description });
  }
  if (build.resources) {
    fields.push({
      label: "Resources",
      content: (
        <div className={styles.resourcesTags}>
          {build.resources.cpu && (
            <Tag type="blue" size="sm">
              CPU {build.resources.cpu}
            </Tag>
          )}
          {build.resources.memory && (
            <Tag type="green" size="sm">
              Mem {build.resources.memory}
            </Tag>
          )}
          {build.resources.gpu != null && (
            <Tag type="purple" size="sm">
              GPU ×{build.resources.gpu}
            </Tag>
          )}
        </div>
      ),
    });
  }

  return (
    <>
      {fields.map((field, i) => (
        <DetailField key={field.label} label={field.label} column={1} row={i + 1}>
          {field.content}
        </DetailField>
      ))}
    </>
  );
}
