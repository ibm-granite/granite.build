"use client";

import * as React from "react";
import { CopyButton, SkeletonText, Tag } from "@carbon/react";
import styles from "./DetailsPanel.module.scss";
import type { Build, BuildStatusDetail } from "@granite-build/ui-core/types";
import { BuildStatusBadge } from "@granite-build/ui-core/components/BuildStatusBadge";


interface DetailFieldProps {
  label: string;
  children: React.ReactNode;
}

function DetailField({ label, children }: DetailFieldProps) {
  return (
    <div>
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

  return (
    <div style={{ padding: '0.5rem 0 0.5rem 1rem' }}>
<dl className={styles.detailsList}>
        <DetailField label="Build ID">
          <span
            style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}
          >
            <code
              className={styles.wordBreakAll}
              style={{ fontSize: "0.875rem" }}
            >
              {build.uuid}
            </code>
            <CopyButton
              feedback="Copied!"
              iconDescription="Copy build UUID"
              onClick={() => navigator.clipboard.writeText(build.uuid)}
              size="sm"
            />
          </span>
        </DetailField>
        <DetailField label="Name">
          <span className={styles.wordBreakAll}>{build.name}</span>
        </DetailField>
        <DetailField label="Space">{build.space_name}</DetailField>
        <DetailField label="Username">{build.username}</DetailField>
        <DetailField label="Started">
          {new Date(build.created_time).toLocaleString()}
        </DetailField>
        <DetailField label="Updated">
          {new Date(build.updated_time).toLocaleString()}
        </DetailField>
        {build.finished_at && (
          <DetailField label="Finished">
            {new Date(build.finished_at).toLocaleString()}
          </DetailField>
        )}
        {build.source_uri && (
          <DetailField label="Source URI">
            <a
              href={build.source_uri}
              target="_blank"
              rel="noreferrer"
              className={styles.sourceLink}
            >
              {build.source_uri}
            </a>
          </DetailField>
        )}
        {build.description && (
          <DetailField label="Description">{build.description}</DetailField>
        )}
        {build.resources && (
          <DetailField label="Resources">
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
          </DetailField>
        )}
      </dl>
    </div>
  );
}
