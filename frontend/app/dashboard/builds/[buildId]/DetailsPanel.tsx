"use client";

import * as React from "react";
import { CopyButton, SkeletonText, Tag } from "@carbon/react";
import styles from "./DetailsPanel.module.scss";
import type { Build, BuildStatusDetail } from "@/types";
import { BuildStatusBadge } from "@/components/BuildStatusBadge";


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
        {status?.job && status.job.attempts > 1 && (
          <>
            <DetailField label="Job status">
              <BuildStatusBadge status={status.job.status} />
            </DetailField>
            <DetailField label="Attempts">{status.job.attempts}</DetailField>
            <DetailField label="Targets">
              {status.job.counts.succeeded} of {status.job.counts.total} succeeded
              {status.job.counts.failed > 0 && `, ${status.job.counts.failed} failed`}
              {status.job.counts.running > 0 && `, ${status.job.counts.running} running`}
              {status.job.counts.not_run > 0 && `, ${status.job.counts.not_run} never ran`}
            </DetailField>
            {status.job.targets.length > 0 && (
              <DetailField label="Target results">
                <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
                  {status.job.targets.map((t) => (
                    <span
                      key={t.name}
                      style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                    >
                      <code className={styles.wordBreakAll} style={{ fontSize: "0.875rem" }}>
                        {t.name}
                      </code>
                      {t.status ? (
                        <BuildStatusBadge status={t.status} />
                      ) : (
                        <span style={{ fontSize: "0.875rem", color: "#6f6f6f" }}>Never ran</span>
                      )}
                    </span>
                  ))}
                </div>
              </DetailField>
            )}
            <DetailField label="Attempt builds">
              <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
                {status.job.attempt_builds
                  // Root-first by contract, so the ordinal is the position in
                  // the full list — compute it before filtering out the build
                  // currently being viewed.
                  .map((ab, index) => ({ ...ab, attempt: index + 1 }))
                  .filter((ab) => ab.build_id !== build.uuid)
                  .map((ab) => (
                    <span
                      key={ab.build_id}
                      style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                    >
                      <a
                        href={`/dashboard/builds/_/?id=${ab.build_id}`}
                        className={styles.wordBreakAll}
                      >
                        <code style={{ fontSize: "0.875rem" }}>
                          Attempt {ab.attempt}: {ab.build_id}
                        </code>
                      </a>
                      <BuildStatusBadge status={ab.status} showLabel={false} />
                    </span>
                  ))}
              </div>
            </DetailField>
          </>
        )}
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
