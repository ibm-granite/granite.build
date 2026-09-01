"use client";

import * as React from "react";
import { CopyButton, SkeletonText } from "@carbon/react";
import Link from "next/link";
import styles from "./DetailsPanel.module.scss";
import type { Artifact } from "@granite-build/ui-core/types";
import { getHuggingFaceUrl } from "@granite-build/ui-core/components/LineageGraph/diagramUtilities";

function DetailField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className={styles.fieldLabel}>{label}</div>
      <div className={styles.fieldValue}>{children}</div>
    </div>
  );
}

export function DetailsPanel({
  artifact,
  loading,
}: {
  artifact: Artifact | undefined;
  loading: boolean;
}) {
  if (loading)
    return (
      <div style={{ padding: "1.5rem" }}>
        <SkeletonText paragraph lineCount={8} />
      </div>
    );
  if (!artifact) return null;

  return (
    <div style={{ padding: "1rem 1.5rem" }}>
      <dl className={styles.detailsList}>
        <DetailField label="Artifact ID">
          <span
            style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}
          >
            <code className={styles.mono}>{artifact.uuid}</code>
            <CopyButton
              feedback="Copied!"
              iconDescription="Copy ID"
              onClick={() => navigator.clipboard.writeText(artifact.uuid)}
              size="sm"
            />
          </span>
        </DetailField>
        <DetailField label="Name">
          <span className={styles.wordBreakAll}>{artifact.name}</span>
        </DetailField>
        <DetailField label="Type">{artifact.artifact_type.toLowerCase()}</DetailField>
        <DetailField label="Space">{artifact.space_name}</DetailField>
        <DetailField label="Owner">{artifact.username}</DetailField>
        <DetailField label="URI">
          {(() => {
            const hfUrl = getHuggingFaceUrl(artifact.uri)
            return hfUrl ? (
              <a
                href={hfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.mono}
                style={{ color: 'var(--cds-link-primary, #0f62fe)' }}
              >
                {artifact.uri}
              </a>
            ) : (
              <span className={styles.mono}>{artifact.uri}</span>
            )
          })()}
        </DetailField>
        {artifact.build_id && (
          <DetailField label="Created by build">
            <Link
              href={`/dashboard/builds/_/?id=${artifact.build_id}`}
              style={{ color: "var(--cds-link-primary, #0f62fe)" }}
            >
              <span className={styles.mono}>{artifact.build_id}</span>
            </Link>
          </DetailField>
        )}
        <DetailField label="Created">
          {artifact.created_time ? new Date(artifact.created_time).toLocaleString() : '—'}
        </DetailField>
        <DetailField label="Updated">
          {artifact.updated_time ? new Date(artifact.updated_time).toLocaleString() : '—'}
        </DetailField>
        {artifact.description && (
          <DetailField label="Description">{artifact.description}</DetailField>
        )}
        {artifact.checksum && (
          <DetailField label="Checksum">
            <span className={styles.mono}>{artifact.checksum}</span>
          </DetailField>
        )}
        <DetailField label="Archived">
          {artifact.archived ? "Yes" : "No"}
        </DetailField>
      </dl>
    </div>
  );
}
