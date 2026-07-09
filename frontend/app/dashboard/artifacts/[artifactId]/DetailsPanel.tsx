"use client";

import * as React from "react";
import { CopyButton, SkeletonText } from "@carbon/react";
import Link from "next/link";
import styles from "./DetailsPanel.module.scss";
import type { Artifact } from "@/types";


// Mirrors src/gbcommon/utils/hf_utils.py:convert_hf_uri_to_url — model URLs
// never include a "models/" segment; datasets/spaces/buckets keep their
// pluralized type segment.
function getHuggingFaceUrl(uri: string): string | null {
  if (!uri) return null

  if (uri.startsWith('hf://')) {
    const remainder = uri.slice(5)
    let parts: string[]

    if (remainder.startsWith('/')) {
      // hf:///[type/]org/name
      parts = remainder.replace(/^\/+/, '').split('/')
    } else if (remainder.startsWith('huggingface.co/')) {
      // hf://huggingface.co/[type/]org/name
      parts = remainder.slice('huggingface.co/'.length).split('/')
    } else if (remainder.includes('/')) {
      // hf://<domain>/[type/]org/name — the domain segment is discarded;
      // the browsable URL is always on huggingface.co
      parts = remainder.split('/').slice(1)
    } else {
      return null
    }

    if (parts.length === 2) {
      const [org, name] = parts
      return `https://huggingface.co/${org}/${name}`
    }
    if (parts.length === 3) {
      const [type, org, name] = parts
      switch (type) {
        case 'models':   return `https://huggingface.co/${org}/${name}`
        case 'datasets': return `https://huggingface.co/datasets/${org}/${name}`
        case 'spaces':   return `https://huggingface.co/spaces/${org}/${name}`
        case 'buckets':  return `https://huggingface.co/buckets/${org}/${name}`
        default: return null
      }
    }
    return null
  }

  if (/huggingface\.co/.test(uri)) return uri.startsWith('http') ? uri : `https://${uri}`
  return null
}

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
