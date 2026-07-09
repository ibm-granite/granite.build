"use client";

import React from "react";
import Link from "next/link";
import { DataTableSkeleton } from "@carbon/react";
import type { DPDataset } from "@/api/dataProcessing";
import { adaptStatus } from "@/api/gbserver";
import { BuildStatusBadge } from "@/components/BuildStatusBadge";
import styles from "./page.module.scss";

interface Props {
  datasets: DPDataset[];
  search: string;
  isLoading: boolean;
}

const STAGES: { key: keyof DPDataset; label: string }[] = [
  { key: "parquet_path", label: "Parquet" },
  { key: "arrow_path", label: "Arrow" },
  { key: "megatron_path", label: "Megatron" },
  { key: "merged_text_path", label: "Merged Text" },
  { key: "merged_bin_path", label: "Merged Bin" },
];

function formatAge(iso: string): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function PipelineTab({ datasets, search, isLoading }: Props) {
  if (isLoading)
    return (
      <DataTableSkeleton
        rowCount={8}
        columnCount={3}
        showHeader={false}
        showToolbar={false}
      />
    );

  const searchLower = search.toLowerCase();
  const filtered = searchLower
    ? datasets.filter(
        (d) =>
          d.name.toLowerCase().includes(searchLower) ||
          d.arrow_path?.toLowerCase().includes(searchLower) ||
          d.megatron_path?.toLowerCase().includes(searchLower),
      )
    : datasets;

  if (filtered.length === 0) {
    return (
      <div
        style={{
          padding: "3rem 0",
          textAlign: "center",
          color: "var(--cds-text-secondary)",
        }}
      >
        No datasets found.
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <div
        className={styles.pipelineRow}
        style={{
          fontWeight: 600,
          fontSize: "0.75rem",
          color: "var(--cds-text-secondary)",
          background: "var(--cds-layer-01)",
        }}
      >
        <div className={styles.pipelineName}>Dataset</div>
        <div className={styles.pipelineStages}>Stages</div>
        <div className={styles.pipelineMeta}>Latest build</div>
        <div
          style={{ minWidth: "4rem", textAlign: "right", fontSize: "inherit" }}
        >
          Count
        </div>
        <div
          style={{ minWidth: "5rem", textAlign: "right", fontSize: "inherit" }}
        >
          Updated
        </div>
      </div>

      {filtered.map((ds) => (
        <div key={ds.name} className={styles.pipelineRow}>
          <div
            className={styles.pipelineName}
            title={ds.megatron_path || ds.arrow_path}
          >
            {ds.name}
          </div>

          <div className={styles.pipelineStages}>
            {STAGES.map((stage, i) => {
              const present = Boolean(ds[stage.key]);
              return (
                <React.Fragment key={stage.key}>
                  {i > 0 && <span className={styles.stageArrow}>→</span>}
                  <span
                    className={`${styles.stageChip} ${present ? styles.stagePresent : styles.stageMissing}`}
                    title={
                      present
                        ? String(ds[stage.key])
                        : `${stage.label}: not found`
                    }
                  >
                    {stage.label}
                  </span>
                </React.Fragment>
              );
            })}
          </div>

          <div className={styles.pipelineMeta}>
            {ds.latest_build_id ? (
              <>
                <BuildStatusBadge
                  status={adaptStatus(ds.latest_build_status ?? "")}
                />
                <Link
                  href={`/dashboard/builds/_/?id=${ds.latest_build_id}`}
                  style={{ fontSize: "0.8125rem" }}
                >
                  {ds.builds[0]?.name ?? ds.latest_build_id.slice(0, 8)}
                </Link>
              </>
            ) : (
              <span>—</span>
            )}
          </div>

          <div
            style={{
              minWidth: "4rem",
              textAlign: "right",
              fontSize: "0.8125rem",
            }}
          >
            {ds.build_count}
          </div>

          <div
            style={{
              minWidth: "5rem",
              textAlign: "right",
              fontSize: "0.8125rem",
              color: "var(--cds-text-secondary)",
            }}
          >
            {formatAge(ds.latest_build_time)}
          </div>
        </div>
      ))}
    </div>
  );
}
