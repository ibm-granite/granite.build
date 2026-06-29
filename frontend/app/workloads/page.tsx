"use client";

import { useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { ENV_OVERRIDE_PARAM } from "@/config/environments";
import {
  Accordion,
  AccordionItem,
  ProgressBar,
  SkeletonText,
  Tag,
  Tile,
  StructuredListWrapper,
  StructuredListBody,
  StructuredListRow,
  StructuredListCell,
} from "@carbon/react";
import {
  ChevronDown,
  ChevronUp,
  Folder,
  Time,
  User,
} from "@carbon/icons-react";
import { useQuery } from "@tanstack/react-query";
import {
  getQueueCapacity,
  getNodePools,
  getLeaderboard,
  getBuildResources,
  getBuildK8sResources,
  isSidecarConfigured,
} from "@/api/analytics";
import { listBuilds } from "@/api/gbserver";
import { PageHeader } from "@/components/PageHeader";
import { BuildStatusBadge } from "@/components/BuildStatusBadge";
import styles from "./page.module.scss";
import type { BuildResources } from "@/api/analytics";
import type { Build, K8sResource } from "@/types";

const ACTIVE_STATUSES = ["running", "submitted", "pending"] as const;
const REFETCH = 30_000;

function NotConfigured() {
  return (
    <div
      style={{
        padding: "3rem",
        textAlign: "center",
        color: "var(--cds-text-secondary)",
      }}
    >
      <h2 style={{ marginBottom: "0.5rem" }}>Workloads view not available</h2>
      <p>
        Configure <code>GB_UI_KUBECONFIG</code> and a running analytics sidecar
        with K8s access.
      </p>
    </div>
  );
}

function StatTile({
  label,
  value,
  loading,
}: {
  label: string;
  value: string | number;
  loading: boolean;
}) {
  return (
    <Tile>
      <p
        style={{
          fontSize: "0.875rem",
          color: "var(--cds-text-secondary)",
          marginBottom: "0.25rem",
        }}
      >
        {label}
      </p>
      {loading ? (
        <SkeletonText width="60px" />
      ) : (
        <p
          style={{
            fontSize: "2rem",
            fontWeight: 300,
            lineHeight: 1,
            margin: 0,
          }}
        >
          {value}
        </p>
      )}
    </Tile>
  );
}

// ── K8s resources for one build, loaded lazily on expand ──────────────────────

function K8sResourcesSection({ buildId }: { buildId: string }) {
  const {
    data: resources,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["build-k8s", buildId],
    queryFn: () => getBuildK8sResources(buildId),
    staleTime: REFETCH,
    retry: false,
  });

  const byKind = useMemo(() => {
    if (!resources?.length) return {};
    return resources.reduce<Record<string, K8sResource[]>>((acc, r) => {
      (acc[r.kind] ??= []).push(r);
      return acc;
    }, {});
  }, [resources]);

  if (isLoading)
    return (
      <div style={{ marginTop: "0.75rem" }}>
        <SkeletonText paragraph lineCount={3} />
      </div>
    );

  if (isError || resources == null)
    return (
      <p
        style={{
          color: "var(--cds-text-secondary)",
          fontSize: "0.875rem",
          marginTop: "0.75rem",
        }}
      >
        K8s data unavailable — database may not be configured.
      </p>
    );
  if (!resources.length)
    return (
      <p
        style={{
          color: "var(--cds-text-secondary)",
          fontSize: "0.875rem",
          marginTop: "0.75rem",
        }}
      >
        No K8s resources recorded for this build.
      </p>
    );

  return (
    <div
      style={{
        marginTop: "0.75rem",
        paddingTop: "0.75rem",
      }}
    >
      <Accordion size="sm" align="start">
        {Object.entries(byKind).map(([kind, items]) => (
          <AccordionItem key={kind} title={`${kind} (${items.length})`}>
            <table className={styles.k8sTable}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Namespace</th>
                  <th>Status</th>
                  <th>CPU</th>
                  <th>Memory</th>
                  <th>GPU</th>
                </tr>
              </thead>
              <tbody>
                {items.map((r) => (
                  <tr key={`${r.kind}/${r.name}/${r.namespace}`}>
                    <td>
                      <code style={{ fontSize: "0.75rem" }}>{r.name}</code>
                      {r.failure_message && (
                        <p
                          style={{
                            margin: "0.125rem 0 0",
                            fontSize: "0.75rem",
                            color: "var(--cds-support-error)",
                          }}
                        >
                          {r.failure_message}
                        </p>
                      )}
                    </td>
                    <td style={{ color: "var(--cds-text-secondary)" }}>
                      {r.namespace ?? "—"}
                    </td>
                    <td>
                      {r.build_status ? (
                        <BuildStatusBadge
                          status={r.build_status as never}
                          showLabel
                        />
                      ) : (
                        <span style={{ color: "var(--cds-text-secondary)" }}>
                          {r.status ?? "—"}
                        </span>
                      )}
                    </td>
                    <td style={{ color: "var(--cds-text-secondary)" }}>
                      {r.cpu ?? "—"}
                    </td>
                    <td style={{ color: "var(--cds-text-secondary)" }}>
                      {r.memory ?? "—"}
                    </td>
                    <td>
                      {r.gpu != null && r.gpu > 0 ? (
                        <Tag type="purple" size="sm">
                          ×{r.gpu}
                        </Tag>
                      ) : (
                        <span style={{ color: "var(--cds-text-secondary)" }}>
                          —
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  );
}

// ── One card per running build ────────────────────────────────────────────────

function BuildWorkloadCard({
  build,
  resources,
}: {
  build: Build;
  resources?: BuildResources;
}) {
  const router = useRouter();
  const [expanded, setExpanded] = useState(false);

  const age = build.created_time
    ? (() => {
        const ms = Date.now() - new Date(build.created_time).getTime();
        const h = Math.floor(ms / 3_600_000);
        const m = Math.floor((ms % 3_600_000) / 60_000);
        if (h >= 24) {
          const d = Math.floor(h / 24);
          const rh = h % 24;
          return rh > 0 ? `${d}d ${rh}h` : `${d}d`;
        }
        return h > 0 ? `${h}h ${m}m` : `${m}m`;
      })()
    : null;

  return (
    <Tile className={styles.buildCard}>
      {/* Clickable header → navigate to build */}
      <div
        className={styles.buildCardHeader}
        onClick={() => router.push(`/builds/${build.uuid}`)}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            minWidth: 0,
          }}
        >
          <span
            style={{
              fontWeight: 600,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {build.name}
          </span>
          {resources?.gpu != null && resources.gpu > 0 && (
            <Tag type="purple" size="sm">
              {resources.gpu} GPU
            </Tag>
          )}
          {resources?.cpu && (
            <Tag type="teal" size="sm">
              {resources.cpu} CPU
            </Tag>
          )}
          {resources?.memory && (
            <Tag type="blue" size="sm">
              {resources.memory}
            </Tag>
          )}
        </div>

        {age && (
          <span
            style={{
              fontSize: "0.875rem",
              color: "var(--cds-text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "0.2rem",
            }}
          >
            <Time size={12} />
            {age}
          </span>
        )}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.75rem",
          flexShrink: 0,
          marginTop: "0.5rem",
        }}
      >
        <span
          style={{
            fontSize: "0.875rem",
            color: "var(--cds-text-secondary)",
            display: "flex",
            alignItems: "center",
            gap: "0.2rem",
          }}
        >
          <Folder size={12} />
          {build.space_name}
        </span>
        <span
          style={{
            fontSize: "0.875rem",
            color: "var(--cds-text-secondary)",
            display: "flex",
            alignItems: "center",
            gap: "0.2rem",
          }}
        >
          <User size={12} />
          {build.username}
        </span>
      </div>
      <K8sResourcesSection buildId={build.uuid} />
    </Tile>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function WorkloadsPage() {
  const configured = isSidecarConfigured();
  const searchParams = useSearchParams();
  const envId = searchParams.get(ENV_OVERRIDE_PARAM) ?? undefined;

  const { data: queues } = useQuery({
    queryKey: ["queue-capacity", envId],
    queryFn: getQueueCapacity,
    refetchInterval: REFETCH,
    enabled: configured,
  });

  const { data: nodes } = useQuery({
    queryKey: ["node-pools", envId],
    queryFn: getNodePools,
    refetchInterval: REFETCH,
    enabled: configured,
  });

  const { data: leaderboard } = useQuery({
    queryKey: ["leaderboard", "running_jobs", envId],
    queryFn: () => getLeaderboard("running_jobs"),
    refetchInterval: REFETCH,
    enabled: configured,
  });

  const { data: activeBuilds, isLoading: buildsLoading } = useQuery({
    queryKey: ["active-builds", envId],
    queryFn: () => listBuilds({ status: [...ACTIVE_STATUSES], page_size: 100 }),
    refetchInterval: REFETCH,
  });

  const activeBuildIds = useMemo(
    () => (activeBuilds?.items ?? []).map((b) => b.uuid),
    [activeBuilds],
  );

  const { data: buildResources } = useQuery({
    queryKey: ["build-resources", activeBuildIds, envId],
    queryFn: () => getBuildResources(activeBuildIds),
    enabled: activeBuildIds.length > 0,
    refetchInterval: REFETCH,
  });

  const resourceMap = useMemo(() => {
    const map = new Map<string, BuildResources>();
    for (const r of buildResources ?? []) map.set(r.build_id, r);
    return map;
  }, [buildResources]);

  // Derive summary stats from env-specific active builds + their resources.
  // Queue/node data from the K8s sidecar is not env-aware, so we don't use it
  // for the headline numbers.
  const summary = useMemo(() => {
    const items = activeBuilds?.items ?? [];
    const resources = buildResources ?? [];
    let gpuUsed = 0;
    let cpuMilliUsed = 0;
    for (const r of resources) {
      gpuUsed += r.gpu ?? 0;
      if (r.cpu) {
        const c = r.cpu.trim();
        cpuMilliUsed += c.endsWith("m") ? parseFloat(c) : parseFloat(c) * 1000;
      }
    }
    return {
      active: items.length,
      pending: items.filter(
        (b) => b.status === "pending" || b.status === "submitted",
      ).length,
      gpuUsed,
      // K8s queue capacity for GPU total — falls back to 0 if no K8s data
      gpuTotal: (queues ?? []).reduce((s, x) => s + x.gpu_capacity, 0),
      cpuUsed: cpuMilliUsed / 1000,
    };
  }, [activeBuilds, buildResources, queues]);

  const totalRunningPods = useMemo(
    () => (nodes ?? []).reduce((s, n) => s + n.running_pods, 0),
    [nodes],
  );

  if (!configured) return <NotConfigured />;

  const builds = (activeBuilds?.items ?? [])
    .slice()
    .sort(
      (a, b) =>
        new Date(b.created_time).getTime() - new Date(a.created_time).getTime(),
    );

  return (
    <div style={{ padding: "1.5rem" }}>
      <PageHeader
        crumbs={[{ label: "Granite.build", to: "/" }, { label: "Workloads" }]}
      />
      <h4 style={{ marginBottom: "2rem" }}>Workloads</h4>

      {/* ── Summary stats ── */}
      <div className={styles.statsRow}>
        <StatTile
          label="Active workloads"
          value={summary.active}
          loading={buildsLoading}
        />
        <StatTile
          label="Pending"
          value={summary.pending}
          loading={buildsLoading}
        />
        <StatTile
          label="Running pods"
          value={totalRunningPods}
          loading={false}
        />
        <StatTile
          label="GPUs in use"
          value={summary.gpuTotal > 0 ? `${summary.gpuUsed} / ${summary.gpuTotal}` : summary.gpuUsed}
          loading={buildsLoading}
        />
        <StatTile
          label="CPU cores in use"
          value={summary.cpuUsed.toFixed(0)}
          loading={buildsLoading}
        />
      </div>

      {/* ── Active builds ── */}
      <div style={{ marginTop: "2rem" }}>
        <h5 style={{ marginBottom: "1rem" }}>
          Active builds
          {builds.length > 0 && (
            <Tag type="blue" size="sm" style={{ marginLeft: "0.5rem" }}>
              {builds.length}
            </Tag>
          )}
        </h5>
        {buildsLoading ? (
          <SkeletonText paragraph lineCount={6} />
        ) : builds.length === 0 ? (
          <p style={{ color: "var(--cds-text-secondary)" }}>
            No active builds.
          </p>
        ) : (
          <div className={styles.buildGrid}>
            {builds.map((build) => (
              <BuildWorkloadCard
                key={build.uuid}
                build={build}
                resources={resourceMap.get(build.uuid)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
