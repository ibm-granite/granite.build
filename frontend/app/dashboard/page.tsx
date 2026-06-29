"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.scss";
import { DonutChart, GaugeChart, StackedBarChart } from "@carbon/charts-react";
import {
  type DonutChartOptions,
  type GaugeChartOptions,
  type StackedBarChartOptions,
  type ChartTabularData,
  ScaleTypes,
} from "@carbon/charts";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  ClickableTile,
  Dropdown,
  Layer,
  ProgressBar,
  SkeletonText,
  Tag,
  Tile,
} from "@carbon/react";
import { Folder, User } from "@carbon/icons-react";
import { useQuery } from "@tanstack/react-query";
import { listBuilds, getBuildCount } from "@/api/gbserver";
import {
  getBuildStatusChart,
  getLeaderboard,
  getQueueCapacity,
  getNodePools,
  isSidecarConfigured,
} from "@/api/analytics";
import { useAuth } from "@/auth/useAuth";
import { useChartsTheme } from "@/hooks/useTheme";
import { BuildStatusBadge } from "@/components/BuildStatusBadge";
import { BaseTile } from "@/components/BaseTile";
import type { Build, LeaderboardEntry, QueueCapacity, NodePool } from "@/types";

// ── constants ─────────────────────────────────────────────────────────────────

const LB_VIEWS = [
  { id: "running_jobs", label: "Running jobs" },
  { id: "gpu", label: "GPU usage" },
  { id: "cpu", label: "CPU usage" },
  { id: "memory", label: "Memory usage" },
  { id: "total_builds", label: "Total builds" },
] as const;
type LbView = (typeof LB_VIEWS)[number]["id"];

const BUILD_STATUS_OPTS = [
  { id: "", label: "All jobs" },
  { id: "running", label: "Running jobs" },
  { id: "failed", label: "Failed jobs" },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

// Tracks whether *this tile* triggered a refresh, independent of shared query keys.
function useRefreshState(isFetching: boolean): [boolean, () => void] {
  const [isRefreshing, setIsRefreshing] = useState(false)
  useEffect(() => { if (!isFetching) setIsRefreshing(false) }, [isFetching])
  return [isRefreshing, () => setIsRefreshing(true)]
}

// ── Shared sub-components ─────────────────────────────────────────────────────

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
      }}
    >
      <span
        className="cds--helper-text-01"
        style={{ color: "var(--cds-text-secondary)" }}
      >
        {label}
      </span>
      <span className="cds--body-short-01" style={{ fontWeight: 600 }}>
        {value}
      </span>
    </div>
  );
}

function StatDivider() {
  return (
    <div
      style={{
        borderTop: "1px solid var(--cds-border-subtle)",
        margin: "0.5rem 0",
      }}
    />
  );
}

// ── Summary tiles ─────────────────────────────────────────────────────

function MyBuildsTile() {
  const { auth } = useAuth();
  const username = auth?.username;
  const theme = useChartsTheme();

  const { data, isFetching, refetch } = useQuery({
    queryKey: ["my-recent-builds", username],
    queryFn: () => listBuilds({ username: username! }),
    enabled: !!username,
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)

  const builds = data?.items ?? [];
  const running = builds.filter((b) => b.status === "running").length;
  const succeeded = builds.filter((b) => b.status === "success").length;
  const failed = builds.filter((b) => b.status === "failed").length;

  const chartData: ChartTabularData = [
    { group: "Running", value: running },
    { group: "Succeeded", value: succeeded },
    { group: "Failed", value: failed },
  ];
  const chartOptions: DonutChartOptions = {
    donut: {
      center: { label: "total builds", number: builds.length },
      alignment: "center",
    },
    legend: { position: "bottom" },
    height: "220px",
    toolbar: { enabled: false },
    theme,
    data: { loading: isFetching || !username },
    color: {
      pairing: {
        option: 4,
      },
    },
  };

  return (
    <BaseTile
      title="My builds"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      <DonutChart data={chartData} options={chartOptions} />
    </BaseTile>
  );
}

function ClusterStatusTile() {
  const configured = isSidecarConfigured();
  const { data: queues, isFetching: isFetchingQueues, refetch: refetchQueues } = useQuery({
    queryKey: ["queue-capacity"],
    queryFn: getQueueCapacity,
    enabled: configured,
  });
  const { data: nodes, isFetching: isFetchingNodes, refetch: refetchNodes } = useQuery({
    queryKey: ["node-pools"],
    queryFn: getNodePools,
    enabled: configured,
  });
  const { data: todayData, isFetching: isFetchingToday, refetch: refetchToday } = useQuery({
    queryKey: ["builds-today"],
    queryFn: () => getBuildStatusChart(1, false),
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetchingQueues || isFetchingNodes || isFetchingToday)

  const runningPods = (nodes ?? []).reduce((s, n) => s + n.running_pods, 0);
  const pendingPods = (nodes ?? []).reduce((s, n) => s + n.pending_pods, 0);
  const pendingJobs = (queues ?? []).reduce(
    (s, q) => s + q.pending_workloads,
    0,
  );
  const admittedJobs = (queues ?? []).reduce(
    (s, q) => s + q.admitted_workloads + q.reserving_workloads,
    0,
  );

  const today = todayData?.[todayData.length - 1];
  const todayRunning = today?.running ?? 0;
  const todaySucceeded = today?.success ?? 0;
  const todayFailed = today?.failed ?? 0;

  return (
    <BaseTile
      title="Cluster status"
      onRefresh={() => { markRefreshing(); void refetchQueues(); void refetchNodes(); void refetchToday() }}
      isRefreshing={isRefreshing}
    >
      <div
        style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}
      >
        <StatRow label="Running builds" value={todayRunning} />
        <StatRow label="Succeeded today" value={todaySucceeded} />
        <StatRow label="Failed today" value={todayFailed} />
        <StatDivider />
        <StatRow label="Running pods" value={runningPods} />
        <StatRow label="Pending pods" value={pendingPods} />
        <StatRow label="Active jobs" value={admittedJobs} />
        <StatRow label="Queued jobs" value={pendingJobs} />
      </div>
    </BaseTile>
  );
}

function NodePoolsTile() {
  const configured = isSidecarConfigured();
  const {
    data: nodes,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ["node-pools"],
    queryFn: getNodePools,
    enabled: configured,
  });

  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)
  const pools = nodes ?? [];

  return (
    <BaseTile
      title="Node pools"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      {!configured || isLoading ? (
        <SkeletonText paragraph lineCount={3} />
      ) : pools.length === 0 ? (
        <p
          className="cds--body-short-01"
          style={{ color: "var(--cds-text-secondary)" }}
        >
          No node data
        </p>
      ) : (
        pools.map((pool) => (
          <NodePoolEntry
            key={`${pool.cluster_name}/${pool.pool_name}`}
            pool={pool}
          />
        ))
      )}
    </BaseTile>
  );
}

function GpuAvailabilityTile() {
  const configured = isSidecarConfigured();
  const { data: queues, isFetching: isFetchingQueues, refetch: refetchQueues } = useQuery({
    queryKey: ["queue-capacity"],
    queryFn: getQueueCapacity,
    enabled: configured,
  });
  const { data: nodes, isFetching: isFetchingNodes, refetch: refetchNodes } = useQuery({
    queryKey: ["node-pools"],
    queryFn: getNodePools,
    enabled: configured,
  });
  const isFetchingGpu = isFetchingQueues || isFetchingNodes;
  const [isRefreshing, markRefreshing] = useRefreshState(isFetchingGpu)

  const totalCapacity = (queues ?? []).reduce((s, q) => s + q.gpu_capacity, 0);
  const totalUsed = (queues ?? []).reduce((s, q) => s + q.gpu_used, 0);
  const available = totalCapacity - totalUsed;
  const pct =
    totalCapacity > 0 ? Math.round((totalUsed / totalCapacity) * 100) : 0;

  const scalePools = (nodes ?? []).filter(
    (n) =>
      n.autoscale_enabled && n.max_nodes != null && n.node_count < n.max_nodes!,
  );
  const scaleHeadroom = scalePools.reduce(
    (s, n) => s + (n.max_nodes! - n.node_count),
    0,
  );

  const gaugeColor = pct > 95 ? "#da1e28" : pct >= 80 ? "#ff832b" : "#24a148";
  const theme = useChartsTheme();

  const gaugeOptions: GaugeChartOptions = {
    resizable: true,
    gauge: { type: "semi", alignment: "center" },
    color: { scale: { value: gaugeColor } },
    height: "8rem",
    toolbar: { enabled: false },
    theme,
    data: { loading: isFetchingGpu },
  };

  return (
    <BaseTile
      title="GPU availability"
      onRefresh={() => { markRefreshing(); void refetchQueues(); void refetchNodes() }}
      isRefreshing={isRefreshing}
    >
      {totalCapacity === 0 ? (
        <p
          className="cds--body-short-01"
          style={{ color: "var(--cds-text-secondary)" }}
        >
          No GPU data
        </p>
      ) : (
        <>
          <p
            className="cds--helper-text-01"
            style={{
              color: "var(--cds-text-secondary)",
              marginBottom: "1rem",
              textAlign: "center",
            }}
          >
            {available} / {totalCapacity} available · {pct}% in use
          </p>
          <GaugeChart
            data={[{ group: "value", value: pct }]}
            options={gaugeOptions}
          />
          {scaleHeadroom > 0 && (
            <p
              className="cds--helper-text-01"
              style={{
                color: "var(--cds-text-secondary)",
                marginTop: "1rem",
                textAlign: "center",
              }}
            >
              ↑ {scaleHeadroom} more nodes via autoscale ({scalePools.length}{" "}
              pool{scalePools.length > 1 ? "s" : ""})
            </p>
          )}
        </>
      )}
    </BaseTile>
  );
}

// ── Chart tiles ───────────────────────────────────────────────────────

const BUILD_VOLUME_OPTIONS: StackedBarChartOptions = {
  axes: {
    left: { mapsTo: "value", stacked: true, title: "Builds" },
    bottom: { mapsTo: "date", scaleType: ScaleTypes.TIME },
  },
  height: "420px",
  toolbar: { enabled: false },
  legend: { alignment: "center" },
};

function BuildVolumeSparkline() {
  const { data, isFetching, refetch } = useQuery({
    queryKey: ["build-volume"],
    queryFn: () => getBuildStatusChart(14, false),
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)

  const chartData: ChartTabularData = (data ?? []).flatMap((p) => [
    {
      group: "Success",
      date: new Date(p.date + "T12:00:00"),
      value: p.success,
    },
    { group: "Failed", date: new Date(p.date + "T12:00:00"), value: p.failed },
    {
      group: "Running",
      date: new Date(p.date + "T12:00:00"),
      value: p.running,
    },
    {
      group: "Queued",
      date: new Date(p.date + "T12:00:00"),
      value: p.pending + p.submitted,
    },
  ]);

  const theme = useChartsTheme();
  const opts: StackedBarChartOptions = {
    ...BUILD_VOLUME_OPTIONS,
    theme,
    data: { loading: isFetching },
  };

  return (
    <BaseTile
      title="Build volume (14 days)"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      <StackedBarChart data={chartData} options={opts} />
    </BaseTile>
  );
}

// ── Leaderboard ───────────────────────────────────────────────────────────────

function LeaderboardPanel() {
  const [lbView, setLbView] = useState<LbView>("total_builds");
  const configured = isSidecarConfigured();
  const { data, isFetching, refetch } = useQuery({
    queryKey: ["leaderboard", lbView],
    queryFn: () => getLeaderboard(lbView),
    enabled: configured,
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)
  useEffect(() => {
    refetch();
  }, [lbView, refetch]);
  const entries: LeaderboardEntry[] = isFetching ? [] : (data ?? []);

  function metricText(entry: LeaderboardEntry) {
    if (lbView === "running_jobs") return `${entry.running_jobs} running`;
    if (lbView === "gpu") return `${entry.gpu_count} GPU`;
    if (lbView === "cpu") return `${entry.cpu_cores.toFixed(1)} cores`;
    if (lbView === "memory") return `${entry.memory_gib.toFixed(1)} GiB`;
    return `${entry.total_builds} builds`;
  }

  const dropdown = (
    <div style={{ position: "relative" }} className={styles.dropdownInline}>
      <Dropdown
        id="home-lb-view"
        titleText=""
        label=""
        size="sm"
        items={[...LB_VIEWS]}
        itemToString={(i) => i?.label ?? ""}
        selectedItem={LB_VIEWS.find((v) => v.id === lbView) ?? LB_VIEWS[4]}
        type="inline"
        onChange={({ selectedItem }) =>
          selectedItem && setLbView(selectedItem.id)
        }
      />
    </div>
  );

  return (
    <BaseTile
      title="Leaderboard"
      action={dropdown}
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      {isFetching ? (
        <div
          style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}
        >
          {Array.from({ length: 8 }).map((_, i) => (
            <SkeletonText key={i} lineCount={1} />
          ))}
        </div>
      ) : entries.length === 0 ? (
        <p
          className="cds--body-short-01"
          style={{ color: "var(--cds-text-secondary)" }}
        >
          No data
        </p>
      ) : (
        <div>
          {entries.slice(0, 8).map((entry, i) => (
            <div
              key={entry.username}
              style={{
                display: "flex",
                alignItems: "baseline",
                padding: "1rem 0",
              }}
            >
              <span
                style={{
                  fontSize: "0.875rem",
                  color: "var(--cds-text-secondary)",
                  width: "1.75rem",
                  flexShrink: 0,
                  paddingRight: "0.5rem",
                }}
              >
                #{i + 1}
              </span>
              <span
                className="cds--body-short-01"
                style={{
                  color: "var(--cds-text-primary)",
                  flex: "1 1 auto",
                  paddingRight: "0.75rem",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {entry.username}
              </span>
              <span
                style={{
                  display: "flex",
                  alignItems: "center",
                  flexShrink: 0,
                  whiteSpace: "nowrap",
                }}
              >
                <span
                  className="cds--helper-text-01"
                  style={{ color: "var(--cds-text-secondary)" }}
                >
                  {metricText(entry)}
                </span>
                {lbView === "running_jobs" && entry.gpu_count > 0 && (
                  <Tag
                    type="purple"
                    size="sm"
                    style={{ marginLeft: "0.375rem" }}
                  >
                    {entry.gpu_count} GPU
                  </Tag>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </BaseTile>
  );
}

// ── Builds ────────────────────────────────────────────────────────────────────

function BuildTile({ build }: { build: Build }) {
  const router = useRouter();
  return (
    <ClickableTile
      id={`home-build-${build.uuid}`}
      onClick={() => router.push(`/builds/${build.uuid}`)}
      style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}
    >
      <div className={styles.buildTileHeader}>
        <p
          className="cds--body-short-02"
          style={{
            fontWeight: 600,
            color: "var(--cds-text-primary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            margin: 0,
          }}
        >
          {build.name}
        </p>
        <Layer>
          <BuildStatusBadge status={build.status} />
        </Layer>
      </div>

      <p
        className="cds--code-01"
        style={{
          color: "var(--cds-text-secondary)",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          margin: 0,
          fontSize: "0.875rem",
        }}
      >
        {build.uuid}
      </p>

      <div style={{ display: "flex", gap: "1rem", marginTop: "0.25rem" }}>
        <span
          className="cds--helper-text-01"
          style={{
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
          className="cds--helper-text-01"
          style={{
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
    </ClickableTile>
  );
}

function BuildTileSkeleton() {
  return (
    <Tile style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <SkeletonText width="60%" />
      <SkeletonText width="90%" />
      <SkeletonText width="35%" />
    </Tile>
  );
}

function BuildsPanel() {
  const [statusOpt, setStatusOpt] = useState(BUILD_STATUS_OPTS[0]);
  const { data, isFetching, refetch } = useQuery({
    queryKey: ["home-builds", statusOpt.id],
    queryFn: () =>
      listBuilds({
        status: statusOpt.id || undefined,
        sort: "created_time:desc",
        page_size: 10,
        page_index: 0,
      }),
  });
  const {
    data: total,
    isFetching: isTotalFetching,
    refetch: refetchTotal,
  } = useQuery({
    queryKey: ["home-builds-count", statusOpt.id],
    queryFn: () => getBuildCount({ status: statusOpt.id || undefined }),
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching || isTotalFetching)
  useEffect(() => {
    refetch();
    refetchTotal();
  }, [statusOpt.id, refetch, refetchTotal]);
  const builds: Build[] = isFetching ? [] : (data?.items ?? []);

  const controls = (
    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
      <Link
        href="/builds"
        className="cds--link"
        style={{ fontSize: "0.875rem" }}
      >
        View all builds
      </Link>
      <div style={{ position: "relative" }} className={styles.dropdownInline}>
        <Dropdown
          id="home-build-status"
          titleText=""
          label=""
          size="sm"
          items={BUILD_STATUS_OPTS}
          itemToString={(i) => i?.label ?? ""}
          selectedItem={statusOpt}
          type="inline"
          onChange={({ selectedItem }) =>
            selectedItem && setStatusOpt(selectedItem)
          }
        />
      </div>
    </div>
  );

  return (
    <BaseTile
      title="Builds"
      action={controls}
      onRefresh={() => { markRefreshing(); void refetch(); void refetchTotal() }}
      isRefreshing={isRefreshing}
    >
      {(isFetching || builds.length > 0) && (
        <div
          className="cds--helper-text-01"
          style={{
            color: "var(--cds-text-secondary)",
            margin: "-0.5rem 0 0.75rem",
            display: "flex",
            alignItems: "center",
            gap: "0.375rem",
          }}
        >
          {isFetching ? (
            <span style={{ display: "inline-block", width: "8rem" }}>
              <SkeletonText />
            </span>
          ) : (
            <>
              Showing 1–{builds.length} of{" "}
              {isTotalFetching ? (
                <span style={{ display: "inline-block", width: "2rem" }}>
                  <SkeletonText />
                </span>
              ) : (
                `${total}`
              )}{" "}
              builds
            </>
          )}
        </div>
      )}
      <Layer>
        {isFetching ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "1rem",
            }}
          >
            {Array.from({ length: 10 }).map((_, i) => (
              <BuildTileSkeleton key={i} />
            ))}
          </div>
        ) : builds.length === 0 ? (
          <p
            className="cds--body-short-01"
            style={{ color: "var(--cds-text-secondary)", padding: "1rem 0" }}
          >
            No {statusOpt.label.toLowerCase()} found.
          </p>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "1rem",
            }}
          >
            {builds.map((build) => (
              <BuildTile key={build.uuid} build={build} />
            ))}
          </div>
        )}
      </Layer>
    </BaseTile>
  );
}

// ── Compute Resources ─────────────────────────────────────────────────────────

function QueueEntry({ q }: { q: QueueCapacity }) {
  const running = q.admitted_workloads + q.reserving_workloads;
  return (
    <div style={{ marginBottom: "1.25rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: "0.5rem",
        }}
      >
        <div>
          <p className="cds--body-short-01">
            {q.name.replace(/-cluster-queue$/i, "")}
          </p>
          <p className={styles.queueClusterName}>{q.cluster_name}</p>
        </div>
        <span
          className="cds--helper-text-01"
          style={{
            color: "var(--cds-text-secondary)",
            flexShrink: 0,
            marginLeft: "0.5rem",
          }}
        >
          {running} running
        </span>
      </div>
      {q.gpu_capacity > 0 && (
        <ProgressBar
          label={`GPU  ${q.gpu_used.toFixed(0)} / ${q.gpu_capacity.toFixed(0)}`}
          value={q.gpu_used}
          max={q.gpu_capacity}
          status={q.gpu_used >= q.gpu_capacity ? "error" : "active"}
          className="home-gpu-bar"
        />
      )}
      <ProgressBar
        label={`CPU  ${q.cpu_used_cores.toFixed(0)} / ${q.cpu_capacity_cores.toFixed(0)}`}
        value={q.cpu_used_cores}
        max={q.cpu_capacity_cores}
        status={q.cpu_used_cores >= q.cpu_capacity_cores ? "error" : "active"}
        className="home-cpu-bar"
      />
      <ProgressBar
        label={`Mem  ${q.memory_used_gib.toFixed(0)} / ${q.memory_capacity_gib.toFixed(0)} GiB`}
        value={q.memory_used_gib}
        max={q.memory_capacity_gib}
        status={q.memory_used_gib >= q.memory_capacity_gib ? "error" : "active"}
        className="home-mem-bar"
      />
    </div>
  );
}

function NodePoolEntry({ pool }: { pool: NodePool }) {
  return (
    <div style={{ marginBottom: "1.25rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: "0.5rem",
        }}
      >
        <span
          className="cds--body-short-01"
          style={{
            fontWeight: 600,
            color: "var(--cds-text-primary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            marginRight: "0.5rem",
          }}
        >
          {pool.pool_name}
        </span>
        <span
          className="cds--helper-text-01"
          style={{ color: "var(--cds-text-secondary)", flexShrink: 0 }}
        >
          {pool.ready_nodes}/{pool.node_count} nodes
          {pool.autoscale_enabled ? " · auto" : ""}
        </span>
      </div>
      {pool.gpu_allocatable > 0 && (
        <ProgressBar
          label={`GPU  ${pool.gpu_requested.toFixed(0)} / ${pool.gpu_allocatable.toFixed(0)}`}
          value={pool.gpu_requested}
          max={pool.gpu_allocatable}
          status={
            pool.gpu_requested >= pool.gpu_allocatable ? "error" : "active"
          }
          className="home-gpu-bar"
        />
      )}
      <ProgressBar
        label={`CPU  ${pool.cpu_requested_cores.toFixed(0)} / ${pool.cpu_allocatable_cores.toFixed(0)}`}
        value={pool.cpu_requested_cores}
        max={pool.cpu_allocatable_cores}
        status={
          pool.cpu_requested_cores >= pool.cpu_allocatable_cores
            ? "error"
            : "active"
        }
        className="home-cpu-bar"
      />
      <ProgressBar
        label={`Mem  ${pool.memory_requested_gib.toFixed(0)} / ${pool.memory_allocatable_gib.toFixed(0)} GiB`}
        value={pool.memory_requested_gib}
        max={pool.memory_allocatable_gib}
        status={
          pool.memory_requested_gib >= pool.memory_allocatable_gib
            ? "error"
            : "active"
        }
        className="home-mem-bar"
      />
    </div>
  );
}

function ComputeResourcesPanel() {
  const configured = isSidecarConfigured();
  const { data: queues, isLoading, isFetching, refetch } = useQuery({
    queryKey: ["queue-capacity"],
    queryFn: getQueueCapacity,
    enabled: configured,
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)

  return (
    <BaseTile
      title="Queue capacity"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      {isLoading ? (
        <SkeletonText paragraph lineCount={3} />
      ) : (queues ?? []).length === 0 ? (
        <p
          className="cds--body-short-01"
          style={{ color: "var(--cds-text-secondary)" }}
        >
          No queue data
        </p>
      ) : (
        (queues ?? []).map((q) => (
          <QueueEntry key={`${q.cluster_name}/${q.name}`} q={q} />
        ))
      )}
    </BaseTile>
  );
}

// ── Home page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  const { auth } = useAuth();
  const username = auth?.username;

  return (
    <div style={{ padding: "2rem 1.5rem" }}>
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          marginBottom: "1.5rem",
        }}
      >
        <div>
          <h2
            // style={{ margin: "0 0 0.25rem", fontWeight: 300 }}
          >
            Hi {auth?.name || username ? `${auth?.name || username}.` : "there."}
          </h2>
          <p
            className="cds--body-short-01"
            style={{ color: "var(--cds-text-secondary)", margin: 0 }}
          >
            Welcome to Granite.build!
          </p>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: "1rem",
          marginBottom: "1rem",
          alignItems: "stretch",
        }}
      >
        <MyBuildsTile />
        <ClusterStatusTile />
        <NodePoolsTile />
        <GpuAvailabilityTile />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "2fr 1fr",
          gap: "1rem",
          marginBottom: "1rem",
          alignItems: "stretch",
        }}
      >
        <BuildsPanel />
        <ComputeResourcesPanel />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "1rem",
          marginBottom: "1rem",
          alignItems: "stretch",
        }}
      >
        <BuildVolumeSparkline />
        <LeaderboardPanel />
      </div>
    </div>
  );
}
