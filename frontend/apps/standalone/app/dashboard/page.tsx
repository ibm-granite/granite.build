"use client";

import { useEffect, useState } from "react";
import styles from "./page.module.scss";
import { DonutChart, StackedBarChart } from "@carbon/charts-react";
import {
  type DonutChartOptions,
  type StackedBarChartOptions,
  type ChartTabularData,
  ScaleTypes,
} from "@carbon/charts";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ClickableTile,
  Dropdown,
  Layer,
  SkeletonText,
  Tag,
  Tile,
} from "@carbon/react";
import { Folder, User } from "@carbon/icons-react";
import { useQuery } from "@tanstack/react-query";
import { listBuilds, getBuildCount, listSpaces } from "@granite-build/ui-core/api/gbserver";
import {
  getBuildStatusChart,
} from "@granite-build/ui-core/api/analytics";
import { useChartsTheme } from "@granite-build/ui-core/hooks/useTheme";
import { BuildStatusBadge } from "@granite-build/ui-core/components/BuildStatusBadge";
import { BaseTile } from "@granite-build/ui-core/components/BaseTile";
import type { Build } from "@granite-build/ui-core/types";

// ── constants ─────────────────────────────────────────────────────────────────

// Mirrors gbserver's Status enum (src/gbserver/types/status.py).
const BUILD_STATUS_OPTS = [
  { id: "", label: "All jobs" },
  { id: "running", label: "Running jobs" },
  { id: "success", label: "Succeeded jobs" },
  { id: "failed", label: "Failed jobs" },
  { id: "invalid", label: "Invalid jobs" },
  { id: "pending", label: "Pending jobs" },
  { id: "submitted", label: "Submitted jobs" },
  { id: "retry_pending", label: "Retrying jobs" },
  { id: "cancel_requested", label: "Cancelling jobs" },
  { id: "cancelled", label: "Cancelled jobs" },
];

// Mirrors gbserver's Status enum (src/gbserver/types/status.py), in display order.
const BUILD_STATUS_CHART_LABELS: Record<string, string> = {
  submitted: "Submitted",
  pending: "Pending",
  running: "Running",
  success: "Succeeded",
  failed: "Failed",
  invalid: "Invalid",
  retry_pending: "Retrying",
  cancel_requested: "Cancelling",
  cancelled: "Cancelled",
};

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
  const username = 'standalone'
  const theme = useChartsTheme();

  const { data, isFetching, refetch } = useQuery({
    queryKey: ["my-recent-builds", username],
    queryFn: () => listBuilds({ username: username! }),
    enabled: !!username,
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)

  const builds = data?.items ?? [];
  const statusCounts = new Map<string, number>();
  for (const b of builds) {
    statusCounts.set(b.status, (statusCounts.get(b.status) ?? 0) + 1);
  }

  // Only non-zero statuses get a slice — keeps the legend focused on what's
  // actually present, while still covering every status gbserver can report.
  const chartData: ChartTabularData = Object.entries(BUILD_STATUS_CHART_LABELS)
    .map(([status, label]) => ({ group: label, value: statusCounts.get(status) ?? 0 }))
    .filter((d) => d.value > 0);
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
// ── AutoTuneX tiles ─────────────────────────────────────────────────────

// A build counts as model-customisation activity under any of these tags — the
// same three spellings BuildDetails.tsx gates its AutoTuneX tabs on. Keep the two
// lists in step: a tag only one of them knows about makes the tile disagree with
// the build page.
const MODEL_CUSTOMISATION_TAGS = [
  "model-customization",
  "model-customisation",
  "autotunex",
];

// The gbserver tags filter is AND-only, so one tag per request, then dedupe by
// uuid (a build carrying two spellings would otherwise be counted twice).
async function fetchModelCustomisationStats() {
  const results = await Promise.all(
    MODEL_CUSTOMISATION_TAGS.map((tag) => listBuilds({ tags: [tag] })),
  );
  const byId = new Map<string, Build>();
  for (const r of results) for (const b of r.items) byId.set(b.uuid, b);
  const builds = Array.from(byId.values());

  return {
    total: builds.length,
    running: builds.filter((b) => b.status === "running").length,
    pending: builds.filter((b) => b.status === "pending" || b.status === "submitted").length,
    completed: builds.filter((b) => b.status === "success").length,
    failed: builds.filter((b) => b.status === "failed").length,
    cancelled: builds.filter((b) => b.status === "cancelled").length,
  };
}

function AutoTuneXTile() {
  const startTuningLink = (
    <Link
      href="/dashboard/autotunex/start-tuning"
      className="cds--link"
      style={{ fontSize: "0.875rem" }}
    >
      Start tuning
    </Link>
  );

  const { data: stats, isFetching, refetch } = useQuery({
    queryKey: ["model-customisation-stats"],
    queryFn: fetchModelCustomisationStats,
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching);
  return (
    <BaseTile
      title="Model Customization"
      action={startTuningLink}
      isRefreshing={isRefreshing}
      onRefresh={() => { markRefreshing(); void refetch() }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
        <StatRow label="Total tunings" value={stats?.total ?? "—"} />
        <StatRow label="Running" value={stats?.running ?? "—"} />
        <StatRow label="Pending" value={stats?.pending ?? "—"} />
        <StatRow label="Completed" value={stats?.completed ?? "—"} />
        <StatRow label="Cancelled" value={stats?.cancelled ?? "—"} />
        <StatRow label="Failed" value={stats?.failed ?? "—"} />
      </div>
    </BaseTile>
  );
}

function ClusterStatusTile() {
  const { data: todayData, isFetching, refetch } = useQuery({
    queryKey: ["builds-today"],
    queryFn: () => getBuildStatusChart(1, false),
  });
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)

  const today = todayData?.[todayData.length - 1];
  const todayRunning = today?.running ?? 0;
  const todaySucceeded = today?.success ?? 0;
  const todayFailed = today?.failed ?? 0;

  return (
    <BaseTile
      title="Build status"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
        <StatRow label="Running" value={todayRunning} />
        <StatRow label="Succeeded today" value={todaySucceeded} />
        <StatRow label="Failed today" value={todayFailed} />
      </div>
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
  const { data, isFetching, isError, refetch } = useQuery({
    queryKey: ["build-volume"],
    queryFn: () => getBuildStatusChart(14, false),
    retry: 3,
    retryDelay: 2000,
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

  const isEmpty = !isFetching && !isError && (!data || data.length === 0);
  const isUnavailable = !isFetching && isError;

  return (
    <BaseTile
      title="Build volume (14 days)"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      {isEmpty ? (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "200px", color: "var(--cds-text-secondary)", fontSize: "0.875rem" }}>
          No build data
        </div>
      ) : isUnavailable ? (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "200px", color: "var(--cds-text-secondary)", fontSize: "0.875rem" }}>
          Analytics unavailable
        </div>
      ) : (
        <StackedBarChart data={chartData} options={opts} />
      )}
    </BaseTile>
  );
}

// ── Standalone-only tiles ─────────────────────────────────────────────────────

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const m = Math.floor(seconds / 60), s = seconds % 60
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`
  const h = Math.floor(m / 60), rm = m % 60
  return rm > 0 ? `${h}h ${rm}m` : `${h}h`
}

function BuildStatsTile() {
  const { data: stats, isFetching, refetch } = useQuery({
    queryKey: ["build-stats"],
    queryFn: async () => {
      const [total, recent] = await Promise.all([
        getBuildCount({}),
        listBuilds({ page_size: 200, sort: "created_time:desc" }),
      ])
      const completed = recent.items.filter((b) =>
        b.status === "success" || b.status === "failed"
      )
      const succeeded = completed.filter((b) => b.status === "success").length
      const successRate =
        completed.length > 0
          ? Math.round((succeeded / completed.length) * 100)
          : null
      const durations = completed
        .filter((b) => b.created_time && b.finished_at)
        .map(
          (b) =>
            (new Date(b.finished_at!).getTime() -
              new Date(b.created_time).getTime()) /
            1000
        )
        .filter((d) => d > 0)
      const avgDuration =
        durations.length > 0
          ? Math.round(
              durations.reduce((a, b) => a + b, 0) / durations.length
            )
          : null
      return { total, successRate, avgDuration, completedCount: completed.length }
    },
  })
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)

  return (
    <BaseTile
      title="Build statistics"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
        <StatRow label="Total builds" value={stats?.total ?? "—"} />
        <StatRow
          label="Success rate"
          value={
            stats?.successRate != null ? `${stats.successRate}%` : "—"
          }
        />
        <StatRow
          label="Avg duration"
          value={
            stats?.avgDuration != null
              ? formatDuration(stats.avgDuration)
              : "—"
          }
        />
        <StatDivider />
        <StatRow
          label="Sample size"
          value={
            stats?.completedCount != null
              ? `${stats.completedCount} completed builds`
              : "—"
          }
        />
      </div>
    </BaseTile>
  )
}

function SpacesOverviewTile() {
  const { data: spaces, isFetching, refetch } = useQuery({
    queryKey: ["spaces-overview"],
    queryFn: async () => {
      const spaces = await listSpaces()
      const counts = await Promise.all(
        spaces.map((s) => getBuildCount({ space_name: s.name }))
      )
      return spaces
        .map((s, i) => ({ ...s, buildCount: counts[i] }))
        .sort((a, b) => b.buildCount - a.buildCount)
    },
  })
  const [isRefreshing, markRefreshing] = useRefreshState(isFetching)
  const isEmpty = !isFetching && (!spaces || spaces.length === 0)

  return (
    <BaseTile
      title="Spaces"
      onRefresh={() => { markRefreshing(); void refetch() }}
      isRefreshing={isRefreshing}
    >
      {isEmpty ? (
        <div style={{ color: "var(--cds-text-secondary)", fontSize: "0.875rem" }}>
          No spaces registered
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.375rem" }}>
          {(spaces ?? []).map((s) => (
            <StatRow key={s.uuid} label={s.name} value={s.buildCount} />
          ))}
        </div>
      )}
    </BaseTile>
  )
}

// ── Builds ────────────────────────────────────────────────────────────────────

function BuildTile({ build }: { build: Build }) {
  const router = useRouter();
  return (
    <ClickableTile
      id={`home-build-${build.uuid}`}
      onClick={() => router.push(`/dashboard/builds/_/?id=${build.uuid}`)}
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
        page_size: 4,
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
        href="/dashboard/builds"
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
              gridTemplateColumns: "1fr",
              gap: "1rem",
            }}
          >
            {Array.from({ length: 4 }).map((_, i) => (
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
              gridTemplateColumns: "1fr",
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

// ── Home page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
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
          <h2>Hi there.</h2>
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
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: "1rem",
          marginBottom: "1rem",
          alignItems: "stretch",
        }}
      >
        <MyBuildsTile />
        <AutoTuneXTile />
        <ClusterStatusTile />
        <SpacesOverviewTile />
        <BuildStatsTile />
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
        <BuildsPanel />
        <BuildVolumeSparkline />
      </div>
    </div>
  );
}
