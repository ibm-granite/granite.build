"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { InlineLoading, InlineNotification, SkeletonText, Tag } from "@carbon/react";
import styles from "./page.module.scss";
import { useQuery } from "@tanstack/react-query";
import {
  getBuild,
  describeBuild,
  getBuildStatus,
  getBuildEvents,
} from "@granite-build/ui-core/api/gbserver";
import { BuildStatusBadge } from "@granite-build/ui-core/components/BuildStatusBadge";
import { PageHeader } from "@granite-build/ui-core/components/PageHeader";
import { BuildDetails } from "./BuildDetails";

const ACTIVE_STATUSES = new Set(["running", "submitted", "pending"]);

// useSearchParams() bails the page out to client-side rendering up to the
// nearest Suspense boundary during static export — without one here, the
// statically-exported HTML (built with no query param) and the client's
// first render (which already sees the real ?id=) disagree, tripping a
// hydration mismatch (React error #418). The fallback below matches what
// the static export produces so hydration has nothing to reconcile against.
export default function BuildDetailPage() {
  return (
    <Suspense fallback={<BuildDetailFallback />}>
      <BuildDetailContent />
    </Suspense>
  );
}

function BuildDetailFallback() {
  return (
    <div style={{ padding: "2rem 1.5rem 1.5rem" }}>
      <PageHeader
        crumbs={[
          { label: "Granite.build", to: "/" },
          { label: "Builds", to: "/dashboard/builds" },
          { label: "…" },
        ]}
      />
      <div className={styles.buildHeaderRow}>
        <SkeletonText width="300px" />
      </div>
    </div>
  );
}

function BuildDetailContent() {
  // The real id lives in the ?id= query param, not location.hash — a hash read
  // in a mount-only effect breaks when navigating from one build's page to
  // another without an intervening route change, since both are the same "_"
  // route to Next's router and a one-time hash read never sees the new id.
  // useSearchParams() is reactive, but it isn't enough by itself: Next's router
  // patches window.history, so our own cosmetic replaceState below (stripping
  // the query param once we've adopted the id) makes useSearchParams() briefly
  // report no id again. Latching the id into state — only ever overwritten by
  // a new *non-empty* param value — survives that revert.
  const searchParams = useSearchParams();
  const paramId = searchParams.get('id');
  const [buildId, setBuildId] = useState(paramId ?? '');

  useEffect(() => {
    if (paramId && paramId !== buildId) {
      setBuildId(paramId);
    }
  }, [paramId, buildId]);

  useEffect(() => {
    if (buildId) {
      window.history.replaceState(null, '', `/dashboard/builds/${buildId}/`);
    }
  }, [buildId]);

  const refetchInterval = (data: unknown) => {
    const b = data as { status?: string } | undefined;
    return b && ACTIVE_STATUSES.has(b.status ?? "") ? 30_000 : false;
  };

  const {
    data: build,
    isLoading: loadingBuild,
    dataUpdatedAt,
    error: buildError,
  } = useQuery({
    queryKey: ["build", buildId],
    queryFn: () => getBuild(buildId!),
    refetchInterval,
    enabled: Boolean(buildId),
  });

  // Show the indicator just before each expected 30 s refetch.
  // Driven by dataUpdatedAt so the countdown resets after every completed fetch.
  const [showRefreshing, setShowRefreshing] = useState(false);
  const refreshTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  useEffect(() => {
    refreshTimers.current.forEach(clearTimeout);
    refreshTimers.current = [];

    if (!build || !ACTIVE_STATUSES.has(build.status) || loadingBuild) {
      setShowRefreshing(false);
      return;
    }

    refreshTimers.current = [
      setTimeout(() => setShowRefreshing(true),  29_500),
      setTimeout(() => setShowRefreshing(false), 32_000),
    ];
    return () => refreshTimers.current.forEach(clearTimeout);
  }, [build?.status, dataUpdatedAt, loadingBuild]);

  const { data: describe } = useQuery({
    queryKey: ["build-describe", buildId],
    queryFn: () => describeBuild(buildId!),
    enabled: Boolean(buildId),
  });

  const {
    data: status,
    isLoading: loadingStatus,
    error: statusError,
  } = useQuery({
    queryKey: ["build-status", buildId],
    queryFn: () => getBuildStatus(buildId!),
    refetchInterval: () =>
      build && ACTIVE_STATUSES.has(build.status) ? 30_000 : false,
    enabled: Boolean(buildId),
    retry: 1,
  });

  const { data: events = [] } = useQuery({
    queryKey: ["build-events", buildId],
    queryFn: () => getBuildEvents(buildId!),
    refetchInterval: () =>
      build && ACTIVE_STATUSES.has(build.status) ? 30_000 : false,
    enabled: Boolean(buildId),
  });

  if (buildError) {
    return (
      <div style={{ padding: "1rem 1.5rem" }}>
        <InlineNotification
          kind="error"
          title="Failed to load build"
          subtitle={String(buildError)}
        />
      </div>
    );
  }

  return (
    <div>
      {/* Page header */}
      <div style={{ padding: "2rem 1.5rem 1.5rem" }}>
        <PageHeader
          crumbs={[
            { label: "Granite.build", to: "/" },
            { label: "Builds", to: "/dashboard/builds" },
            { label: build?.name ?? "…" },
          ]}
        />
        <div className={styles.buildHeaderRow}>
          {loadingBuild ? (
            <SkeletonText width="300px" />
          ) : (
            <>
              <h4>{build?.name}</h4>
              {build && (showRefreshing
                ? <InlineLoading description="Refreshing build progress" status="active" style={{ width: 'auto' }} />
                : <BuildStatusBadge status={build.status} />
              )}
              {build?.tags && build.tags.length > 0 && (
                <span style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                  {build.tags.map((t, i) => (
                    <Tag key={t} type={(['blue', 'purple', 'teal', 'magenta'] as const)[i % 4]} size="sm">{t}</Tag>
                  ))}
                </span>
              )}
            </>
          )}
        </div>
      </div>

      {/* Build details tile */}
      <BuildDetails
        build={build}
        status={status}
        describe={describe}
        events={events}
        loadingBuild={loadingBuild}
        loadingStatus={loadingStatus}
        statusError={statusError}
        buildId={buildId!}
      />
    </div>
  );
}
