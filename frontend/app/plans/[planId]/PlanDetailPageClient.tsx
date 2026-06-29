"use client";

import { useState, use } from "react";
import { useRouter } from "next/navigation";
import ReactMarkdown from "react-markdown";
import { useQuery } from "@tanstack/react-query";
import {
  CopyButton,
  DataTable,
  DataTableSkeleton,
  InlineNotification,
  Pagination,
  SkeletonText,
  Tab,
  TabListVertical,
  TabPanel,
  TabPanels,
  TabsVertical,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableHeader,
  TableRow,
  Tag,
  Tile,
  Layer,
} from "@carbon/react";
import { Copy } from "@carbon/icons-react";
import { getPlan } from "@/api/plans";
import { getBuild, getBuildStatus } from "@/api/gbserver";
import { PageHeader } from "@/components/PageHeader";
import { BuildStatusBadge } from "@/components/BuildStatusBadge";
import { PlanStatusBadge } from "@/components/PlanStatusBadge";
import { TagsCell } from "@/components/TagsCell";
import styles from "./page.module.scss";
import type { LinkedBuild, Build, BuildTargetRun } from "@/types";

const BUILD_HEADERS = [
  { key: "name", header: "Name" },
  { key: "build_id", header: "Build ID" },
  { key: "targets", header: "Targets" },
  { key: "username", header: "Username" },
  { key: "tags", header: "Tags" },
  { key: "space_name", header: "Space" },
  { key: "status", header: "Status" },
  { key: "duration", header: "Duration" },
  { key: "updated_time", header: "Updated" },
];

function formatDuration(created: string, updated: string): string {
  const ms = new Date(updated).getTime() - new Date(created).getTime();
  if (ms < 0 || !created || !updated) return "—";
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function formatDate(iso: string): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function copyText(e: React.MouseEvent, text: string) {
  e.stopPropagation();
  navigator.clipboard.writeText(text).catch(() => undefined);
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

interface EnrichedLinkedBuild extends LinkedBuild {
  build: Build | null;
  targets: BuildTargetRun[];
}

export default function FlightPlanDetail({
  params,
}: {
  params: Promise<{ planId: string }>;
}) {
  const { planId } = use(params);
  const router = useRouter();
  const [buildPage, setBuildPage] = useState(1);
  const [buildPageSize, setBuildPageSize] = useState(10);

  const {
    data: planData,
    isLoading: planLoading,
    error: planError,
  } = useQuery({
    queryKey: ["plan", planId],
    queryFn: () => getPlan(planId!),
    enabled: !!planId,
    refetchInterval: (query) => {
      const status = query.state.data?.plan?.status;
      return status === "executing" ? 30_000 : false;
    },
  });

  const linkedBuilds: LinkedBuild[] = planData?.builds ?? [];

  const { data: enrichedBuilds, isLoading: buildsLoading } = useQuery({
    queryKey: ["plan-builds", planId, linkedBuilds.map((b) => b.build_id)],
    queryFn: async (): Promise<EnrichedLinkedBuild[]> => {
      return Promise.all(
        linkedBuilds.map(async (lb): Promise<EnrichedLinkedBuild> => {
          const [build, status] = await Promise.all([
            getBuild(lb.build_id).catch(() => null),
            getBuildStatus(lb.build_id).catch(() => null),
          ]);
          const targetNames = status ? Object.keys(status.targets) : [];
          const targets: BuildTargetRun[] = targetNames.map(
            (n) => status!.targets[n],
          );
          return { ...lb, build, targets };
        }),
      );
    },
    enabled: linkedBuilds.length > 0,
    staleTime: 30_000,
  });

  const plan = planData?.plan;
  const allEnriched = enrichedBuilds ?? [];

  const buildRows = allEnriched.map((lb) => ({
    id: lb.build_id,
    name: lb.build?.name ?? "—",
    build_id: lb.build_id,
    targets: lb.targets,
    username: lb.build?.username ?? "—",
    tags: lb.build?.tags ?? [],
    space_name: lb.build?.space_name ?? "—",
    status: lb.build?.status ?? null,
    duration: lb.build
      ? formatDuration(lb.build.created_time, lb.build.updated_time)
      : "—",
    updated_time: lb.build?.updated_time ?? "",
  }));

  const pagedBuildRows = buildRows.slice(
    (buildPage - 1) * buildPageSize,
    buildPage * buildPageSize,
  );

  if (planError) {
    return (
      <div style={{ padding: "1rem 1.5rem" }}>
        <PageHeader
          crumbs={[
            { label: "Granite.build", to: "/" },
            { label: "Flight Plans", to: "/plans" },
            { label: planId ?? "" },
          ]}
        />
        <InlineNotification
          kind="error"
          title="Failed to load plan"
          subtitle={String(planError)}
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
            { label: "Flight Plans", to: "/plans" },
            { label: plan?.name ?? "…" },
          ]}
        />
        <div className={styles.headerRow}>
          {planLoading ? (
            <SkeletonText width="300px" />
          ) : (
            plan && (
              <>
                <h4 style={{ margin: 0 }}>{plan.name}</h4>
                <PlanStatusBadge status={plan.status} />
                {(plan.tags?.length ?? 0) > 0 && (
                  <span
                    style={{
                      display: "flex",
                      flexWrap: "wrap",
                      gap: "0.25rem",
                    }}
                  >
                    {plan.tags!.map((t, i) => (
                      <Tag
                        key={t}
                        type={
                          (["blue", "purple", "teal", "magenta"] as const)[
                            i % 4
                          ]
                        }
                        size="sm"
                      >
                        {t}
                      </Tag>
                    ))}
                  </span>
                )}
              </>
            )
          )}
        </div>
      </div>

      {/* Vertical tabs */}
      <div
        className={styles.tabsWrapper}
        style={{
          height: "calc(100vh - 220px)",
          minHeight: "500px",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <div style={{ flex: 1, overflow: "hidden", minHeight: 0 }}>
          <TabsVertical height="100%">
            <TabListVertical aria-label="Flight plan detail tabs">
              <Tab>Details</Tab>
              <Tab>Markdown</Tab>
            </TabListVertical>
            <TabPanels>
              {/* Details */}
              <TabPanel style={{ overflowY: "auto", height: "100%" }}>
                <Tile>
                  {planLoading ? (
                    <SkeletonText paragraph lineCount={6} />
                  ) : (
                    plan && (
                      <dl className={styles.detailsList}>
                        <DetailField label="Plan ID">
                          <span
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "0.25rem",
                            }}
                          >
                            <code
                              className={styles.wordBreakAll}
                              style={{ fontSize: "0.75rem" }}
                            >
                              {plan.plan_id}
                            </code>
                            <CopyButton
                              feedback="Copied!"
                              iconDescription="Copy Plan ID"
                              onClick={() =>
                                navigator.clipboard.writeText(plan.plan_id)
                              }
                              size="sm"
                            />
                          </span>
                        </DetailField>
                        {plan.created_by && (
                          <DetailField label="Created By">
                            {plan.created_by}
                          </DetailField>
                        )}
                        {plan.space_name && (
                          <DetailField label="Space">
                            {plan.space_name}
                          </DetailField>
                        )}
                        <DetailField label="Revision">
                          {plan.revision}
                        </DetailField>
                        {plan.created_at && (
                          <DetailField label="Created">
                            {new Date(plan.created_at).toLocaleString()}
                          </DetailField>
                        )}
                        <DetailField label="Updated">
                          {new Date(plan.updated_at).toLocaleString()}
                        </DetailField>
                        {plan.summary && (
                          <DetailField label="Summary">
                            <ReactMarkdown components={{
                              p: ({ children }) => <p style={{ margin: 0, fontSize: '0.875rem' }}>{children}</p>,
                            }}>
                              {plan.summary}
                            </ReactMarkdown>
                          </DetailField>
                        )}
                        <DetailField label="Linked builds">
                          {buildsLoading ||
                          (linkedBuilds.length > 0 &&
                            allEnriched.length === 0) ? (
                            <DataTableSkeleton
                              headers={BUILD_HEADERS}
                              rowCount={linkedBuilds.length || 3}
                              showHeader={false}
                              showToolbar={false}
                            />
                          ) : allEnriched.length === 0 ? (
                            <p
                              style={{
                                fontSize: "0.875rem",
                                color: "var(--cds-text-secondary)",
                                margin: 0,
                              }}
                            >
                              No linked builds.
                            </p>
                          ) : (
                            <Layer>
                              <DataTable
                                rows={pagedBuildRows}
                                headers={BUILD_HEADERS}
                                isSortable
                              >
                                {({
                                  rows: tableRows,
                                  headers,
                                  getTableProps,
                                  getHeaderProps,
                                  getRowProps,
                                }) => (
                                  <TableContainer>
                                    <Table {...getTableProps()} size="md">
                                      <TableHead>
                                        <TableRow>
                                          {headers.map((h) => {
                                            const { key: _k, ...hProps } =
                                              getHeaderProps({ header: h });
                                            return (
                                              <TableHeader
                                                key={h.key}
                                                {...hProps}
                                              >
                                                {h.header}
                                              </TableHeader>
                                            );
                                          })}
                                        </TableRow>
                                      </TableHead>
                                      <TableBody>
                                        {tableRows.map((row) => {
                                          const { key: _k, ...rowProps } =
                                            getRowProps({
                                              row,
                                            });
                                          return (
                                            <TableRow
                                              key={row.id}
                                              {...rowProps}
                                              onClick={() => router.push(`/builds/${row.id}`)}
                                              style={{ cursor: "pointer" }}
                                            >
                                              {row.cells.map((cell) => (
                                                <TableCell key={cell.id}>
                                                  {cell.info.header ===
                                                  "build_id" ? (
                                                    <span
                                                      style={{
                                                        display: "flex",
                                                        alignItems: "center",
                                                        gap: "0.25rem",
                                                      }}
                                                    >
                                                      <code
                                                        style={{
                                                          fontSize: "0.75rem",
                                                        }}
                                                      >
                                                        {(
                                                          cell.value as string
                                                        ).slice(0, 8)}
                                                        …
                                                      </code>
                                                      <Copy
                                                        size={14}
                                                        style={{
                                                          cursor: "pointer",
                                                          opacity: 0.6,
                                                          flexShrink: 0,
                                                        }}
                                                        onClick={(e) =>
                                                          copyText(
                                                            e,
                                                            cell.value as string,
                                                          )
                                                        }
                                                        title="Copy Build ID"
                                                      />
                                                    </span>
                                                  ) : cell.info.header ===
                                                    "targets" ? (
                                                    <span
                                                      style={{
                                                        display: "flex",
                                                        flexWrap: "wrap",
                                                        gap: "0.25rem",
                                                      }}
                                                    >
                                                      {(
                                                        cell.value as BuildTargetRun[]
                                                      ).length > 0
                                                        ? (
                                                            cell.value as BuildTargetRun[]
                                                          ).map((t) => (
                                                            <Tag
                                                              key={
                                                                t.target_name
                                                              }
                                                              type="blue"
                                                              size="sm"
                                                            >
                                                              {t.target_name}
                                                            </Tag>
                                                          ))
                                                        : "—"}
                                                    </span>
                                                  ) : cell.info.header ===
                                                    "tags" ? (
                                                    <TagsCell
                                                      tags={
                                                        (cell.value as string[]) ??
                                                        []
                                                      }
                                                    />
                                                  ) : cell.info.header ===
                                                    "status" ? (
                                                    cell.value ? (
                                                      <BuildStatusBadge
                                                        status={
                                                          cell.value as import("../../../types").BuildStatus
                                                        }
                                                      />
                                                    ) : (
                                                      "—"
                                                    )
                                                  ) : cell.info.header ===
                                                    "updated_time" ? (
                                                    formatDate(
                                                      cell.value as string,
                                                    )
                                                  ) : (
                                                    (cell.value as React.ReactNode)
                                                  )}
                                                </TableCell>
                                              ))}
                                            </TableRow>
                                          );
                                        })}
                                      </TableBody>
                                    </Table>
                                    {allEnriched.length > buildPageSize && (
                                      <Pagination
                                        totalItems={allEnriched.length}
                                        pageSize={buildPageSize}
                                        page={buildPage}
                                        pageSizes={[10, 25, 50]}
                                        onChange={({
                                          page: p,
                                          pageSize: ps,
                                        }) => {
                                          setBuildPage(p);
                                          setBuildPageSize(ps);
                                        }}
                                      />
                                    )}
                                  </TableContainer>
                                )}
                              </DataTable>
                            </Layer>
                          )}
                        </DetailField>
                      </dl>
                    )
                  )}
                </Tile>
              </TabPanel>

              {/* Markdown */}
              <TabPanel
                style={{ padding: 0, height: "100%", overflow: "hidden" }}
              >
                {planLoading ? (
                  <SkeletonText paragraph lineCount={10} />
                ) : plan?.markdown_body ? (
                  <div
                    style={{
                      height: "100%",
                      display: "flex",
                      flexDirection: "column",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "flex-end",
                        padding: "0.25rem 0.5rem",
                        flexShrink: 0,
                      }}
                    >
                      <CopyButton
                        autoAlign
                        feedback="Copied!"
                        iconDescription="Copy markdown"
                        onClick={() =>
                          navigator.clipboard.writeText(plan.markdown_body!)
                        }
                        size="sm"
                      />
                    </div>
                    <div
                      style={{
                        margin: 0,
                        padding: "0.5rem 2rem 1rem",
                        flex: 1,
                        overflow: "auto",
                        fontSize: "0.875rem",
                        lineHeight: 1.6,
                      }}
                    >
                      <ReactMarkdown components={{
                        p: ({ children }) => <p style={{ marginTop: 0, marginBottom: '0.75rem' }}>{children}</p>,
                        h1: ({ children }) => <h1 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>{children}</h1>,
                        h2: ({ children }) => <h2 style={{ fontSize: '1.125rem', fontWeight: 600, marginBottom: '0.5rem' }}>{children}</h2>,
                        h3: ({ children }) => <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.5rem' }}>{children}</h3>,
                        code: ({ children }) => <code style={{ fontFamily: '"IBM Plex Mono", monospace', fontSize: '0.8125em', background: 'var(--cds-layer-accent)', padding: '0 0.25rem', borderRadius: '2px' }}>{children}</code>,
                        pre: ({ children }) => <pre style={{ background: 'var(--cds-layer)', border: '1px solid var(--cds-border-subtle-01)', padding: '0.75rem', borderRadius: '4px', overflow: 'auto', marginBottom: '0.75rem' }}>{children}</pre>,
                        ul: ({ children }) => <ul style={{ paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ul>,
                        ol: ({ children }) => <ol style={{ paddingLeft: '1.5rem', marginBottom: '0.75rem' }}>{children}</ol>,
                      }}>
                        {plan.markdown_body}
                      </ReactMarkdown>
                    </div>
                  </div>
                ) : (
                  <p
                    style={{
                      fontSize: "0.875rem",
                      color: "var(--cds-text-secondary)",
                      margin: "1rem",
                    }}
                  >
                    No markdown content.
                  </p>
                )}
              </TabPanel>
            </TabPanels>
          </TabsVertical>
        </div>
      </div>
    </div>
  );
}
