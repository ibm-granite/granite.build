"use client";

import { useState } from "react";
import {
  DataTable,
  DataTableSkeleton,
  InlineNotification,
  Pagination,
  SkeletonText,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableHeader,
  TableRow,
  TableToolbar,
  TableToolbarContent,
  TableToolbarSearch,
} from "@carbon/react";
import { useQuery, useQueries } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { Copy } from "@carbon/icons-react";
import { listPlans, getPlan } from "@/api/plans";
import { PageHeader } from "@/components/PageHeader";
import { PlanStatusBadge } from "@/components/PlanStatusBadge";
import { TagsCell } from "@/components/TagsCell";
import type { Plan } from "@/types";

const HEADERS = [
  { key: "name", header: "Name" },
  { key: "plan_id", header: "Plan ID" },
  { key: "created_by", header: "Created By" },
  { key: "tags", header: "Tags" },
  { key: "status", header: "Status" },
  { key: "updated_at", header: "Updated" },
];

function formatAge(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60_000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function copyId(e: React.MouseEvent, id: string) {
  e.stopPropagation();
  navigator.clipboard.writeText(id).catch(() => undefined);
}

export default function FlightPlansPage() {
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  const { data, isLoading, error } = useQuery({
    queryKey: ["plans"],
    queryFn: listPlans,
    staleTime: 30_000,
  });

  const allPlans: Plan[] = data?.plans ?? [];
  const filtered = search
    ? allPlans.filter(
        (p) =>
          p.name.toLowerCase().includes(search.toLowerCase()) ||
          p.plan_id.toLowerCase().includes(search.toLowerCase()) ||
          (p.created_by ?? "").toLowerCase().includes(search.toLowerCase()),
      )
    : allPlans;

  const rows = filtered.map((p) => ({
    id: p.plan_id,
    name: p.name,
    plan_id: p.plan_id,
    created_by: p.username ?? p.created_by ?? "—",
    tags: p.tags ?? [],
    status: p.status,
    updated_at: p.updated_at,
  }));

  const pagedRows = rows.slice((page - 1) * pageSize, page * pageSize);

  const enrichQueries = useQueries({
    queries: pagedRows.map((row) => ({
      queryKey: ["plan", row.id],
      queryFn: () => getPlan(row.id),
      staleTime: 60_000,
    })),
  });

  const enrichMap = Object.fromEntries(
    pagedRows.map((row, i) => [
      row.id,
      {
        isLoading: enrichQueries[i]?.isLoading ?? false,
        createdBy: enrichQueries[i]?.data?.plan.created_by,
        tags: enrichQueries[i]?.data?.plan.tags,
      },
    ]),
  );

  if (isLoading) {
    return (
      <div style={{ padding: "1.5rem" }}>
        <PageHeader
          crumbs={[
            { label: "Granite.build", to: "/" },
            { label: "Flight Plans" },
          ]}
        />
        <DataTableSkeleton
          headers={HEADERS}
          rowCount={10}
          showHeader={false}
          showToolbar={false}
        />
      </div>
    );
  }

  return (
    <div style={{ padding: "1.5rem", marginBottom: "2rem" }}>
      <PageHeader
        crumbs={[
          { label: "Granite.build", to: "/" },
          { label: "Flight Plans" },
        ]}
      />
      <h4 style={{ marginBottom: "2rem" }}>Flight Plans</h4>

      {error && (
        <InlineNotification
          kind="error"
          title="Failed to load flight plans"
          subtitle={String(error)}
          style={{ marginBottom: "1rem" }}
        />
      )}

      <DataTable rows={pagedRows} headers={HEADERS} isSortable>
        {({
          rows: tableRows,
          headers,
          getTableProps,
          getHeaderProps,
          getRowProps,
        }) => (
          <TableContainer>
            <TableToolbar>
              <TableToolbarContent>
                <TableToolbarSearch
                  placeholder="Search plans…"
                  onChange={(_e, value) => {
                    setSearch(value ?? "");
                    setPage(1);
                  }}
                />
              </TableToolbarContent>
            </TableToolbar>
            <Table {...getTableProps()} size="md">
              <TableHead>
                <TableRow>
                  {headers.map((h) => {
                    const { key: _k, ...hProps } = getHeaderProps({
                      header: h,
                    });
                    return (
                      <TableHeader
                        key={h.key}
                        {...hProps}
                        style={
                          h.key === "plan_id"
                            ? { minWidth: "22rem" }
                            : undefined
                        }
                      >
                        {h.header}
                      </TableHeader>
                    );
                  })}
                </TableRow>
              </TableHead>
              <TableBody>
                {tableRows.map((row) => {
                  const { key: _k, ...rowProps } = getRowProps({ row });
                  return (
                    <TableRow
                      key={row.id}
                      {...rowProps}
                      onClick={() => router.push(`/plans/${row.id}`)}
                      style={{ cursor: "pointer" }}
                    >
                      {row.cells.map((cell) => (
                        <TableCell key={cell.id}>
                          {cell.info.header === "plan_id" ? (
                            <span
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "0.25rem",
                              }}
                            >
                              <code
                                style={{
                                  fontSize: "0.875rem",
                                  color: "var(--cds-link-primary)",
                                }}
                              >
                                {cell.value as string}
                              </code>
                              <Copy
                                size={14}
                                style={{
                                  cursor: "pointer",
                                  flexShrink: 0,
                                  opacity: 0.6,
                                }}
                                onClick={(e) => copyId(e, cell.value as string)}
                                title="Copy Plan ID"
                              />
                            </span>
                          ) : cell.info.header === "tags" ? (
                            enrichMap[row.id]?.isLoading ? (
                              <span
                                style={{
                                  display: "inline-block",
                                  width: "4rem",
                                }}
                              >
                                <SkeletonText />
                              </span>
                            ) : (
                              <TagsCell tags={enrichMap[row.id]?.tags ?? []} />
                            )
                          ) : cell.info.header === "status" ? (
                            <PlanStatusBadge status={cell.value as string} />
                          ) : cell.info.header === "created_by" ? (
                            enrichMap[row.id]?.isLoading ? (
                              <span
                                style={{
                                  display: "inline-block",
                                  width: "6rem",
                                }}
                              >
                                <SkeletonText />
                              </span>
                            ) : (
                              (enrichMap[row.id]?.createdBy ?? "—")
                            )
                          ) : cell.info.header === "updated_at" ? (
                            formatAge(cell.value as string)
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
            <Pagination
              totalItems={filtered.length}
              pageSize={pageSize}
              page={page}
              pageSizes={[10, 20, 30, 40, 50]}
              onChange={({ page: p, pageSize: ps }) => {
                setPage(p);
                setPageSize(ps);
              }}
            />
          </TableContainer>
        )}
      </DataTable>
    </div>
  );
}
