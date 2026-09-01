"use client";

import {
  DataTable,
  DataTableSkeleton,
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
  Pagination,
  Tag,
} from "@carbon/react";

import React from "react";
import { useRouter } from "next/navigation";
import type { Build, BuildStatus } from "../types";
import { BuildStatusBadge } from "./BuildStatusBadge";
import { TagsCell } from "./TagsCell";

interface Props {
  builds: Build[];
  total: number;
  page: number;
  pageSize: number;
  isLoading: boolean;
  onPageChange: (page: number, pageSize: number) => void;
  onSearch: (term: string) => void;
}

const HEADERS = [
  { key: "name", header: "Build" },
  { key: "status", header: "Status" },
  { key: "space_name", header: "Space" },
  { key: "username", header: "Owner" },
  { key: "tags", header: "Tags" },
  { key: "updated_time", header: "Updated" },
  { key: "cpu", header: "CPU" },
  { key: "memory", header: "Memory" },
  { key: "gpu", header: "GPU" },
];

function formatAge(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export function BuildsTable({
  builds,
  total,
  page,
  pageSize,
  isLoading,
  onPageChange,
  onSearch,
}: Props) {
  const router = useRouter();

  if (isLoading) {
    return (
      <DataTableSkeleton
        headers={HEADERS}
        rowCount={10}
        showHeader={false}
        showToolbar={false}
      />
    );
  }

  const rows = builds.map((b) => ({
    id: b.uuid,
    name: b.name,
    status: b.status,
    space_name: b.space_name,
    username: b.username,
    tags: b.tags,
    updated_time: b.updated_time,
    cpu: b.resources?.cpu,
    memory: b.resources?.memory,
    gpu: b.resources?.gpu,
    _build: b,
  }));

  return (
    <DataTable rows={rows} headers={HEADERS} isSortable>
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
                placeholder="Search builds…"
                onChange={(_e, value) => onSearch(value ?? "")}
              />
            </TableToolbarContent>
          </TableToolbar>
          <Table {...getTableProps()} size="md">
            <TableHead>
              <TableRow>
                {headers.map((h) => {
                  const { key: _k, ...hProps } = getHeaderProps({ header: h });
                  return (
                    <TableHeader key={h.key} {...hProps}>
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
                    onClick={() => router.push(`/dashboard/builds/_/?id=${row.id}`)}
                    style={{ cursor: "pointer" }}
                  >
                    {row.cells.map((cell) => (
                      <TableCell key={cell.id}>
                        {cell.info.header === "status" ? (
                          <BuildStatusBadge
                            status={cell.value as BuildStatus}
                          />
                        ) : cell.info.header === "tags" ? (
                          <TagsCell tags={(cell.value as string[]) ?? []} />
                        ) : cell.info.header === "updated_time" ? (
                          formatAge(cell.value as string)
                        ) : cell.info.header === "gpu" ? (
                          cell.value != null && (cell.value as number) > 0 ? (
                            <Tag type="purple" size="sm">
                              ×{cell.value as number}
                            </Tag>
                          ) : (
                            "—"
                          )
                        ) : cell.info.header === "cpu" ||
                          cell.info.header === "memory" ? (
                          ((cell.value as string | undefined) ?? "—")
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
            totalItems={total}
            pageSize={pageSize}
            page={page}
            pageSizes={[10, 20, 50, 100]}
            onChange={({ page: p, pageSize: ps }) => onPageChange(p, ps)}
          />
        </TableContainer>
      )}
    </DataTable>
  );
}
