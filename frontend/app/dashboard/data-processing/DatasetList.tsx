"use client";

import { useState } from "react";
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
  Pagination,
} from "@carbon/react";
import type { DPDataset } from "@/api/dataProcessing";
import { adaptStatus } from "@/api/gbserver";
import { BuildStatusBadge } from "@/components/BuildStatusBadge";

interface Props {
  datasets: DPDataset[];
  search: string;
  isLoading: boolean;
}

const HEADERS = [
  { key: "name", header: "Dataset" },
  { key: "latest_status", header: "Status" },
  { key: "build_count", header: "Builds" },
  { key: "megatron_path", header: "COS Path" },
  { key: "age", header: "Updated" },
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

export function DatasetList({ datasets, search, isLoading }: Props) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  if (isLoading)
    return (
      <DataTableSkeleton
        headers={HEADERS}
        rowCount={10}
        showHeader={false}
        showToolbar={false}
      />
    );

  const searchLower = search.toLowerCase();
  const filtered = searchLower
    ? datasets.filter(
        (d) =>
          d.name.toLowerCase().includes(searchLower) ||
          d.megatron_path?.toLowerCase().includes(searchLower) ||
          d.arrow_path?.toLowerCase().includes(searchLower),
      )
    : datasets;

  const start = (page - 1) * pageSize;
  const paged = filtered.slice(start, start + pageSize);

  const rows = paged.map((ds) => ({
    id: ds.name,
    name: ds.name,
    latest_status: ds.latest_build_status ?? "",
    build_count: ds.build_count,
    megatron_path: ds.megatron_path || ds.arrow_path || "—",
    age: formatAge(ds.latest_build_time),
  }));

  return (
    <DataTable rows={rows} headers={HEADERS}>
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
                  <TableRow key={row.id} {...rowProps}>
                    {row.cells.map((cell) => (
                      <TableCell key={cell.id}>
                        {cell.info.header === "latest_status" ? (
                          cell.value ? (
                            <BuildStatusBadge
                              status={adaptStatus(cell.value as string)}
                            />
                          ) : (
                            "—"
                          )
                        ) : cell.info.header === "megatron_path" ? (
                          <span
                            title={cell.value as string}
                            style={{
                              fontFamily: "monospace",
                              fontSize: "0.75rem",
                            }}
                          >
                            {(cell.value as string).length > 50
                              ? "…" + (cell.value as string).slice(-47)
                              : (cell.value as string)}
                          </span>
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
            pageSizes={[10, 20, 50]}
            onChange={({ page: p, pageSize: ps }) => {
              setPage(p);
              setPageSize(ps);
            }}
          />
        </TableContainer>
      )}
    </DataTable>
  );
}
