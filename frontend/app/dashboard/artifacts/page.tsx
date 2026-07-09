"use client";

import React, { useState, useCallback } from "react";
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
  Dropdown,
  MultiSelect,
  InlineNotification,
  Tag,
} from "@carbon/react";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { listArtifacts, listSpaces, getArtifactTags } from "@/api/gbserver";
import { TagsCell } from "@/components/TagsCell";
import { PageHeader } from "@/components/PageHeader";
import styles from "./page.module.scss";

function TagDropdownItem({ label }: { id: string; label: string }) {
  return (
    <span
      title={label}
      style={{
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
        display: "block",
      }}
    >
      {label}
    </span>
  );
}

const ARTIFACT_TYPES = [
  { id: "all", label: "All types" },
  { id: "MODEL", label: "Model" },
  { id: "DATASET", label: "Dataset" },
  { id: "FILESET", label: "Fileset" },
  { id: "TABLE", label: "Table" },
];

// Mirrors gbserver's ArtifactRegistrationStatus enum
// (src/gbserver/storage/artifact_registration.py).
const STATUS_OPTIONS = [
  { id: "all", label: "All statuses" },
  { id: "success", label: "Success" },
  { id: "pending", label: "Pending" },
  { id: "failed", label: "Failed" },
  { id: "cancelled", label: "Cancelled" },
];

const TYPE_COLORS: Record<string, "blue" | "green" | "teal" | "purple"> = {
  MODEL: "purple",
  DATASET: "teal",
  FILESET: "blue",
  TABLE: "green",
};

const HEADERS = [
  { key: "name", header: "Name" },
  { key: "artifact_type", header: "Type" },
  { key: "space_name", header: "Space" },
  { key: "username", header: "Owner" },
  { key: "tags", header: "Tags" },
  { key: "updated_time", header: "Updated" },
];

function formatDate(iso: string): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function ArtifactsPage() {
  const router = useRouter();

  const [spaceName, setSpaceName] = useState<string | undefined>();
  const [artifactType, setArtifactType] = useState<string>("all");
  const [artifactStatus, setArtifactStatus] = useState<string>("all");
  const [selectedTags, setTags] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [sortKey, setSortKey] = useState("updated_time");
  const [sortDir, setSortDir] = useState<"ASC" | "DESC">("DESC");

  function handleSort(key: string) {
    if (key === sortKey) {
      setSortDir((d) => (d === "DESC" ? "ASC" : "DESC"));
    } else {
      setSortKey(key);
      setSortDir("ASC");
    }
    setPage(1);
  }

  const { data: spaces = [] } = useQuery({
    queryKey: ["spaces"],
    queryFn: listSpaces,
  });

  const { data: tags = [] } = useQuery({
    queryKey: ["artifact-tags", spaceName],
    queryFn: () => getArtifactTags(spaceName),
  });

  const { data, isLoading, isFetching, error } = useQuery({
    queryKey: [
      "artifacts",
      spaceName,
      artifactType,
      selectedTags,
    ],
    queryFn: () =>
      listArtifacts({
        space_name: spaceName,
        tags: selectedTags.length ? selectedTags : undefined,
      }),
    placeholderData: (prev) => prev,
  });

  const handlePageChange = useCallback((p: number, ps: number) => {
    setPage(p);
    setPageSize(ps);
  }, []);

  const handleSearch = useCallback((term: string) => {
    setSearch(term);
    setPage(1);
  }, []);

  const spaceItems = [
    { id: "__all__", label: "All spaces" },
    ...spaces.map((s) => ({ id: s.name, label: s.name })),
  ];

  const allRows = (data?.items ?? [])
    .filter((a) => artifactType === "all" || a.artifact_type === artifactType)
    .filter((a) => artifactStatus === "all" || a.status === artifactStatus)
    .filter((a) => a.name.toLowerCase().includes(search.toLowerCase()))
    .map((a) => ({
      id: a.uuid,
      name: a.name,
      artifact_type: a.artifact_type,
      space_name: a.space_name,
      username: a.username,
      tags: a.tags,
      updated_time: a.updated_time,
    }))
    .sort((a, b) => {
      const mult = sortDir === "DESC" ? -1 : 1;
      if (sortKey === "updated_time") {
        return (
          mult *
          (new Date(a.updated_time).getTime() -
            new Date(b.updated_time).getTime())
        );
      }
      const av =
        sortKey === "tags"
          ? (a.tags ?? []).join(",")
          : String((a as Record<string, unknown>)[sortKey] ?? "");
      const bv =
        sortKey === "tags"
          ? (b.tags ?? []).join(",")
          : String((b as Record<string, unknown>)[sortKey] ?? "");
      return mult * av.localeCompare(bv);
    });
  const rows = allRows.slice((page - 1) * pageSize, page * pageSize);

  if (isLoading && !data)
    return (
      <div style={{ padding: "1.5rem" }}>
        <PageHeader
          crumbs={[{ label: "Granite.build", to: "/" }, { label: "Artifacts" }]}
        />
        <DataTableSkeleton
          headers={HEADERS}
          rowCount={20}
          showHeader={false}
          showToolbar={false}
        />
      </div>
    );

  return (
    <div style={{ padding: "1.5rem" }}>
      <PageHeader
        crumbs={[{ label: "Granite.build", to: "/" }, { label: "Artifacts" }]}
      />
      <h4 style={{ marginBottom: "2rem" }}>Artifacts</h4>

      <div
        className={styles.toolbar}
        style={isFetching ? { pointerEvents: "none", opacity: 0.5 } : undefined}
      >
        <div
          style={{
            display: "flex",
            gap: "1rem",
            flexWrap: "wrap",
            marginBottom: "1rem",
            alignItems: "flex-end",
          }}
        >
          <div style={{ minWidth: "12rem" }}>
            <Dropdown
              id="space-filter"
              titleText="Space"
              label="Space"
              size="sm"
              items={spaceItems}
              itemToString={(i) => i?.label ?? ""}
              selectedItem={
                spaceItems.find((i) => i.id === (spaceName ?? "__all__")) ??
                spaceItems[0]
              }
              onChange={({ selectedItem }) => {
                setSpaceName(
                  selectedItem?.id === "__all__" ? undefined : selectedItem?.id,
                );
                setPage(1);
              }}
            />
          </div>
          <div style={{ minWidth: "12rem" }}>
            <Dropdown
              id="type-filter"
              titleText="Type"
              label="Type"
              size="sm"
              items={ARTIFACT_TYPES}
              itemToString={(i) => i?.label ?? ""}
              selectedItem={
                ARTIFACT_TYPES.find((i) => i.id === artifactType) ??
                ARTIFACT_TYPES[0]
              }
              onChange={({ selectedItem }) => {
                setArtifactType(selectedItem?.id ?? "all");
                setPage(1);
              }}
            />
          </div>
          <div style={{ minWidth: "12rem" }}>
            <Dropdown
              id="status-filter"
              titleText="Status"
              label="Status"
              size="sm"
              items={STATUS_OPTIONS}
              itemToString={(i) => i?.label ?? ""}
              selectedItem={
                STATUS_OPTIONS.find((i) => i.id === artifactStatus) ??
                STATUS_OPTIONS[0]
              }
              onChange={({ selectedItem }) => {
                setArtifactStatus(selectedItem?.id ?? "all");
                setPage(1);
              }}
            />
          </div>
          {tags.length > 0 && (
            <div style={{ minWidth: "12rem" }}>
              <MultiSelect
                id="tag-filter"
                titleText="Tags"
                label="Tags"
                size="sm"
                items={tags.map((t) => ({ id: t, label: t }))}
                itemToString={(i) => i?.label ?? ""}
                itemToElement={TagDropdownItem}
                onChange={({ selectedItems }) => {
                  setTags((selectedItems ?? []).map((i) => i.id));
                  setPage(1);
                }}
              />
            </div>
          )}
        </div>
      </div>

      {error && (
        <InlineNotification
          kind="error"
          title="Failed to load artifacts"
          subtitle={String(error)}
          style={{ marginBottom: "1rem" }}
        />
      )}

      {isFetching ? (
        <DataTableSkeleton
          headers={HEADERS}
          rowCount={pageSize}
          showHeader={false}
          showToolbar={false}
        />
      ) : (
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
                    placeholder="Search artifacts…"
                    onChange={(_e, value) => handleSearch(value ?? "")}
                  />
                </TableToolbarContent>
              </TableToolbar>
              <Table {...getTableProps()} size="md">
                <TableHead>
                  <TableRow>
                    {headers.map((h) => {
                      const {
                        key: _k,
                        onClick: _o,
                        isSortHeader: _ish,
                        sortDirection: _sd,
                        ...hProps
                      } = getHeaderProps({ header: h });
                      return (
                        <TableHeader
                          key={h.key}
                          {...hProps}
                          isSortHeader={sortKey === h.key}
                          sortDirection={sortKey === h.key ? sortDir : "NONE"}
                          onClick={() => handleSort(h.key)}
                        >
                          {h.header}
                        </TableHeader>
                      );
                    })}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {tableRows.map((row) => {
                    const rowProps = getRowProps({ row });
                    return (
                      <TableRow
                        {...rowProps}
                        key={row.id}
                        onClick={() => router.push(`/dashboard/artifacts/_/?id=${row.id}`)}
                        style={{ cursor: "pointer" }}
                      >
                        {row.cells.map((cell) => (
                          <TableCell key={cell.id}>
                            {cell.info.header === "artifact_type" ? (
                              ARTIFACT_TYPES.find((t) => t.id === cell.value)?.label ??
                                (cell.value as string)
                            ) : cell.info.header === "tags" ? (
                              <TagsCell tags={(cell.value as string[]) ?? []} />
                            ) : cell.info.header === "updated_time" ? (
                              formatDate(cell.value as string)
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
                totalItems={allRows.length}
                pageSize={pageSize}
                page={page}
                pageSizes={[10, 20, 50, 100]}
                onChange={({ page: p, pageSize: ps }) =>
                  handlePageChange(p, ps)
                }
              />
            </TableContainer>
          )}
        </DataTable>
      )}
    </div>
  );
}
