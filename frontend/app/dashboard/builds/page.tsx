"use client";

import { useState, useCallback } from "react";
import {
  Dropdown,
  MultiSelect,
  InlineNotification,
} from "@carbon/react";
import { useQuery } from "@tanstack/react-query";
import { listBuilds, getBuildTags, listSpaces } from "@/api/gbserver";
import { BuildsTable } from "@/components/BuildsTable";
import { PageHeader } from "@/components/PageHeader";
import type { BuildStatus } from "@/types";
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

// Mirrors gbserver's Status enum (src/gbserver/types/status.py).
const STATUS_OPTIONS = [
  { id: "all", label: "All statuses" },
  { id: "running", label: "Running" },
  { id: "success", label: "Success" },
  { id: "failed", label: "Failed" },
  { id: "invalid", label: "Invalid" },
  { id: "pending", label: "Pending" },
  { id: "submitted", label: "Submitted" },
  { id: "retry_pending", label: "Retrying" },
  { id: "cancel_requested", label: "Cancelling" },
  { id: "cancelled", label: "Cancelled" },
];

export default function BuildsPage() {
  // Filters
  const [spaceName, setSpaceName] = useState<string | undefined>();
  const [selectedTags, setTags] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  // Spaces
  const { data: spaces = [] } = useQuery({
    queryKey: ["spaces"],
    queryFn: listSpaces,
  });

  // Tags
  const { data: tags = [], isLoading: tagsLoading } = useQuery({
    queryKey: ["build-tags", spaceName],
    queryFn: () => getBuildTags(spaceName),
  });

  // Builds
  const { data, isLoading, error } = useQuery({
    queryKey: [
      "builds",
      spaceName,
      selectedTags,
      status,
      page,
      pageSize,
    ],
    queryFn: () =>
      listBuilds({
        space_name: spaceName,
        tags: selectedTags.length ? selectedTags : undefined,
        status: status === "all" ? undefined : (status as BuildStatus),
        sort: "created_time:desc",
        page_index: page - 1,
        page_size: pageSize,
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

  // gbserver's list endpoint has no name/search param, so search filters the
  // current page of already-fetched builds rather than querying the server.
  const allItems = data?.items ?? [];
  const visibleItems = search
    ? allItems.filter((b) =>
        b.name.toLowerCase().includes(search.toLowerCase()),
      )
    : allItems;
  const visibleTotal = search ? visibleItems.length : data?.total ?? 0;

  const spaceItems = [
    { id: "__all__", label: "All spaces" },
    ...spaces.map((s) => ({ id: s.name, label: s.name })),
  ];

  return (
    <div style={{ padding: "1.5rem", marginBottom: "2rem" }}>
      <PageHeader
        crumbs={[{ label: "Granite.build", to: "/" }, { label: "Builds" }]}
      />
      <h4 style={{ marginBottom: "2rem" }}>Builds</h4>

      {/* Toolbar */}
      <div className={styles.toolbar}>
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
              id="status-filter"
              titleText="Status"
              label="Status"
              size="sm"
              items={STATUS_OPTIONS}
              itemToString={(i) => i?.label ?? ""}
              selectedItem={
                STATUS_OPTIONS.find((i) => i.id === status) ?? STATUS_OPTIONS[0]
              }
              onChange={({ selectedItem }) => {
                setStatus(selectedItem?.id ?? "all");
                setPage(1);
              }}
            />
          </div>

          <div style={{ minWidth: "12rem" }}>
            <MultiSelect
              id="tag-filter"
              titleText="Tags"
              label={tagsLoading ? "Loading..." : "Tags"}
              size="sm"
              disabled={tagsLoading}
              items={tags.map((t) => ({ id: t, label: t }))}
              itemToString={(i) => i?.label ?? ""}
              itemToElement={TagDropdownItem}
              onChange={({ selectedItems }) => {
                setTags((selectedItems ?? []).map((i) => i.id));
                setPage(1);
              }}
            />
          </div>
        </div>
      </div>
      {error && (
        <InlineNotification
          kind="error"
          title="Failed to load builds"
          subtitle={String(error)}
          style={{ marginBottom: "1rem" }}
        />
      )}

      <BuildsTable
        builds={visibleItems}
        total={visibleTotal}
        page={page}
        pageSize={pageSize}
        isLoading={isLoading}
        onPageChange={handlePageChange}
        onSearch={handleSearch}
      />
    </div>
  );
}
