"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  Dropdown,
  MultiSelect,
  InlineNotification,
} from "@carbon/react";
import { useQuery } from "@tanstack/react-query";
import {
  listBuilds,
  searchBuilds,
  getBuildTags,
  listSpaces,
  type BuildSearchResult,
} from "@granite-build/ui-core/api/gbserver";
import { BuildsTable } from "@granite-build/ui-core/components/BuildsTable";
import { PageHeader } from "@granite-build/ui-core/components/PageHeader";
import { SEARCH_WINDOW } from "@granite-build/ui-core/lib/buildSearch";
import type { BuildStatus } from "@granite-build/ui-core/types";
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

  const term = search.trim();

  // Builds — server-side search and pagination (see searchBuilds for the
  // older-gbserver fallback).
  const { data, isLoading, error } = useQuery<BuildSearchResult & { term: string }>({
    queryKey: ["builds", spaceName, selectedTags, status, term, page, pageSize],
    queryFn: () => {
      const params = {
        space_name: spaceName,
        tags: selectedTags.length ? selectedTags : undefined,
        status: status === "all" ? undefined : (status as BuildStatus),
        sort: "created_time:desc",
        page_index: page - 1,
        page_size: pageSize,
      };
      // SEARCH_WINDOW only applies against a gbserver that predates name_contains.
      const result = term
        ? searchBuilds({ ...params, name_contains: term }, SEARCH_WINDOW)
        : listBuilds(params).then((r) => ({ ...r, truncated: false }));
      return result.then((r) => ({ ...r, term }));
    },
    // Keep the previous result while the next loads: without it BuildsTable
    // shows its skeleton, which unmounts the search box mid-typing.
    placeholderData: (prev) => prev,
  });

  const handlePageChange = useCallback((p: number, ps: number) => {
    setPage(p);
    setPageSize(ps);
  }, []);

  // Each keystroke would otherwise be a server round-trip.
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => () => clearTimeout(searchDebounceRef.current), []);
  const handleSearch = useCallback((value: string) => {
    clearTimeout(searchDebounceRef.current);
    searchDebounceRef.current = setTimeout(() => {
      setSearch(value);
      setPage(1);
    }, 300);
  }, []);

  // Only for the term on screen, so a placeholder from the previous search
  // can't flash the notice while the next one loads.
  const searchTruncated = (data?.term === term && data.truncated) || false;

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

      {searchTruncated && (
        <InlineNotification
          kind="info"
          lowContrast
          hideCloseButton
          title="Search limited to recent builds"
          subtitle={`Searching the ${SEARCH_WINDOW} most recent builds matching the current filters. Narrow the filters to search further back.`}
          style={{ marginBottom: "1rem" }}
        />
      )}

      <BuildsTable
        builds={data?.items ?? []}
        total={data?.total ?? 0}
        page={page}
        pageSize={pageSize}
        isLoading={isLoading}
        onPageChange={handlePageChange}
        onSearch={handleSearch}
      />
    </div>
  );
}
