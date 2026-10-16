"use client";

import { useState, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ContentSwitcher,
  Switch,
  Checkbox,
  Dropdown,
  Search,
  Button,
  InlineNotification,
  IconSwitch,
  Layer,
  CheckboxGroup,
} from "@carbon/react";
import {
  getDPLineage,
  getDPNodeCounts,
  getDPPipelineStatus,
  type DPNode,
} from "@granite-build/ui-core/api/dataProcessing";
import { PageHeader } from "@granite-build/ui-core/components/PageHeader";
import { BaseTile } from "@granite-build/ui-core/components/BaseTile";
import { PipelineDAG } from "./PipelineDAG";
import { DatasetList } from "./DatasetList";
import { LoadReportForm } from "./LoadReportForm";
import styles from "./page.module.scss";
import { List, ModelBuilder } from "@carbon/icons-react";

const TIME_OPTIONS = [
  { id: "7", label: "7 days" },
  { id: "1", label: "24 hours" },
  { id: "30", label: "30 days" },
];

function useRefreshState(isFetching: boolean): [boolean, () => void] {
  const [isRefreshing, setIsRefreshing] = useState(false);
  useEffect(() => {
    if (!isFetching) setIsRefreshing(false);
  }, [isFetching]);
  return [isRefreshing, () => setIsRefreshing(true)];
}

export default function DataProcessingPage() {
  const [days, setDays] = useState(7);
  const [completedOnly, setCompletedOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const [view, setView] = useState<"graph" | "list">("graph");
  const queryClient = useQueryClient();

  const { data, isLoading, isFetching, error } = useQuery({
    queryKey: ["dp-lineage", days],
    queryFn: () => getDPLineage(days),
    staleTime: 60_000,
  });

  const [isRefreshing, triggerRefresh] = useRefreshState(isFetching);

  function handleRefresh() {
    triggerRefresh();
    queryClient.invalidateQueries({ queryKey: ["dp-lineage", days] });
    queryClient.invalidateQueries({ queryKey: ["dp-node-counts"] });
    queryClient.invalidateQueries({ queryKey: ["dp-pipeline-status"] });
  }

  const nodes = data?.nodes ?? [];
  const edges = data?.edges ?? [];
  const datasets = data?.datasets ?? [];

  const nodePathArgs = nodes.map((n: DPNode) => ({ id: n.id, path: n.path }));
  const { data: nodeCounts } = useQuery({
    queryKey: ["dp-node-counts", nodes.map((n: DPNode) => n.id)],
    queryFn: () => getDPNodeCounts(nodePathArgs),
    enabled: nodePathArgs.length > 0,
    staleTime: 120_000,
  });

  const megatronPathArgs = nodes
    .filter((n: DPNode) => n.type === "megatron")
    .map((n: DPNode) => ({ id: n.id, path: n.path }));
  const { data: pipelineStatuses } = useQuery({
    queryKey: ["dp-pipeline-status", megatronPathArgs.map((p) => p.id)],
    queryFn: () => getDPPipelineStatus(megatronPathArgs),
    enabled: megatronPathArgs.length > 0,
    staleTime: 120_000,
  });

  return (
    <div style={{ padding: "1.5rem" }}>
      <PageHeader
        crumbs={[
          { label: "Granite.build", to: "/" },
          { label: "Data Processing" },
        ]}
      />
      <h4 style={{ marginBottom: "1.5rem" }}>Data Processing Pipeline</h4>

      <BaseTile
        title="Recent Datasets"
        onRefresh={handleRefresh}
        isRefreshing={isRefreshing}
      >
        {/* Sub-toolbar: search + filters left, ContentSwitcher right */}
        <div className={styles.subToolbar}>
          {view === "graph" && (
            <CheckboxGroup legendText={""} orientation="horizontal">
              <Checkbox
                id="dp-completed-only"
                labelText="Completed only"
                checked={completedOnly}
                onChange={(_, { checked }) => setCompletedOnly(checked)}
              />
            </CheckboxGroup>
          )}
          <Dropdown
            id="dp-time-range"
            titleText=""
            label="Time range"
            size="sm"
            style={{ minWidth: "9rem" }}
            items={TIME_OPTIONS}
            itemToString={(i) => i?.label ?? ""}
            selectedItem={
              TIME_OPTIONS.find((i) => i.id === String(days)) ?? TIME_OPTIONS[0]
            }
            onChange={({ selectedItem }) => {
              setDays(Number(selectedItem?.id ?? 1));
              setFocusedNodeId(null);
            }}
          />
          <div style={{ flexShrink: 0 }}>
            <ContentSwitcher
              size="sm"
              selectedIndex={view === "graph" ? 0 : 1}
              onChange={({ index }) => setView(index === 0 ? "graph" : "list")}
            >
              <IconSwitch name="graph" text="Graph view" align="left">
                <ModelBuilder />
              </IconSwitch>
              <IconSwitch name="list" text="List view" align="left">
                <List />
              </IconSwitch>
            </ContentSwitcher>
          </div>
        </div>
        <div className={styles.searchWrapper}>
          <Layer>
            <Search
              id="dp-search"
              labelText=""
              placeholder="Dataset or build ID…"
              size="md"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onClear={() => setSearch("")}
            />
          </Layer>
        </div>

        {error && (
          <InlineNotification
            kind="error"
            title="Failed to load pipeline data"
            subtitle={String(error)}
            style={{ marginBottom: "1rem" }}
          />
        )}
        {data?.warning && (
          <InlineNotification
            kind="warning"
            title="Dataset scan incomplete"
            subtitle={data.warning}
            style={{ marginBottom: "1rem" }}
            lowContrast
          />
        )}

        <div className={styles.statsRow}>
          {data && data.matched > 0 && (
            <span>
              {nodes.length} paths, {edges.length} connections
              {data?.scanned != null
                ? ` — ${data.scanned} builds scanned, ${data.matched} matched`
                : ""}
            </span>
          )}
          {focusedNodeId && (
            <Button
              kind="ghost"
              size="sm"
              onClick={() => setFocusedNodeId(null)}
            >
              Clear focus
            </Button>
          )}
        </div>

        {view === "graph" ? (
          <PipelineDAG
            nodes={nodes}
            edges={edges}
            datasets={datasets}
            nodeCounts={nodeCounts}
            pipelineStatuses={pipelineStatuses}
            completedOnly={completedOnly}
            search={search}
            focusedNodeId={focusedNodeId}
            onFocusNode={setFocusedNodeId}
            isLoading={isLoading || isRefreshing}
            scanned={data?.scanned}
          />
        ) : (
          <DatasetList
            datasets={datasets}
            search={search}
            isLoading={isLoading}
          />
        )}

        <LoadReportForm />
      </BaseTile>
    </div>
  );
}
