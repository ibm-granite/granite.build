"use client";

import { useState, useEffect, useRef, useMemo } from "react";
import {
  Toggle,
  Dropdown,
  InlineNotification,
  SkeletonText,
  Button,
  TextInput,
  RadioButtonGroup,
  RadioButton,
  Accordion,
  AccordionItem,
  Modal,
  Layer,
  Table,
  TableHead,
  TableHeader,
  TableBody,
  TableRow,
  TableCell,
  Pagination,
  Tag,
} from "@carbon/react";
import {
  Download,
  Save,
  TrashCan,
  Launch,
  Renew,
  Categories,
} from "@carbon/icons-react";
import Link from "next/link";
import { PageHeader } from "@granite-build/ui-core/components/PageHeader";
import { BaseTile } from "@granite-build/ui-core/components/BaseTile";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getBuildStatusChart,
  getFailureTrends,
  getAIDaemonStatus,
  runAnalysis,
  getTrendHistory,
  saveTrendAnalysis,
  getSavedTrend,
  deleteSavedTrend,
  toggleTrendVisibility,
} from "@granite-build/ui-core/api/analytics";
import { BuildStatusChart } from "@granite-build/ui-core/components/BuildStatusChart";
import { FailureTrendChart } from "@granite-build/ui-core/components/FailureTrendChart";
import type { FailureTrendResponse, TrendHistoryItem } from "@granite-build/ui-core/types";

const DAYS_OPTIONS = [
  { id: "7", label: "Last 7 days" },
  { id: "14", label: "Last 14 days" },
  { id: "30", label: "Last 30 days" },
  { id: "60", label: "Last 60 days" },
  { id: "90", label: "Last 90 days" },
];

function exportCSV(data: FailureTrendResponse) {
  const rows: string[][] = [
    [
      "Build ID",
      "Name",
      "Username",
      "Space",
      "Date",
      "Category",
      "Confidence",
      "Summary",
    ],
  ];
  for (const builds of Object.values(data.builds_by_category)) {
    for (const b of builds) {
      rows.push([
        b.build_id,
        b.name,
        b.username,
        b.space_name,
        b.created_at,
        b.category,
        String(b.confidence.toFixed(2)),
        b.summary ?? "",
      ]);
    }
  }
  // Prefix a leading =/+/-/@ with an apostrophe to neutralize CSV formula
  // injection when opened in a spreadsheet app — quoting alone doesn't stop
  // it. Leading whitespace/tab/CR before the trigger character is still a
  // trigger for some spreadsheet apps, so check past it too.
  const escapeCsvCell = (c: string) =>
    /^\s*[=+\-@]/.test(c) ? `'${c}` : c;
  const csv = rows
    .map((r) =>
      r.map((c) => `"${escapeCsvCell(c).replace(/"/g, '""')}"`).join(","),
    )
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "failure-trends.csv";
  a.click();
  URL.revokeObjectURL(url);
}

export default function AnalyticsPage() {
  const [daysBack, setDaysBack] = useState(30);
  const [showTestRuns, setShowTestRuns] = useState(true);
  const [trendDays, setTrendDays] = useState(90);
  const [lastRunSource, setLastRunSource] = useState<
    "llm_phase1" | "llm_custom"
  >("llm_phase1");
  const [customModalOpen, setCustomModalOpen] = useState(false);
  const [customInput, setCustomInput] = useState("");
  const [isCurrentAnalysisSaved, setIsCurrentAnalysisSaved] = useState(false);
  const [saveTitle, setSaveTitle] = useState("");
  const [savePublic, setSavePublic] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveModalOpen, setSaveModalOpen] = useState(false);
  const username = 'standalone'
  const queryClient = useQueryClient();

  const {
    data: statusData,
    isLoading: statusLoading,
    error: statusError,
  } = useQuery({
    queryKey: ["build-status-chart", daysBack],
    queryFn: () => getBuildStatusChart(daysBack, false),
  });

  const [trendRefreshing, setTrendRefreshing] = useState(false);
  const [categoryPages, setCategoryPages] = useState<Record<string, number>>(
    {},
  );
  const CATEGORY_PAGE_SIZE = 10;

  const {
    data: trendData,
    isLoading: trendLoading,
    isFetching: trendFetching,
    error: trendError,
    refetch: refetchTrends,
  } = useQuery({
    queryKey: ["failure-trends", trendDays, !showTestRuns, lastRunSource],
    queryFn: () =>
      getFailureTrends({
        days_back: trendDays,
        exclude_tests: !showTestRuns,
        source: lastRunSource,
      }),
  });
  useEffect(() => {
    if (!trendFetching) setTrendRefreshing(false);
  }, [trendFetching]);

  const { data: daemonStatus, isSuccess: daemonQueried } = useQuery({
    queryKey: ["ai-daemon-status"],
    queryFn: getAIDaemonStatus,
    refetchInterval: 10_000,
  });
  const llmConfigured =
    !daemonQueried || (daemonStatus?.llm_configured ?? true);

  const prevAnalyzing = useRef(false);
  useEffect(() => {
    if (prevAnalyzing.current && !daemonStatus?.analyzing) {
      void refetchTrends();
    }
    prevAnalyzing.current = daemonStatus?.analyzing ?? false;
  }, [daemonStatus?.analyzing]);

  useEffect(() => {
    setIsCurrentAnalysisSaved(false);
  }, [trendDays, showTestRuns, lastRunSource]);

  const { data: mineData, refetch: refetchMine } = useQuery({
    queryKey: ["trend-history", "mine", username],
    queryFn: () => getTrendHistory("mine", username),
  });
  const { data: publicData, refetch: refetchPublic } = useQuery({
    queryKey: ["trend-history", "public", username],
    queryFn: () => getTrendHistory("public", username),
  });
  const allSavedItems = useMemo(() => {
    const seen = new Set<string>();
    return [...(mineData?.items ?? []), ...(publicData?.items ?? [])]
      .filter((item) => {
        const ok = !seen.has(item.update_id);
        seen.add(item.update_id);
        return ok;
      })
      .sort((a, b) => b.created_at.localeCompare(a.created_at));
  }, [mineData, publicData]);
  function refetchHistory() {
    void refetchMine();
    void refetchPublic();
  }

  const sortedCategories = trendData
    ? Object.entries(trendData.builds_by_category).sort(
        ([, a], [, b]) => b.length - a.length,
      )
    : [];

  async function handleSave() {
    if (!trendData) return;
    setSaving(true);
    setSaveError(null);
    const res = await saveTrendAnalysis(
      trendData,
      saveTitle || undefined,
      savePublic,
      username,
    );
    setSaving(false);
    if (!res?.success) {
      setSaveError("Failed to save analysis");
    } else {
      setIsCurrentAnalysisSaved(true);
      setSaveTitle("");
      setSavePublic(false);
      refetchHistory();
    }
  }

  async function handleLoad(item: TrendHistoryItem) {
    const res = await getSavedTrend(item.update_id);
    if (res?.data) {
      queryClient.setQueryData(
        ["failure-trends", trendDays, !showTestRuns, lastRunSource],
        res.data,
      );
      setIsCurrentAnalysisSaved(true);
    }
  }

  async function handleDelete(item: TrendHistoryItem) {
    await deleteSavedTrend(item.update_id, username);
    refetchHistory();
  }

  async function handleTogglePublic(item: TrendHistoryItem) {
    await toggleTrendVisibility(item.update_id, !item.is_public, username);
    refetchHistory();
  }

  const statusAction = (
    <Dropdown
      id="status-days"
      titleText=""
      label=""
      size="sm"
      items={DAYS_OPTIONS}
      itemToString={(i) => i?.label ?? ""}
      selectedItem={
        DAYS_OPTIONS.find((i) => i.id === String(daysBack)) ?? DAYS_OPTIONS[2]
      }
      onChange={({ selectedItem }) =>
        setDaysBack(Number(selectedItem?.id ?? 30))
      }
    />
  );

  async function handleRunAuto() {
    setIsCurrentAnalysisSaved(false);
    await runAnalysis({ mode: "auto", days_back: trendDays });
    setLastRunSource("llm_phase1");
  }

  async function handleRunCustom() {
    const categories = customInput
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);
    if (!categories.length) return;
    setCustomModalOpen(false);
    setIsCurrentAnalysisSaved(false);
    await runAnalysis({ mode: "custom", categories, days_back: trendDays });
    setLastRunSource("llm_custom");
  }

  return (
    <div style={{ padding: "1.5rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1rem",
          flexWrap: "wrap",
          gap: "1rem",
        }}
      >
        <div>
          <PageHeader
            crumbs={[
              { label: "Granite.build", to: "/" },
              { label: "Analytics" },
            ]}
          />
          <h4 style={{ marginBottom: "2rem" }}>Analytics</h4>
        </div>
        <Toggle
          id="show-tests"
          labelText="Show test runs"
          toggled={showTestRuns}
          onToggle={setShowTestRuns}
          size="sm"
        />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <BaseTile title="Build status over time" action={statusAction}>
          {statusError && (
            <InlineNotification
              kind="error"
              title="Failed to load chart"
              subtitle={String(statusError)}
            />
          )}
          {statusLoading ? (
            <SkeletonText paragraph lineCount={8} />
          ) : (
            <BuildStatusChart
              data={statusData ?? []}
              showTestRuns={showTestRuns}
            />
          )}
        </BaseTile>
      </div>

      <BaseTile
        title="Failure trend analysis"
        action={
          <div
            style={{ display: "flex", alignItems: "center", gap: "0.25rem" }}
          >
            <Dropdown
              id="trend-days"
              titleText=""
              label=""
              size="sm"
              items={DAYS_OPTIONS}
              itemToString={(i) => i?.label ?? ""}
              selectedItem={
                DAYS_OPTIONS.find((i) => i.id === String(trendDays)) ??
                DAYS_OPTIONS[4]
              }
              onChange={({ selectedItem }) =>
                setTrendDays(Number(selectedItem?.id ?? 90))
              }
              style={{ width: "160px" }}
            />
            {!isCurrentAnalysisSaved && trendData && (
              <Button
                size="sm"
                hasIconOnly
                iconDescription="Save analysis"
                kind="ghost"
                renderIcon={Save}
                tooltipPosition="bottom"
                onClick={() => setSaveModalOpen(true)}
              />
            )}
            {trendData && (
              <Button
                size="sm"
                hasIconOnly
                iconDescription="Export"
                kind="ghost"
                renderIcon={Download}
                tooltipPosition="bottom"
                onClick={() => exportCSV(trendData)}
              />
            )}
          </div>
        }
        onRefresh={() => {
          setTrendRefreshing(true);
          void refetchTrends();
        }}
        isRefreshing={trendRefreshing}
      >
        <p
          style={{
            fontSize: "0.875rem",
            color: "var(--cds-text-secondary)",
            marginBottom: "0.75rem",
          }}
        >
          {lastRunSource === "llm_custom"
            ? "Showing custom category analysis. Run auto analysis to switch back to AI-assigned categories."
            : "Classifies each failed build using AI-assigned error categories, then plots failure counts over time."}
        </p>
        {llmConfigured ? (
          <div
            style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}
          >
            <Button
              size="sm"
              kind="primary"
              renderIcon={Renew}
              disabled={daemonStatus?.analyzing}
              onClick={() => void handleRunAuto()}
            >
              Run auto analysis
            </Button>
            <Button
              size="sm"
              kind="secondary"
              renderIcon={Categories}
              disabled={daemonStatus?.analyzing}
              onClick={() => setCustomModalOpen(true)}
            >
              Run custom analysis
            </Button>
          </div>
        ) : (
          <p
            style={{
              fontSize: "0.875rem",
              color: "var(--cds-text-secondary)",
              marginBottom: "1.25rem",
            }}
          >
            AI-powered categorization is not available. Set{" "}
            <code>GB_UI_LLM_BASE_URL</code> and <code>GB_UI_LLM_API_KEY</code>{" "}
            to enable failure classification.
          </p>
        )}

        <Modal
          open={customModalOpen}
          modalHeading="Run custom category analysis"
          primaryButtonText="Run analysis"
          secondaryButtonText="Cancel"
          primaryButtonDisabled={!customInput.trim()}
          onRequestClose={() => setCustomModalOpen(false)}
          onSecondarySubmit={() => setCustomModalOpen(false)}
          onRequestSubmit={() => void handleRunCustom()}
        >
          <p
            style={{
              marginBottom: "1rem",
              fontSize: "0.875rem",
              color: "#525252",
            }}
          >
            Enter comma-separated categories. Each failed build will be
            classified into one of them.
          </p>
          <TextInput
            id="custom-categories"
            labelText="Categories (comma-delimited)"
            placeholder="e.g. Infrastructure, OOM, Code Error, Network, Timeout"
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
          />
        </Modal>

        {trendError && (
          <InlineNotification
            kind="error"
            title="Failed to load trends"
            subtitle={String(trendError)}
          />
        )}

        {trendLoading ? (
          <SkeletonText paragraph lineCount={8} />
        ) : (
          <>
            {llmConfigured && (
              <FailureTrendChart
                data={trendData}
                daysBack={trendDays}
                isAnalyzing={daemonStatus?.analyzing}
              />
            )}
            {trendData ? (
              <div style={{ marginTop: "1.25rem" }}>
                {sortedCategories.length === 0 ? (
                  <p style={{ color: "#525252", fontSize: "0.875rem" }}>
                    No failure data for this period.
                  </p>
                ) : (
                  <>
                    <h6 style={{ marginBottom: "1rem" }}>Categories</h6>
                    <style>{`.gb-cat-accordion .cds--accordion__content { padding-inline: 0; }`}</style>
                    <div className="gb-cat-accordion">
                      <Accordion>
                        {sortedCategories.map(([cat, builds], i) => {
                          const pct =
                            trendData.total_analyzed > 0
                              ? Math.round(
                                  (builds.length / trendData.total_analyzed) *
                                    100,
                                )
                              : 0;
                          return (
                            <AccordionItem
                              key={cat}
                              title={
                                <span
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "1rem",
                                  }}
                                >
                                  <span style={{ flex: 1 }}>{cat}</span>
                                  <span
                                    style={{
                                      fontSize: "0.875rem",
                                      color:
                                        "var(--cds-text-secondary, #525252)",
                                    }}
                                  >
                                    {builds.length} ({pct}%)
                                  </span>
                                </span>
                              }
                            >
                              {(() => {
                                const page = categoryPages[cat] ?? 1;
                                const start = (page - 1) * CATEGORY_PAGE_SIZE;
                                const pageBuilds = builds.slice(
                                  start,
                                  start + CATEGORY_PAGE_SIZE,
                                );
                                return (
                                  <Layer>
                                    <Table size="sm">
                                      <TableHead>
                                        <TableRow>
                                          <TableHeader>Build</TableHeader>
                                          <TableHeader>User</TableHeader>
                                          <TableHeader>Date</TableHeader>
                                          <TableHeader>Summary</TableHeader>
                                        </TableRow>
                                      </TableHead>
                                      <TableBody>
                                        {pageBuilds.map((b) => (
                                          <TableRow key={b.build_id}>
                                            <TableCell>
                                              <Link
                                                href={`/dashboard/builds/_/?id=${b.build_id}`}
                                                style={{
                                                  color:
                                                    "var(--cds-link-primary, #0f62fe)",
                                                }}
                                              >
                                                {b.name}
                                              </Link>
                                            </TableCell>
                                            <TableCell
                                              style={{
                                                color: "#525252",
                                                fontSize: "0.875rem",
                                              }}
                                            >
                                              {b.username}
                                            </TableCell>
                                            <TableCell
                                              style={{
                                                color: "#525252",
                                                fontSize: "0.875rem",
                                              }}
                                            >
                                              {b.created_at.slice(0, 10)}
                                            </TableCell>
                                            <TableCell
                                              style={{ fontSize: "0.875rem" }}
                                            >
                                              {b.summary ?? ""}
                                            </TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                    {builds.length > CATEGORY_PAGE_SIZE && (
                                      <Pagination
                                        page={page}
                                        pageSize={CATEGORY_PAGE_SIZE}
                                        pageSizes={[10, 25, 50]}
                                        totalItems={builds.length}
                                        onChange={({ page: p }) =>
                                          setCategoryPages((prev) => ({
                                            ...prev,
                                            [cat]: p,
                                          }))
                                        }
                                      />
                                    )}
                                  </Layer>
                                );
                              })()}
                            </AccordionItem>
                          );
                        })}
                      </Accordion>
                    </div>
                  </>
                )}
              </div>
            ) : (
              <p style={{ color: "#525252", fontSize: "0.875rem" }}>
                No failure trend data available.
              </p>
            )}
          </>
        )}

        <Modal
          open={saveModalOpen}
          onRequestClose={() => setSaveModalOpen(false)}
          modalHeading="Analysis"
          passiveModal
          size="lg"
        >
          <h5 style={{ marginBottom: "0.75rem" }}>Save this analysis</h5>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "1rem",
              flexWrap: "wrap",
            }}
          >
            <div
              style={{ display: "flex", alignItems: "flex-start", gap: "2rem" }}
            >
              <RadioButtonGroup
                legendText="Type"
                name="save-visibility"
                valueSelected={savePublic ? "public" : "private"}
                onChange={(value) => setSavePublic(value === "public")}
                // orientation={"vertical"}
              >
                <RadioButton
                  labelText="Public"
                  value="public"
                  id="save-public"
                />
                <RadioButton
                  labelText="Private"
                  value="private"
                  id="save-private"
                />
              </RadioButtonGroup>
              <div style={{ width: "240px" }}>
                <TextInput
                  id="save-title"
                  labelText="Optional title"
                  placeholder="e.g. Weekly infra review"
                  value={saveTitle}
                  onChange={(e) => setSaveTitle(e.target.value)}
                  size="sm"
                />
              </div>
            </div>
            <Button
              size="sm"
              renderIcon={Save}
              onClick={handleSave}
              disabled={!trendData || saving}
            >
              {saving ? "Saving…" : "Save Analysis"}
            </Button>
          </div>
          {saveError && (
            <p
              style={{
                color: "var(--cds-support-error, #da1e28)",
                fontSize: "0.875rem",
                marginTop: "0.5rem",
              }}
            >
              {saveError}
            </p>
          )}

          <hr
            style={{
              border: "none",
              borderTop: "1px solid var(--cds-border-subtle-00)",
              margin: "1.5rem 0 0",
            }}
          />
          <h5 style={{ marginTop: "1.5rem", marginBottom: "0.75rem" }}>
            Saved analyses
          </h5>
          {allSavedItems.length === 0 ? (
            <p style={{ color: "#525252", fontSize: "0.875rem" }}>
              No saved analyses found.
            </p>
          ) : (
            <Table size="sm">
              <TableHead>
                <TableRow>
                  <TableHeader>Title</TableHeader>
                  <TableHeader>Author</TableHeader>
                  <TableHeader>Date</TableHeader>
                  <TableHeader>Tags</TableHeader>
                  <TableHeader />
                </TableRow>
              </TableHead>
              <TableBody>
                {allSavedItems.map((item: TrendHistoryItem) => (
                  <TableRow key={item.update_id}>
                    <TableCell>
                      <div style={{ fontWeight: 500 }}>
                        {item.title || item.summary}
                      </div>
                      {item.title && (
                        <div style={{ fontSize: "0.875rem" }}>
                          {item.summary}
                        </div>
                      )}
                    </TableCell>
                    <TableCell style={{ fontSize: "0.875rem" }}>
                      {item.author}
                    </TableCell>
                    <TableCell style={{ fontSize: "0.875rem" }}>
                      {item.created_at.slice(0, 10)}
                    </TableCell>
                    <TableCell>
                      <div
                        style={{
                          display: "flex",
                          gap: "0.25rem",
                          alignItems: "center",
                        }}
                      >
                        {item.author === username && (
                          <Tag type="purple" size="sm">
                            Mine
                          </Tag>
                        )}
                        {item.is_public && (
                          <Tag type="blue" size="sm">
                            Public
                          </Tag>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <div style={{ display: "flex", gap: "0.25rem" }}>
                        <Button
                          size="sm"
                          kind="ghost"
                          renderIcon={Launch}
                          iconDescription="Load"
                          hasIconOnly
                          onClick={() => {
                            handleLoad(item);
                            setSaveModalOpen(false);
                          }}
                          tooltipPosition="left"
                        />
                        {item.author === username && (
                          <>
                            <Button
                              size="sm"
                              kind="ghost"
                              onClick={() => handleTogglePublic(item)}
                            >
                              {item.is_public ? "Make private" : "Make public"}
                            </Button>
                            <Button
                              size="sm"
                              kind="danger--ghost"
                              renderIcon={TrashCan}
                              iconDescription="Delete"
                              hasIconOnly
                              onClick={() => handleDelete(item)}
                              tooltipPosition="left"
                            />
                          </>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Modal>
      </BaseTile>
    </div>
  );
}
