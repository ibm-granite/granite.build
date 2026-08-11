"use client";

import { LineChart } from "@carbon/charts-react";
import { ScaleTypes } from "@carbon/charts";
import { InlineLoading } from "@carbon/react";
import type { FailureTrendResponse } from "../types";
import { useChartsTheme } from "../hooks/useTheme";

interface Props {
  data: FailureTrendResponse | null | undefined;
  daysBack?: number;
  isAnalyzing?: boolean;
}

export function FailureTrendChart({ data, daysBack, isAnalyzing }: Props) {
  const theme = useChartsTheme();

  if (
    !data ||
    !Array.isArray(data.categories) ||
    !Array.isArray(data.labels) ||
    !data.series ||
    typeof data.series !== "object"
  ) {
    return (
      <p style={{ color: "#525252", padding: "1rem" }}>
        No failure trend data available.
      </p>
    );
  }

  const { labels, categories, series } = data;

  const chartData = categories.flatMap((cat) =>
    labels.map((label, i) => ({
      group: cat,
      date: label,
      value: series[cat]?.[i] ?? 0,
    })),
  );

  const options = {
    title: "",
    axes: {
      bottom: {
        title: "Date",
        mapsTo: "date",
        scaleType: ScaleTypes.TIME,
        ...(daysBack
          ? {
              domain: [
                new Date(Date.now() - daysBack * 24 * 60 * 60 * 1000),
                new Date(),
              ],
            }
          : {}),
      },
      left: {
        title: "Failures",
        mapsTo: "value",
        scaleType: ScaleTypes.LINEAR,
      },
    },
    curve: "curveMonotoneX",
    timeScale: { addSpaceOnEdges: 0 },
    toolbar: {
      enabled: true,
      numberOfIcons: 2,
      controls: [
        { type: "Reset zoom" },
        { type: "Zoom in" },
        { type: "Zoom out" },
      ],
    },
    zoomBar: {
      top: { enabled: true },
    },
    height: "450px",
    theme,
  };

  return (
    <div>
      <LineChart data={chartData} options={options} />
      {isAnalyzing ? (
        <InlineLoading
          description="AI analysis in progress — categories will update shortly"
          status="active"
          style={{ marginTop: "0.5rem", fontSize: "0.75rem" }}
        />
      ) : data.total_analyzed > 0 ? (
        <p
          style={{
            fontSize: "0.75rem",
            color: "var(--cds-text-secondary)",
            marginTop: "0.5rem",
          }}
        >
          Analyzed {data.total_analyzed} builds and found{" "}
          {data.categories.length} categories
          {typeof data.analysis_time_ms === "number"
            ? ` in ${(data.analysis_time_ms / 1000).toFixed(1)}s`
            : ""}
        </p>
      ) : null}
    </div>
  );
}
