"use client";

import { useState } from "react";

import {
  CodeSnippet,
  InlineNotification,
  Modal,
  SkeletonText,
  Tag,
  Button,
  TextArea,
  StructuredListWrapper,
  StructuredListBody,
  StructuredListRow,
  StructuredListCell,
  Layer,
  AILabel,
  AILabelContent,
} from "@carbon/react";
import { ThumbsUp, ThumbsDown } from "@carbon/icons-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getAIAnalysis,
  submitAIFeedback,
} from "../api/analytics";
import type { AIAnalysis } from "../types";

interface Props {
  buildId: string;
  failureReason?: string;
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color = pct >= 70 ? "green" : pct >= 40 ? "warm-gray" : "red";
  return (
    <Tag type={color} size="sm">
      Confidence: {pct}%
    </Tag>
  );
}

function FeedbackModal({
  analysis,
  buildId,
  open,
  onClose,
}: {
  analysis: AIAnalysis;
  buildId: string;
  open: boolean;
  onClose: () => void;
}) {
  const qc = useQueryClient();
  const [helpful, setHelpful] = useState<boolean | undefined>(
    analysis.feedback_helpful,
  );
  const [correction, setCorrection] = useState(
    analysis.corrected_root_cause ?? "",
  );
  const [comment, setComment] = useState(analysis.feedback_comment ?? "");

  const { mutate, isPending: isLoading } = useMutation({
    mutationFn: () =>
      submitAIFeedback(buildId, analysis.update_id, {
        helpful,
        corrected_root_cause: correction || undefined,
        comment: comment || undefined,
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["ai-analysis", buildId] });
      onClose();
    },
  });

  return (
    <Modal
      open={open}
      modalHeading="Provide feedback"
      primaryButtonText={isLoading ? "Submitting…" : "Submit feedback"}
      primaryButtonDisabled={isLoading}
      secondaryButtonText="Cancel"
      onRequestClose={onClose}
      onSecondarySubmit={onClose}
      onRequestSubmit={() => mutate()}
      size="sm"
    >
      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", alignItems: "center" }}>
        <Button
          kind={helpful === true ? "primary" : "ghost"}
          size="sm"
          renderIcon={ThumbsUp}
          iconDescription="Helpful"
          hasIconOnly
          onClick={() => setHelpful(true)}
        />
        <Button
          kind={helpful === false ? "danger" : "ghost"}
          size="sm"
          renderIcon={ThumbsDown}
          iconDescription="Not helpful"
          hasIconOnly
          onClick={() => setHelpful(false)}
        />
        <span style={{ fontSize: "0.875rem", color: "var(--cds-text-secondary)", marginLeft: "0.5rem" }}>
          Was this analysis helpful?
        </span>
      </div>
      <TextArea
        id={`correction-${analysis.update_id}`}
        labelText="Correct root cause (optional)"
        value={correction}
        onChange={(e) => setCorrection(e.target.value)}
        rows={2}
        style={{ marginBottom: "1rem" }}
      />
      <TextArea
        id={`comment-${analysis.update_id}`}
        labelText="Additional comments (optional)"
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        rows={2}
      />
    </Modal>
  );
}

export function AIAnalysisPanel({ buildId, failureReason }: Props) {
  const [showFeedback, setShowFeedback] = useState(false);
  const {
    data: analyses,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["ai-analysis", buildId],
    queryFn: () => getAIAnalysis(buildId),
    enabled: Boolean(buildId),
  });

  if (isLoading)
    return (
      <div style={{ padding: "1rem" }}>
        <SkeletonText paragraph lineCount={6} />
      </div>
    );
  if (error)
    return (
      <InlineNotification
        kind="error"
        title="Failed to load AI analysis"
        subtitle={String(error)}
      />
    );
  if (!analyses || analyses.length === 0) {
    return (
      <p style={{ padding: "1rem", color: "#525252" }}>
        No AI analysis available for this build yet.
      </p>
    );
  }

  const primary =
    analyses.find((a) => a.source === "llm_phase1") ?? analyses[0];
  const phase2 = analyses.find(
    (a) => a.source === "llm_phase2" && a.parent_uid === primary.update_id,
  );

  const analysisTypeLabel: Record<string, string> = {
    failure: "Failure Analysis",
    health: "Health Check",
    scheduling: "Scheduling Analysis",
    performance: "Performance Analysis",
    solution_search: "Knowledge Base Search",
  };

  return (
    <div style={{ padding: "0.5rem 0 0.5rem 1rem" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "0.75rem",
        }}
      >
        <div
          style={{
            display: "flex",
            gap: "0.5rem",
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <AILabel size="xs">
            <AILabelContent>
              {primary.model_name || "N/A"} was used to generate this content
            </AILabelContent>
          </AILabel>
          {primary.analysis_type && (
            <Tag
              type={primary.analysis_type === "failure" ? "red" : "blue"}
              size="sm"
            >
              {analysisTypeLabel[primary.analysis_type] ??
                primary.analysis_type}
            </Tag>
          )}
          <ConfidenceBadge confidence={primary.confidence} />
          {primary.error_category_1 && (
            <Tag type="warm-gray" size="sm">
              {primary.error_category_1}
            </Tag>
          )}
          {primary.error_category_2 && (
            <Tag type="warm-gray" size="sm">
              {primary.error_category_2}
            </Tag>
          )}
        </div>
        <Button
          kind="secondary"
          size="sm"
          onClick={() => setShowFeedback(true)}
        >
          Give feedback
        </Button>
      </div>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
          margin: "0 0 1rem",
        }}
      >
        <div>
          <div
            style={{
              fontSize: "0.875rem",
              fontWeight: 600,
              color: "var(--cds-text-primary)",
              marginBottom: "0.5rem",
            }}
          >
            Summary
          </div>
          <div
            style={{
              fontSize: "0.875rem",
              color: "var(--cds-text-primary)",
              margin: 0,
            }}
          >
            {primary.summary || "N/A"}
          </div>
        </div>
        <div>
          <div
            style={{
              fontSize: "0.875rem",
              fontWeight: 600,
              color: "var(--cds-text-primary)",
              marginBottom: "0.5rem",
            }}
          >
            Root cause
          </div>
          <div
            style={{
              fontSize: "0.875rem",
              color: "var(--cds-text-primary)",
              margin: 0,
            }}
          >
            {primary.root_cause || "N/A"}
          </div>
        </div>
        {primary.suggested_action && (
          <div>
            <div
              style={{
                fontSize: "0.875rem",
                fontWeight: 600,
                color: "var(--cds-text-primary)",
                marginBottom: "0.5rem",
              }}
            >
              Suggested action
            </div>
            <div
              style={{
                fontSize: "0.875rem",
                color: "var(--cds-text-primary)",
                margin: 0,
              }}
            >
              {primary.suggested_action}
            </div>
          </div>
        )}
      </div>

      {(() => {
        const allIssues = primary.issues ?? [];
        const plainIssues = allIssues.filter(
          (i) => i.type !== "step_completed" && i.type !== "output",
        );
        const stepIssues = allIssues.filter((i) => i.type === "step_completed");
        const outputIssues = allIssues.filter((i) => i.type === "output");
        return (
          <>
            {plainIssues.length > 0 && (
              <section style={{ marginBottom: "1rem" }}>
                <div
                  style={{
                    fontSize: "0.875rem",
                    fontWeight: 600,
                    color: "var(--cds-text-primary)",
                    marginBottom: "0.5rem",
                  }}
                >
                  Issues
                </div>
                <ul
                  style={{
                    listStyle: "none",
                    padding: 0,
                    margin: 0,
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.5rem",
                  }}
                >
                  {plainIssues.map((issue, i) => (
                    <li
                      key={i}
                      style={{
                        display: "flex",
                        gap: "0.5rem",
                        alignItems: "flex-start",
                        fontSize: "0.875rem",
                      }}
                    >
                      <Tag
                        type={
                          issue.severity === "critical" || issue.severity === "high"
                            ? "red"
                            : issue.severity === "warning"
                              ? "warm-gray"
                              : "blue"
                        }
                        size="sm"
                      >
                        {issue.severity}
                      </Tag>
                      <span>
                        <em>{issue.type} — </em>
                        {issue.description}
                      </span>
                    </li>
                  ))}
                </ul>
              </section>
            )}
            {stepIssues.length > 0 && (
              <section style={{ marginBottom: "1rem" }}>
                <h4
                  style={{
                    fontSize: "0.875rem",
                    color: "#525252",
                    marginBottom: "0.5rem",
                  }}
                >
                  Steps completed
                </h4>
                <StructuredListWrapper isFlush>
                  <StructuredListBody>
                    {stepIssues.map((issue, i) => (
                      <StructuredListRow key={i}>
                        <StructuredListCell>
                          <Tag type="green" size="sm">
                            step
                          </Tag>
                        </StructuredListCell>
                        <StructuredListCell>
                          {issue.description}
                        </StructuredListCell>
                      </StructuredListRow>
                    ))}
                  </StructuredListBody>
                </StructuredListWrapper>
              </section>
            )}
            {outputIssues.length > 0 && (
              <section style={{ marginBottom: "1rem" }}>
                <h4
                  style={{
                    fontSize: "0.875rem",
                    color: "#525252",
                    marginBottom: "0.5rem",
                  }}
                >
                  Outputs
                </h4>
                <StructuredListWrapper isFlush>
                  <StructuredListBody>
                    {outputIssues.map((issue, i) => (
                      <StructuredListRow key={i}>
                        <StructuredListCell style={{ fontSize: "0.875rem" }}>
                          {issue.description}
                        </StructuredListCell>
                      </StructuredListRow>
                    ))}
                  </StructuredListBody>
                </StructuredListWrapper>
              </section>
            )}
          </>
        );
      })()}

      {phase2?.kb_recommendation && (
        <section
          style={{
            marginBottom: "1rem",
            padding: "0.75rem 1rem",
            background: "var(--cds-layer)",
            border: "1px solid var(--cds-border-subtle-01)",
            borderRadius: "4px",
          }}
        >
          <h4
            style={{
              fontSize: "0.875rem",
              color: "#525252",
              marginBottom: "0.5rem",
            }}
          >
            Knowledge base recommendation
          </h4>
          <p
            style={{
              fontSize: "0.875rem",
              color: "var(--cds-text-primary)",
              margin: 0,
            }}
          >
            {phase2.kb_recommendation}
          </p>
        </section>
      )}

      {failureReason && (
        <section style={{ marginBottom: "1rem" }}>
          <h4
            style={{
              fontSize: "0.875rem",
              fontWeight: 600,
              color: "var(--cds-text-primary)",
              marginBottom: "0.5rem",
            }}
          >
            Failure reason
          </h4>
          <Layer>
            <CodeSnippet type="multi" feedback="Copied!">
              {failureReason}
            </CodeSnippet>
          </Layer>
        </section>
      )}

      <FeedbackModal
        analysis={primary}
        buildId={buildId}
        open={showFeedback}
        onClose={() => setShowFeedback(false)}
      />
    </div>
  );
}
