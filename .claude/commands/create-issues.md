---
description: Read an architecture document and create a hierarchy of GitHub issues (epic > milestone sub-epics > implementation issues)
argument-hint: "Path to architecture document (e.g., docs/plans/2026-02-23-architecture.md)"
---

# Create Implementation Issues from Architecture Document

You are helping a developer turn an architecture/design document into a structured set of GitHub issues. The issues are created in the fork repo (derived from `origin` remote).

Architecture document path: $ARGUMENTS

## Step 0: Validate input

If `$ARGUMENTS` is empty or the file does not exist, ask the user to provide a valid path:
```
Usage: /create-issues <path-to-architecture-document>
Example: /create-issues docs/plans/2026-02-23-architecture.md
```
Do not proceed until a valid path is provided.

## Step 1: Read and analyze the document

1. Read the architecture document at the given path
2. Identify the top-level structure:
   - Document title and overall goal
   - Milestones (major sections, typically `## Milestone N: ...`)
   - Sub-steps within each milestone (typically `### N.M ...`)
   - Verification/acceptance criteria per milestone
   - Files to create or modify per milestone
3. Derive the fork owner and repo:
   ```
   FORK_OWNER=$(git remote get-url origin | sed -E 's|.*[:/]([^/]+)/.*|\1|')
   FORK_REPO=$(git remote get-url origin | sed -E 's|.*[:/][^/]+/(.*)(.git)?$|\1|' | sed 's/\.git$//')
   ```

## Step 2: Decide granularity and propose issue plan

For each milestone, decide whether sub-steps should be individual issues or grouped:
- **Split into separate issues** when sub-steps are independent, touch different files, and are substantial enough to be a standalone PR
- **Group into one issue** when sub-steps are tightly coupled, touch the same files, or are small enough to implement together

Present the proposed issue hierarchy to the user:

```
Epic: <document title>
  Milestone 1: <title>
    - Issue: <description>
    - Issue: <description>
  Milestone 2: <title>
    - Issue: <description>
    ...
```

Ask the user to approve or adjust the plan before creating any issues. Use AskUserQuestion.

## Step 3: Create issues bottom-up

Create issues in this order so parent issues can reference child issue numbers:

### 3a. Implementation issues (leaf level)

For each implementation issue, create it with `gh issue create`. Use a HEREDOC for the body to avoid shell escaping issues:

```bash
gh issue create --repo "$FORK_OWNER/$FORK_REPO" --title "<title>" --body "$(cat <<'EOF'
## Context

<Brief description of what this issue covers and why, extracted from the architecture document.>

## Scope

<List of sub-steps from the architecture document that this issue covers.>

## Files

<List of files to create or modify, extracted from the document.>

## Acceptance Criteria

<Specific verification steps from the document's verification section, filtered to this issue's scope.>

## Reference

Source: `<path-to-architecture-document>`, <section reference>
EOF
)"
```

Save each issue's number and title for use in parent issues.

### 3b. Milestone sub-epic issues

After all implementation issues for a milestone are created, create the milestone sub-epic:

Title format: `<Milestone title>`

Use a HEREDOC for the body. Body format:
```markdown
## Goal

<Milestone goal from the architecture document.>

## Dependencies

<If this milestone depends on a previous milestone, note it here: "Depends on #<N> (<milestone title>)". Omit this section for the first milestone or if there are no dependencies.>

## Implementation Issues

- [ ] #<N> <issue title>
- [ ] #<N> <issue title>
- [ ] #<N> <issue title>

## Verification

<The milestone's verification section from the architecture document.>

## Reference

Source: `<path-to-architecture-document>`, <milestone section>
```

Save each sub-epic's number for the epic.

### 3c. Epic issue

After all milestone sub-epics are created, create the epic:

Title format: `<Document title>`

Body format:
```markdown
## Overview

<Document context/goal section — the "why" behind this architecture.>

## Milestones

- [ ] #<N> <milestone title>
- [ ] #<N> <milestone title>
- [ ] #<N> <milestone title>

## Reference

Source: `<path-to-architecture-document>`
```

## Step 4: Report results

Present a summary table to the user:

```
Epic: #<N> <title>

Milestone sub-epics:
  #<N> <title> (X implementation issues)
  #<N> <title> (X implementation issues)
  ...

Total: X issues created (1 epic + Y sub-epics + Z implementation issues)
```