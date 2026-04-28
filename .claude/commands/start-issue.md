---
description: Load a GitHub issue, create an isolated worktree, load architecture docs, and plan implementation
argument-hint: "Issue number (e.g., 42)"
---

# Start Work on a GitHub Issue

You are helping a developer start work on a GitHub issue. You will verify pre-conditions, create an isolated git worktree, load the issue and architecture documentation, assess what's already been implemented, and plan the work. Follow these steps in order, stopping if any step fails.

Issue number: $ARGUMENTS

## Defaults

These values can be overridden by the user but apply by default:

```
ARCH_DOC_1=docs/plans/2026-02-23-architecture.md
ARCH_DOC_2=docs/plans/2026-02-18-standalone-deployment-design.md
WORKTREE_BASE=/home/cma/de/cma/gbserver-worktrees
```

Derive the fork owner and repo:
```
FORK_OWNER=$(git remote get-url origin | sed -E 's|.*[:/]([^/]+)/.*|\1|')
FORK_REPO=$(git remote get-url origin | sed -E 's|.*[:/][^/]+/(.*)$|\1|' | sed 's/\.git$//')
```

## Step 0: Pre-flight checks

1. If `$ARGUMENTS` is empty or not a number, show usage and stop:
   ```
   Usage: /start-issue <issue-number>
   Example: /start-issue 42
   ```
2. Confirm we are on the `g4os` branch:
   ```
   git branch --show-current
   ```
   If not on `g4os`, warn the user and ask whether to switch (`git checkout g4os`) or abort.
3. Check for uncommitted changes:
   ```
   git status --porcelain
   ```
   If there are uncommitted changes, ask the user what to do:
   - **Commit** them before proceeding
   - **Stash** them (`git stash push -m "stash before issue-$ARGUMENTS"`)
   - **Abort** and let the user handle it manually
4. Fetch the latest upstream:
   ```
   git fetch upstream
   ```

## Step 1: Create branch and worktree

1. Fetch the issue title to verify the issue exists and to derive the branch name:
   ```
   gh issue view $ARGUMENTS --repo "$FORK_OWNER/$FORK_REPO" --json title --template '{{.title}}'
   ```
   If the command fails (issue doesn't exist), report the error and stop.

2. Slugify the title: lowercase, replace non-alphanumeric characters with hyphens, collapse consecutive hyphens, strip leading/trailing hyphens, truncate the slug so the total branch name (`issue-$ARGUMENTS-<slug>`) stays under 50 characters.
   Branch name: `issue-$ARGUMENTS-<slug>`

3. Check if a branch with this issue number already exists:
   ```
   git branch --list "issue-$ARGUMENTS-*"
   ```
   If a matching branch exists, check whether it already has a worktree:
   ```
   git worktree list --porcelain | grep -A2 "branch refs/heads/issue-$ARGUMENTS-"
   ```
   Ask the user:
   - **Reuse** the existing branch — if a worktree exists, use it; if not, create one: `git worktree add $WORKTREE_BASE/<branch-name> <branch-name>`
   - **Abort** so the user can handle it manually

4. Create the worktree base directory if it doesn't exist:
   ```
   mkdir -p $WORKTREE_BASE
   ```

5. Create the worktree with a new branch based on `upstream/g4os`:
   ```
   git worktree add -b <branch-name> $WORKTREE_BASE/<branch-name> upstream/g4os
   ```
   If the worktree directory already exists, ask the user: reuse it, or remove and recreate it.

6. Report to the user:
   > Created worktree at `$WORKTREE_BASE/<branch-name>` on branch `<branch-name>`

7. Copy untracked configuration files that won't exist in the fresh worktree:
   ```
   cp -r .claude/settings.json $WORKTREE_BASE/<branch-name>/.claude/settings.json 2>/dev/null || true
   ```

8. Set up the virtual environment in the worktree:
   ```
   cd $WORKTREE_BASE/<branch-name> && make venv && source .venv/bin/activate
   ```
   If `make venv` fails (e.g., missing `ARTIFACTORY_USER`/`ARTIFACTORY_API_KEY`), warn the user and continue — the worktree is still usable but tests and linting won't work until venv is set up.

9. **All subsequent commands must execute in the worktree directory.** Use `cd $WORKTREE_BASE/<branch-name>` or pass full paths.

## Step 2: Load issue context

1. Fetch full issue details:
   ```
   gh issue view $ARGUMENTS --repo "$FORK_OWNER/$FORK_REPO" --json number,title,body,labels,comments
   ```
2. Present to the user:
   - **Issue #** and **title**
   - **Labels** (if any)
   - **Body** — the full issue description
   - **Comments** — summarize key discussion points (if any)

## Step 3: Load architecture documentation

1. Read the detailed implementation plan (use the Read tool):
   - File: `$ARCH_DOC_1`
2. Read the strategic design overview (use the Read tool):
   - File: `$ARCH_DOC_2`

## Step 4: Summarize and plan

1. Present a consolidated summary:
   - **What the issue asks for** (from Step 2)
   - **What the architecture docs specify** for this area (from Step 3)
   - **What remains to be done** for this specific issue
2. Use the `superpowers:brainstorming` skill to explore the problem space and align with the user on the implementation approach
3. After brainstorming, use the `superpowers:writing-plans` skill to create a detailed implementation plan for the issue
