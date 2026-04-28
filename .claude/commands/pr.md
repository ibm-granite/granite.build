---
description: Format, lint, review, and create or update a PR targeting g4os
argument-hint: "Optional fork issue number (e.g., 42)"
---

# PR for g4os

You are helping a developer create or update a pull request in the upstream repo (`granite-dot-build/gbserver`) targeting the `g4os` branch. Follow these steps in order, stopping if any step fails.

Fork issue number (if provided): $ARGUMENTS

## Step 0: Detect worktree context

1. Check if we are running inside a git worktree:
   ```
   git rev-parse --is-inside-work-tree && git worktree list --porcelain
   ```
   If the current directory is a secondary worktree (not the main working tree), note this — all git operations work normally from worktrees since they share remotes and history.

2. If `$ARGUMENTS` is empty, try to auto-detect the issue number from the branch name:
   ```
   BRANCH=$(git branch --show-current)
   ```
   If `$BRANCH` matches the pattern `issue-<N>-*`, extract `<N>` as the fork issue number. Inform the user:
   > Auto-detected issue #N from branch name `<branch>`

3. Derive the fork owner:
   ```
   FORK_OWNER=$(git remote get-url origin | sed -E 's|.*[:/]([^/]+)/.*|\1|')
   ```

## Step 1: Pre-flight checks

1. Confirm we are NOT on `g4os`, `main`, or `dev` branches — we should be on an issue branch
2. Run `git status` to check for uncommitted changes
3. If there are unstaged/uncommitted changes, ask the user whether to stage and commit them before proceeding
4. Check if a PR already exists for this branch:
   ```
   gh pr list --repo granite-dot-build/gbserver --head "$FORK_OWNER":<current-branch-name> --state open --json number,url
   ```
   Save the result — this determines whether we create or update.

## Step 2: Format and lint (PR files only)

Only check files that will be in the PR — those changed between `upstream/g4os` and `HEAD`.

1. Get the list of Python files changed in this PR:
   ```
   git diff upstream/g4os...HEAD --name-only -- '*.py'
   ```
2. If there are Python files, run `isort --profile black` and `black` on each file
3. If there are Python files, run `pylint` and `mypy --disable-error-code=import-untyped` directly on each changed file
4. If linting produces errors, show them to the user and ask whether to fix them or proceed anyway
5. If formatting changed any files, stage and commit them with message "style: auto-format via pre-commit"

## Step 3: Push to fork

1. Push the current branch to `origin` (the fork) with tracking:
   ```
   git push -u origin HEAD
   ```

## Step 4: Code review

1. Use the `superpowers:requesting-code-review` skill to review the changes that will be in the PR
2. Review the diff between the current branch and `upstream/g4os`:
   ```
   git diff upstream/g4os...HEAD
   ```
3. Present the review findings to the user
4. Ask the user if they want to address any findings before proceeding, or continue

## Step 5: Create or update the PR

### If NO existing PR was found in Step 1:

1. Fetch the latest upstream to ensure we have current refs:
   ```
   git fetch upstream
   ```
2. Create the PR in the upstream repo targeting `g4os`:
   ```
   gh pr create --repo granite-dot-build/gbserver --base g4os --head "$FORK_OWNER":<current-branch-name>
   ```
3. The PR title should be concise (under 70 characters)
4. The PR body should include:
   - A summary section with bullet points describing the changes
   - If a fork issue number was provided ($ARGUMENTS), include: `Closes $FORK_OWNER/gbserver#<N>`
   - A test plan section
   - The standard footer: `Generated with [Claude Code](https://claude.com/claude-code)`
5. Return the PR URL to the user

### If an existing PR WAS found in Step 1:

1. The push in Step 3 already updated the PR with the new commits
2. Report to the user: "Updated existing PR <url> with new commits"
3. If the user wants to update the PR title or description, use:
   ```
   gh pr edit <number> --repo granite-dot-build/gbserver --title "..." --body "..."
   ```

## Step 6: Post-PR guidance

If running inside a worktree, remind the user:

> **Next steps after the PR is merged:**
> Use `/pr-post-merge` to clean up — it will close the fork issue, update the local branch, and delete the issue branch. Then remove the worktree:
> ```
> cd /home/cma/de/cma/gbserver
> git worktree remove <worktree-path>
> ```
