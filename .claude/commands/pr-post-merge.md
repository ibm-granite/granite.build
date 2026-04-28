---
description: Clean up after a PR is merged — close fork issue, update local branch, delete issue branch
argument-hint: "Fork issue number (e.g., 1)"
---

# Post-merge cleanup

You are helping a developer clean up after a PR has been merged into `upstream/g4os`. Follow these steps in order.

Fork issue number: $ARGUMENTS

## Step 1: Validate

1. Derive the fork owner and repo:
   ```
   FORK_OWNER=$(git remote get-url origin | sed -E 's|.*[:/]([^/]+)/.*|\1|')
   FORK_REPO=$(git remote get-url origin | sed -E 's|.*[:/][^/]+/(.*)$|\1|' | sed 's/\.git$//')
   ```
2. Detect if we are running inside a git worktree:
   ```
   WORKTREE_DIR=$(git rev-parse --show-toplevel)
   MAIN_REPO=$(git worktree list --porcelain | head -1 | awk '{print $2}')
   IS_WORKTREE=false
   if [ "$WORKTREE_DIR" != "$MAIN_REPO" ]; then IS_WORKTREE=true; fi
   ```
3. Confirm the current branch is an issue branch (not `g4os`, `main`, or `dev`)
4. Save the current branch name for later deletion
5. If no issue number was provided ($ARGUMENTS), try to auto-detect from the branch name:
   ```
   BRANCH=$(git branch --show-current)
   ```
   If `$BRANCH` matches the pattern `issue-<N>-*`, extract `<N>` as the issue number. Inform the user:
   > Auto-detected issue #N from branch name `<branch>`

   If the branch doesn't match and no `$ARGUMENTS` was given, ask the user for the issue number.

## Step 2: Close the fork issue

1. Close the issue in the fork repo:
   ```
   gh issue close <N> --repo "$FORK_OWNER/$FORK_REPO"
   ```
2. Confirm the issue was closed successfully

## Step 3: Update local integration branch and clean up

### If running in a worktree (`IS_WORKTREE=true`):

The worktree must be removed before the branch can be deleted. All remaining steps run from the **main repo directory**.

1. Change to the main repo directory:
   ```
   cd $MAIN_REPO
   ```
2. Remove the worktree:
   ```
   git worktree remove $WORKTREE_DIR
   ```
   If removal fails (dirty worktree), ask the user whether to force-remove (`git worktree remove --force $WORKTREE_DIR`) or abort.
3. Update the integration branch:
   ```
   git checkout g4os
   git pull upstream g4os
   ```
4. Delete the local issue branch:
   ```
   git branch -d <saved-branch-name>
   ```
5. Delete the remote issue branch on the fork:
   ```
   git push origin --delete <saved-branch-name>
   ```
6. Report completion to the user:
   > Cleaned up worktree `$WORKTREE_DIR`, deleted branch `<saved-branch-name>` locally and on origin, updated `g4os`.

### If NOT running in a worktree:

1. Switch to the integration branch and pull the latest:
   ```
   git checkout g4os
   git pull upstream g4os
   ```
2. Delete the local issue branch:
   ```
   git branch -d <saved-branch-name>
   ```
3. Delete the remote issue branch on the fork:
   ```
   git push origin --delete <saved-branch-name>
   ```
4. Report completion to the user
