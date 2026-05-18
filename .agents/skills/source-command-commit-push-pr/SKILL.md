---
name: "source-command-commit-push-pr"
description: "Commit, push, and open a PR"
---

# source-command-commit-push-pr

Use this skill when the user asks to run the migrated source command `commit-push-pr`.

## Workflow

1. Inspect the current repository state:
   - `git status`
   - `git diff HEAD`
   - `git branch --show-current`
2. If the current branch is `main` or `master`, create and switch to a new branch
   that follows the repository's branch naming guidance.
3. Stage the intended changes only.
4. Create a single commit with a concise message that matches the change.
5. Push the branch to `origin`.
6. Open a pull request with `gh pr create`.

Use `git` and `gh` shell commands as needed. Codex tool permissions are controlled
by the active session configuration, not by this skill.

Before creating the pull request, summarize the branch, commit, and PR URL in the
final response.
