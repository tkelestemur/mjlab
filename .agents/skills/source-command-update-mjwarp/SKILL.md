---
name: "source-command-update-mjwarp"
description: "Update the mujoco-warp dependency to a given commit"
---

# source-command-update-mjwarp

Use this skill when the user asks to run the migrated source command `update-mjwarp`.

## Workflow

Update the `mujoco-warp` dependency to the commit hash provided by the user.

Steps:
1. Read `pyproject.toml` and find the `mujoco-warp` line under
   `[tool.uv.sources]`.
2. Replace only the current `rev = "..."` value on that line with the requested
   commit hash.
3. Run `uv lock` to regenerate the lockfile.
4. Create and switch to a new branch named
   `update-mjwarp/<first-8-chars-of-hash>` (for example,
   `update-mjwarp/e28c6038`).
5. Stage `pyproject.toml` and `uv.lock`, then commit with message:
   `Update mujoco-warp to <first-8-chars-of-hash>`.
6. Push the branch and open a PR with title:
   `Update mujoco-warp to <first-8-chars-of-hash>`.

Important:
- The commit hash is required. If the user did not provide it, ask for the hash
  before editing files.
- Do NOT modify anything else in `pyproject.toml`.
