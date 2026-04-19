---
name: commit-by-concept
description: Split a git worktree into multiple commits based on concept, then push them. Use when the user asks to commit and push current changes separately by concept, especially when there are mixed code, tests, docs, data, or artifact changes in one tree.
---

# Commit by Concept

Use this skill when the user wants the current worktree committed and pushed in multiple commits organized by concept.

## Workflow

1. Inspect the worktree with `git status --short` and read the diffs for each changed area.
2. Group paths into narrow concepts. Prefer smaller commits when the split is defensible.
3. Keep code and matching tests together when they express one behavior change.
4. Keep generated artifacts, datasets, checkpoints, logs, and run outputs separate from source changes.
5. Stage only one concept at a time, review the staged diff, commit, then move to the next concept.
6. Push only after all intended commits are created.

## Rules

- Split as much as the concept boundaries justify, but do not create arbitrary micro-commits.
- Use the repo's existing artifact policy. If the repo already tracks large artifacts or LFS objects, follow that pattern.
- If a file is obviously too large or generated and the repo does not normally commit it, leave it uncommitted and say so plainly.
- Do not mix unrelated concepts into one commit just because they touch nearby files.
- Do not rewrite or squash existing user commits unless explicitly asked.

## Commit Shape

- Prefer imperative commit messages.
- Make each message behavior-specific.
- Good examples:
  - `tighten TRL chat record conversion`
  - `update SFT tests for chat message rows`
  - `record v3 training environment note`

## Review Checklist

Before each commit:

- The staged paths all support one concept.
- No unrelated files are staged.
- Generated or oversized files are included only if that matches repo practice.
- The commit message names the concept directly.

Before push:

- `git status --short` is clean, or the remaining files are intentionally excluded artifacts.
- `git log --oneline --decorate -N` matches the intended concept split.
- Push the current branch to its configured remote.

## Reporting

When done, report:

- the commits created
- the branch pushed
- any files intentionally left uncommitted and why
