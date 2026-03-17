# v1.0.1 Tag Plan

This follow-up tag is not for a feature release. Its purpose is to realign the public tag snapshot with the current `main` branch after the initial release workflow settled.

## Why v1.0.1 Exists

The current public state has a mild coherence gap:

- the GitHub release page copy is already polished and publicly visible
- the repository landing page on `main` reflects the tightened release-note version
- the `v1.0.0` tag still points to commit `53ca470`, while `main` has moved to `f91667e`

That means the release UI and the current branch presentation are slightly more polished than the tagged source snapshot.

## Delta Since v1.0.0

Current `main` includes these post-tag changes:

1. `Merge remote bootstrap commit`
2. `Tighten v1.0.0 release notes`

Practical effect:

- no new runtime feature was introduced after `v1.0.0`
- no benchmark artifact changed after `v1.0.0`
- the main visible change is tighter release-copy alignment and repository-history cleanup

## Recommended Positioning

Treat `v1.0.1` as a coherence and packaging follow-up release.

Suggested summary:

`Align the public release tag with the current main branch after release-note tightening and remote bootstrap history reconciliation. No runtime API changes.`

## Pre-Tag Checks

1. Confirm `git status --short` is empty.
2. Confirm `main` still points at the intended follow-up commit.
3. Reconfirm that no benchmark artifacts changed after `v1.0.0`.
4. Decide whether to keep the existing `v1.0.0` release as-is and publish `v1.0.1` as the corrected follow-up, rather than editing tag history.

## Tag Commands

From the export repo root:

```bash
git checkout main
git pull --ff-only origin main
git tag -a v1.0.1 -m "semantic-memory-mcp-server v1.0.1

Follow-up release aligning the public tag snapshot with the current main branch after release-note tightening and bootstrap history reconciliation."
git show v1.0.1 --stat
git push origin v1.0.1
```

## Suggested GitHub Release Body

```md
# semantic-memory-mcp-server v1.0.1

Follow-up release aligning the public tag snapshot with the current `main` branch.

## What Changed

- realigns the tagged source snapshot with the current public landing-page state
- includes the tightened release-note copy now present on `main`
- preserves the same runtime surface, benchmark artifacts, and federated MCP server behavior as `v1.0.0`

## Scope

This is a release-coherence update, not a feature expansion.
```

## What Not To Do

- Do not move or recreate `v1.0.0`.
- Do not describe `v1.0.1` as a major runtime change.
- Do not claim new benchmark wins unless the benchmark artifacts themselves change.

## Decision Rule

If the goal is clean public provenance, create `v1.0.1`.

If the goal is only to keep `v1.0.0` as the sole initial public marker and the current mismatch is acceptable, do nothing and leave the existing release history intact.