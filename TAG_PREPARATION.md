# Tag Preparation Notes

These notes are for turning the staged or committed export repo into a remote-ready tagged release.

## Versioning

- Keep the tag aligned with the package version in [pyproject.toml](pyproject.toml).
- Prefer a simple `v<version>` form, for example `v1.0.0`.
- If a release candidate is needed, use an explicit suffix such as `v1.0.0-rc1` and avoid reusing the final tag name.

## Recommended Tag Flow

1. Verify `main` contains the intended release commit.
2. Verify `git status --short` is empty.
3. Verify the current commit is the one you want to publish with `git log --oneline -n 1`.
4. Create an annotated tag:

```bash
git tag -a v1.0.0 -m "semantic-memory-mcp-server v1.0.0"
```

5. Inspect the tag metadata:

```bash
git show v1.0.0 --stat
```

6. Push branch and tag:

```bash
git push origin main
git push origin v1.0.0
```

## Tag Message Guidance

Keep the tag message short and factual. A good annotated tag message should mention:

- the repository name
- the released version
- the major release shape, for example public export, federated config, and published benchmark artifacts

Example:

```text
semantic-memory-mcp-server v1.0.0

Public export release with RWIF-backed MCP server, federated config examples, and published comparative benchmark artifacts.
```

## Release Note Inputs

Good sources for the final hosted release note body:

- [README.md](README.md) for user-facing positioning
- [BENCHMARKS.md](BENCHMARKS.md) for defensible benchmark claims
- [SECURITY.md](SECURITY.md) for operational cautions
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the release gate itself

## Things Not To Do

- Do not retag an already-published version name with different contents.
- Do not create the tag before benchmark artifacts and docs are in their final state.
- Do not publish a release note that claims broader vector-database superiority than the shipped benchmark record supports.
- Do not push a release tag from a dirty working tree.