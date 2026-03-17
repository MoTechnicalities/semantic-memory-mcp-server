# Release Checklist

Use this checklist before pushing the repository to a public remote or cutting a release tag.

## Repository Prep

1. Confirm the working tree is clean with `git status --short`.
2. Rebuild published benchmark artifacts with `python scripts/regenerate_benchmark_artifacts.py`.
3. Re-run the focused validation suite with `PYTHONPATH=src python -m unittest discover -s tests -p 'test_semantic_memory_federation.py'`.
4. Review [BENCHMARKS.md](BENCHMARKS.md) and confirm the benchmark claims still match the generated artifacts.
5. Review [README.md](README.md) and confirm the quick-start commands, Docker examples, and sample-data references still match the repo contents.
6. Review [SECURITY.md](SECURITY.md) and confirm the write-surface guidance still matches the exposed tools and CLI flags.

## Remote Readiness

1. Add the remote: `git remote add origin <REMOTE_URL>`.
2. Verify the remote target with `git remote -v`.
3. Confirm the default branch is `main`.
4. Push the branch with `git push -u origin main`.
5. Check that the hosted repository shows the benchmark artifacts, sample data, workflow file, and release docs exactly as expected.

## Pre-Tag Review

1. Confirm the package version in [pyproject.toml](pyproject.toml) matches the intended release.
2. Confirm the release scope is stable enough to tag: code, sample data, docs, and benchmark artifacts should all be part of the same state.
3. Decide whether the tag should be lightweight internal bookkeeping or an annotated public release tag.
4. Draft the release summary from the benchmark record and README positioning rather than writing it ad hoc at tag time.

## Suggested Release Payload

Include these points in the hosted release notes:

- RWIF-backed MCP semantic-memory server export
- federated config with per-store access policy
- Docker, Compose, and Docker Desktop examples
- shipped benchmark artifacts for storage-shaped, small workload-shaped, and larger workload-shaped comparisons
- focused CI and test coverage for the federated server path

## After Release

1. Create and push the version tag described in [TAG_PREPARATION.md](TAG_PREPARATION.md).
2. Publish the hosted release entry that references the tag.
3. Smoke-test the published quick-start path from a fresh clone.
4. If benchmark artifacts were regenerated for the release, confirm the hosted repo and release archive both include the updated files.