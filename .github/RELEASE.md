# Releasing `jev-rerank`

Pushing an annotated `vX.Y.Z` Git tag starts
[`release.yml`](workflows/release.yml). The workflow validates the version,
runs the checks, builds the package, and creates the GitHub Release. It does
not publish to PyPI, and it does not require a PyPI token.

## Prerequisites

- Write access to this repository and permission to push tags.
- `git`, [uv](https://docs.astral.sh/uv/), and Python 3.11 or newer locally.
- In the repository's **Settings → Actions → General**, workflow permissions
  must allow **Read and write permissions**. An organization policy can prevent
  the workflow token from creating the release even though the workflow asks
  for `contents: write`.
- A clean working tree. The tag should point at the release commit, not at
  uncommitted local changes.

## Release checklist

Replace `X.Y.Z` below with the new
[PEP 440](https://peps.python.org/pep-0440/) version, for example `0.2.0` or
`0.2.1`.

1. Make the release-worthy code and documentation changes.
2. **Manually change** `version` in `[project]` in `pyproject.toml` to `X.Y.Z`.
   This is the project's single release version source of truth.
3. Refresh the lock file so its editable-package entry has the same version:

   ```sh
   uv lock
   ```

   Review `uv.lock` before committing. A plain `uv lock` should only make the
   needed metadata change unless dependencies were deliberately changed.
4. Run the release checks locally:

   ```sh
   uv sync --locked --all-groups
   uv run --locked ruff check .
   uv run --locked pytest
   uv build
   ```

   `dist/` is ignored and can be left alone; it is recreated in CI.
5. Commit the version and lock-file change, then push the commit to the normal
   release branch:

   ```sh
   git add pyproject.toml uv.lock
   git commit -m "Release X.Y.Z"
   git push origin master
   ```

   Include other release-related files in that commit when applicable.
6. Create and push an **annotated** tag on that exact commit:

   ```sh
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

7. Open the repository's **Actions** tab and wait for the **Release** workflow.
   On success, the **Releases** page contains the generated notes and assets.
   Download the source ZIP and follow the README's release-install section to
   smoke-test it on a clean machine if practical.

## What the workflow publishes

- `jev_rerank-X.Y.Z-py3-none-any.whl` — a small, universal pure-Python wheel.
  Use it with `uvx`, `uv tool install`, `pipx`, or `pip`.
- `jev_rerank-X.Y.Z.tar.gz` — the standard Python source distribution.
- `jev-rerank-X.Y.Z-source.zip` — the recommended unpack-and-run download. It
  has the application source, `pyproject.toml`, `uv.lock`, README, and license,
  but no vendored dependency wheels. After extraction, `uv run --locked
  --no-dev jev-rerank ...` recreates the tested runtime environment.
- `SHA256SUMS.txt` — SHA-256 checksums for the three package files.

The wheel is intentionally not a platform executable and does not bundle
Python or dependencies. That keeps it small and lets `uv`/`pipx` select the
correct dependency wheels for Windows, macOS, and Linux. The project itself is
pure Python, so the same wheel is suitable for all three operating systems.

## If a release fails

- **Tag/version mismatch:** fix `pyproject.toml` and `uv.lock` in a new commit,
  then create a new correctly named tag. Do not move an already-published tag.
- **Checks or build fail:** fix the problem in a new commit and use a new tag.
- **Release creation is forbidden:** enable read/write workflow permissions or
  ask the organization administrator to permit `contents: write`.
- **A later release command says the release already exists:** inspect the
  existing release and its assets before changing anything. Prefer publishing a
  new patch version rather than replacing an artifact people may have already
  downloaded.
