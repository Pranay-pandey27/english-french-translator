# Validation record

## Publication check — 2026-10-05

On Linux with Python 3.12.3, **1 project test passed** using an isolated environment with
the declared runtime dependencies and pytest.
The canonical notebook, saved model or image regeneration, local HTTP service,
Docker build, and hosted GitHub Actions were not rerun in this check. Historical
checks below remain scoped to their recorded environment.

## Earlier validation — 2026-09-11

Date: 2026-09-11. Environment: Linux, Python 3.12.14.

- Existing automated project tests: **1 passed**.
- Canonical notebook: **5 code cells passed** in sequential Python execution from the repository root.
- Notebook metrics and model artifacts were regenerated. Existing scientific result limitations remain applicable.
- All notebook files passed structural validation.
- No high-confidence credential patterns were found in the inspected text files.

- Saved artifacts loaded successfully and a real local HTTP inference request returned successfully.

The Jupyter kernel transport could not start in this execution environment.
Notebook code was therefore executed using the existing code-cell runner,
not through a live Jupyter frontend. Docker builds and hosted GitHub Actions
have not been run.

Direct dependency versions are recorded in `requirements-validated.txt`.
Regenerate saved artifacts after changing dependency versions if compatibility
is uncertain.

These checks validate the included project scope; they do not certify a
production service or resolve all research and evaluation limitations.
