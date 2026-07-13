# Tests

Regression tests for the parts of the VLM Council pipeline that are most
prone to silent breakage:

- `test_coordinates.py`, coordinate parsing from the judge's free-text
  output. Covers the happy path, malformed inputs, out-of-range values,
  and the historical `(0, 0)` fake fallback that we deliberately removed.
- `test_evaluate.py`, country name matching and neighbour lookup used by
  `vlm_council/evaluate.py` to score predictions. Guards against subtle
  regressions in alias handling and normalisation.

## Running

From the project root:

```bash
python -m pytest tests -q
```

Or, without pytest installed, each test file also runs standalone:

```bash
python -m tests.test_coordinates
python -m tests.test_evaluate
```

Both files use only the Python standard library (`unittest`), so no
additional dev dependencies are required.
