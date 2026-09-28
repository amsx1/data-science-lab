# Contributing to DATA AUTOPSY

Thanks for taking the time to contribute. This project values **honest analysis**
over impressive-sounding analysis: a detector that reports "we cannot tell" is
worth more than one that guesses confidently.

## Quick start

```bash
git clone https://github.com/amsx1/data-science-lab.git
cd data-science-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest                      # 146 tests must pass
streamlit run app.py                  # check the UI manually
```

## Project rules

1. **One concern per module.** New analyses go in their own module under `src/`
   and return a dataclass declared in `src/schema.py`.
   `ui_sections.py` contains presentation logic only — never analysis.
2. **Thresholds live in `src/config.py`.** No magic numbers inline, and nothing
   may be tuned to the bundled sample dataset.
3. **Every detector ships with tests**, including degenerate inputs: empty
   columns, single-row frames, all-null columns, constants, non-numeric data and
   columns with fewer than four distinct values.
4. **Be honest in the wording.** Outliers are not errors, anomalies are not
   fraud, leakage signals are not verdicts, correlation is not causation, and the
   health score is not a measure of scientific validity. Any new copy must respect
   that.
5. **Style:** type hints on public functions, Google-style docstrings, meaningful
   names, no bare `except`, no TODOs in core paths.
6. **Dependencies:** justify any new one in the pull request description. Unused
   libraries will be rejected.

## Good first contributions

- A new detector with tests (see the roadmap in the README).
- A new file reader (Excel/Parquet/JSON) behind the existing `LoadResult` type.
- Documentation improvements, especially worked examples on real public datasets.
- Bug reports that include the smallest CSV that reproduces the problem.

## Pull request checklist

- [ ] `python -m pytest` passes locally.
- [ ] New behaviour is covered by tests (including failure modes).
- [ ] Thresholds (if any) were added to `src/config.py`.
- [ ] README updated when behaviour, settings or outputs change.
- [ ] No secrets, tokens, credentials or private datasets are included.

## Reporting bugs

Please include: the command you ran, the full traceback, your Python and package
versions (`pip freeze`), and — most importantly — a minimal CSV that reproduces
the issue. If the data is sensitive, reproduce it with the synthetic generator in
`sample_data/make_sample_data.py` instead.

## Code of conduct

Be kind, be specific, assume good faith. Technical disagreement is welcome;
personal criticism is not.
