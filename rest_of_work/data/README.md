# Supplied assessment data

The supplied files were discovered recursively at `data/data/`:

- `tkh_collection10.json`
- `questions.csv` (semicolon-delimited)
- `ground_truth.json`
- `collection10_articles.csv` (comma-delimited)

Keep them unchanged. `configs/default.yaml` sets `data_dir: data/data`; the loader recursively finds each required file and rejects ambiguous duplicates. If starting without data, place the supplied files here and update the configuration. No replacement data is downloaded automatically. Dataset sharing/publication permission is not assumed by the code.

`meta.date_semantics` distinguishes first corpus evidence from historical origin and asserting-paper dates from fact dates. The implementation records and conservatively defers inconsistent edges; see `QUESTIONS.md` and the generated data-quality artifact.
