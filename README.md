# Knowledge Hypergraph Assessment

## Start here: clean_tshc

**[clean_tshc/](clean_tshc/)** is the main submission: the standalone implementation, reproducibility instructions, tests, evaluated outputs and reports.

**The full write-up is [clean_tshc/report/full_report.tex](clean_tshc/report/full_report.tex). This is where I wrote everything.** A compiled [PDF version](clean_tshc/report/full_report.pdf) is included for convenient reading.

- [Main submission README and reproduction instructions](clean_tshc/README.md)
- [Concise generated assessment report](clean_tshc/report.md)
- [Machine-readable results](clean_tshc/outputs/metrics.json)

## V2 retrieval supplement and A6 technical explanation

**Read [v2_a6_technical_explanation.pdf](clean_tshc_v2/report/v2_a6_technical_explanation.pdf)** for the detailed technical explanation of entity-centric scientific retrieval over the temporal knowledge hypergraph. The editable [LaTeX source](clean_tshc_v2/report/v2_a6_technical_explanation.tex) is included.

The report explains canonical answer entities, answer-family filtering, two query views, dense and BM25 retrieval, safe name matching, evidence retrieval, bounded graph expansion, and reciprocal rank fusion. It then walks through the A6 cross-encoder reranker, which scores the first 100 fused candidates using identity-first, query-dependent evidence cards capped at 512 tokens. It includes the complete algorithm, a worked example, temporal integrity, prediction/evaluation isolation, reproducibility, and failure analysis.

For the report's A6_100 configuration, the development benchmark contains 14 scored questions and 63 target occurrences, including 49 directly mapped to graph identities. The candidate pool admits 37 of those 49 mapped targets; final retrieval recovers 12/63 targets at top 10, 18/63 at top 25, and 37/63 at top 100, with MRR 0.167. These are development diagnostics, not held-out performance: the benchmark influenced design and the choice of reranking depth. This configuration is distinct from the fixed top500 A6/A7 comparison summarized in the V2 README.

- [V2 implementation, reproduction instructions, and A6/A7 comparison](clean_tshc_v2/README.md)
- [V2 report collection](clean_tshc_v2/report/)
- [Additional V2 working copy](clean_tshc_v2%20copy/)

V2 supplements the original hierarchy assessment; the main submission remains in `clean_tshc`. Source, reports, inputs, and experiment artifacts are included. The two archived prediction files exceeding GitHub's regular file-size limit use Git LFS; run `git lfs pull` after cloning to retrieve their full contents.

## Earlier work: rest_of_work

**[rest_of_work/](rest_of_work/)** contains the earlier implementations, retrieval experiments, reports, evaluation artifacts and development history. It is provided for anyone interested in what else I explored; it is separate from the main submission.

See the [archive guide](rest_of_work/ARCHIVE_GUIDE.md) before using those experiments. Historical reports retain the conclusions and paths from their own runs. Local virtual environments, model/embedding caches, temporary build files and generated ZIP bundles are excluded from GitHub; source, reports, inputs and research outputs are included.
