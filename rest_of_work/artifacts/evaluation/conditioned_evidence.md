# Requirement-conditioned evidence retrieval

At 2025, CR6 recovers 12/47 canonical target instances at ten, compared with CR1’s 11; CR7 recovers 10. CR6 median gold rank is 24.0000, versus CR1’s 29.0000.

At 2026, CR6 recovers 13/50 canonical target instances at ten, compared with CR1’s 11; CR7 recovers 10. CR6 median gold rank is 24.0000, versus CR1’s 28.5000.

These are development diagnostics on the repeatedly used benchmark, not held-out validation or human scientific adjudication. H2, CR0, CR1 and CR3 reproduce exactly. CR2/CR4 remain archived; the new path makes zero NLI calls and downloads no models. All previous reports and metric sections are preserved.

## Frozen protocol and implementation

`CandidateEvidencePool` reuses `CandidateEvidenceIndex.pools` before whole-question semantic selection. The same safe mentions, eligible low-arity relations, genericness filter, snapshot, entity groups and provenance-title policy apply. Pools are built once per candidate × snapshot, with fixed field caps 20 direct claims, 20 explicit-name claims, 15 tasks/problems, 15 technical items, 10 datasets/metrics and 10 titles. Stable structural priority and node IDs decide caps; normalized identical text appears once, with merged source links. The pool is bounded eligible graph evidence, not every possible statement about an entity. Some larger structurally capped pools may still omit useful text.

No new path expansion is performed: safe named claims are already discovered by the unchanged MentionIndex. Historical query-seeded paths remain in each selection export as diagnostics; they are not inputs to the query-independent pool. This deliberately avoids importing a neighboring unnamed claim as candidate support. The new pool need not be a strict superset of the old bundle because old query-seeded paths and structural-cap choices can differ. Whole-article claim expansion is excluded.

`ConditionedEvidenceReranker` extends the existing cosine-reranker interface. Each original component substring is embedded using pinned normalized BGE-small-en-v1.5; evidence vectors come from the existing ContextIndex cache. Top three items are selected by nonnegative cosine × attribution weight, with the historical cosine/ID tie rule. Maximum support enters the unchanged `aggregate_support`: core × geometric mean of positive components × exclusion factor × soft type prior. Top-three export does not mean three items are averaged. Weights remain 1 direct/winner, .8 explicit name, .15 mention, .25 uncertain and 0 comparison baseline/unnamed path. The .5 threshold is diagnostic only. Exclusion cosine cannot establish negation or compliance.

`ConditionedQuestionDecomposition` subclasses the frozen `QuestionDecomposition`. It recovers answer-head modifiers before a recognized generic noun, splits a leading hyphenated adjective from a remaining compound noun, and retains coordinated requirements with a shared modal predicate as one exact source span. It adds qualifiers as positive components without filtering candidates. All old parser/scorer modules are byte-preserved. The additive attribution adapter recognizes sentence-initial participial “Including …” followed by a limited generic finite-predicate pattern as safe explicit-name support (.8), while preserving comparison/negation/list safeguards. It does not upgrade this to direct candidate subject support. The heuristic remains incomplete.

The main comparison changes evidence access and these explicitly requested generic grammar fixes together. Two fixed evaluation diagnostics separate them: “pool_only” applies old decomposition and old attribution to the new pool; “without_qualifiers” removes only qualifier components from otherwise identical CR6. Neither control chooses a configuration or trains weights. A within-candidate permutation of the same component scores is largely invariant under geometric pooling and would be an uninformative null; no shuffled variant is added.

| Q | Answer type | Qualifiers | Core | Requirements | Exclusions |
| --- | --- | --- | --- | --- | --- |
| Q1 | method |  | predicting interatomic potentials | accuracy close to DFT; computational efficiency; systems of millions of atoms |  |
| Q2 | method |  | accelerating electronic structure calculations | self-consistent field iterations become prohibitively expensive; large systems (>10³ atoms) |  |
| Q3 | method |  | learning DFT Hamiltonians | dealing with twisted van der Waals heterostructures containing thousands of magnetic atoms |  |
| Q4 | method |  | phonon property prediction | universal MLIPs show inaccuracies despite good energy/force performance |  |
| Q5 | method |  | coarse-graining molecular dynamics | crystalline symmetries and long-range elastic interactions must be preserved |  |
| Q6 | method |  | predicting ionic conductivity | full ab initio MD is too expensive; accuracy must be maintained |  |
| Q7 | method |  | predicting phase transitions | screening ~50,000 inorganic compounds at high throughput |  |
| Q8 | method |  | predicting fracture patterns | generalizing from atomistic MD data to unseen bicrystalline structures |  |
| Q9 | method |  | ensuring physical consistency | transferring information between scales | violating thermodynamic principles |
| Q10 | method |  | representing atomic environments | the system exhibits significant compositional diversity, defects; disorder |  |
| Q11 | method |  | integrating features from three or more distinct scales simultaneously within a unified framework |  |  |
| Q12 | method | ML-based; interatomic potential | large-scale atomistic simulation |  |  |
| Q13 | architecture | GNN | predicting properties of solid-state crystalline materials |  |  |
| Q14 | — |  | evaluate MLIPs and property prediction models for materials discovery |  |  |



## Quality, oracle and cost

### 2025

| Stage | Hit1 | Hit5 | Hit10 | Hit20 | MacroR5 | MacroR10 | MacroR20 | MicroR5 | MicroR10 | MicroR20 | UniqueR10 | Raw | MRR | R-Prec | R@R | Median | Mean rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2 | 0.0833 | 0.1667 | 0.2500 | 0.3333 | 0.1667 | 0.1750 | 0.2028 | 0.0426 | 0.0638 | 0.0851 | 0.0882 | 3/63 | 0.1602 | 0.0917 | 0.0917 | 66.0000 | 103.4043 |
| CR0-100 | 0.2500 | 0.3333 | 0.5000 | 0.5833 | 0.2028 | 0.2208 | 0.3041 | 0.0851 | 0.1277 | 0.1489 | 0.1765 | 6/63 | 0.3278 | 0.2208 | 0.2208 | 89.0000 | 107.7872 |
| CR1-100 | 0.4167 | 0.6667 | 0.6667 | 0.7500 | 0.4026 | 0.4102 | 0.4924 | 0.2128 | 0.2340 | 0.3617 | 0.2941 | 11/63 | 0.5283 | 0.3268 | 0.3268 | 29.0000 | 90.8723 |
| CR3-100 | 0.2500 | 0.5000 | 0.5833 | 0.8333 | 0.2958 | 0.3470 | 0.5693 | 0.1277 | 0.2128 | 0.3830 | 0.2647 | 10/63 | 0.3366 | 0.2915 | 0.2915 | 53.0000 | 93.2979 |
| CR6-100 | 0.4167 | 0.6667 | 0.6667 | 0.7500 | 0.4026 | 0.4379 | 0.5291 | 0.2128 | 0.2553 | 0.4468 | 0.3235 | 12/63 | 0.5134 | 0.3268 | 0.3268 | 24.0000 | 88.0000 |
| CR7-100 | 0.2500 | 0.4167 | 0.5833 | 0.8333 | 0.2854 | 0.3470 | 0.5686 | 0.1064 | 0.2128 | 0.3830 | 0.2647 | 10/63 | 0.3405 | 0.3193 | 0.3193 | 40.0000 | 91.2128 |
| CR6-200 | 0.4167 | 0.5833 | 0.6667 | 0.7500 | 0.3867 | 0.4379 | 0.4783 | 0.1702 | 0.2553 | 0.3404 | 0.3235 | 12/63 | 0.5112 | 0.3268 | 0.3268 | 31.0000 | 81.1489 |



### 2026

| Stage | Hit1 | Hit5 | Hit10 | Hit20 | MacroR5 | MacroR10 | MacroR20 | MicroR5 | MicroR10 | MicroR20 | UniqueR10 | Raw | MRR | R-Prec | R@R | Median | Mean rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2 | 0.0833 | 0.2500 | 0.2500 | 0.3333 | 0.1742 | 0.1742 | 0.2020 | 0.0600 | 0.0600 | 0.0800 | 0.0811 | 3/63 | 0.1694 | 0.0909 | 0.0909 | 73.0000 | 98.9600 |
| CR0-100 | 0.1667 | 0.4167 | 0.4167 | 0.6667 | 0.2124 | 0.2124 | 0.3141 | 0.1000 | 0.1000 | 0.1600 | 0.1351 | 5/63 | 0.2613 | 0.2188 | 0.2188 | 90.5000 | 101.8200 |
| CR1-100 | 0.4167 | 0.5833 | 0.6667 | 0.7500 | 0.3779 | 0.4047 | 0.4883 | 0.1400 | 0.2200 | 0.3600 | 0.2703 | 11/63 | 0.5218 | 0.3214 | 0.3214 | 28.5000 | 84.8400 |
| CR3-100 | 0.3333 | 0.5000 | 0.5000 | 0.7500 | 0.2918 | 0.2982 | 0.4715 | 0.1200 | 0.1400 | 0.3200 | 0.1892 | 7/63 | 0.3977 | 0.2768 | 0.2768 | 50.5000 | 87.9600 |
| CR6-100 | 0.4167 | 0.5833 | 0.6667 | 0.7500 | 0.3843 | 0.4401 | 0.5063 | 0.1600 | 0.2600 | 0.4000 | 0.3243 | 13/63 | 0.5077 | 0.3290 | 0.3290 | 24.0000 | 83.0000 |
| CR7-100 | 0.3333 | 0.5000 | 0.5833 | 0.8333 | 0.2918 | 0.3226 | 0.5624 | 0.1200 | 0.2000 | 0.3600 | 0.2703 | 10/63 | 0.4161 | 0.3226 | 0.3226 | 41.5000 | 86.4000 |
| CR6-200 | 0.4167 | 0.5833 | 0.6667 | 0.7500 | 0.3843 | 0.4123 | 0.4792 | 0.1600 | 0.2400 | 0.3400 | 0.2973 | 12/63 | 0.5054 | 0.3290 | 0.3290 | 30.0000 | 76.5200 |



All 14 Type A questions appear in every stage, with explicit evaluability. Q5 and Q11 have no canonical named-entity units: canonical recall is null, their raw labels remain in /63. Macro/Hit averages use 12 evaluable questions; micro denominators are 47 (2025) and 50 (2026), with 34/37 unique canonical labels. Costs average all 14. Type B retrieval and historical outputs are unchanged. Complete per-question metrics are in both the results export and the additive metrics.json section.

| Year | Stage | Frozen oracle | Oracle microR10 | Actual microR10 | Gap | Oracle macroR10 |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | CR1-100 | 31/47 | 0.6596 | 0.2340 | 0.4255 | 0.7146 |
| 2025 | CR6-100 | 31/47 | 0.6596 | 0.2553 | 0.4043 | 0.7146 |
| 2025 | CR7-100 | 31/47 | 0.6596 | 0.2128 | 0.4468 | 0.7146 |
| 2025 | CR6-200 | 43/47 | 0.9149 | 0.2553 | 0.6596 | 0.9674 |
| 2026 | CR1-100 | 32/50 | 0.6400 | 0.2200 | 0.4200 | 0.7040 |
| 2026 | CR6-100 | 32/50 | 0.6400 | 0.2600 | 0.3800 | 0.7040 |
| 2026 | CR7-100 | 32/50 | 0.6400 | 0.2000 | 0.4400 | 0.7040 |
| 2026 | CR6-200 | 44/50 | 0.8800 | 0.2400 | 0.6400 | 0.8794 |



The exact prior subset-coverage oracle remains evaluation-only; every per-question oracle dictionary is identical. Candidate prefixes and the entire unreranked tail are preserved. No oracle or expected label reaches pool construction, decomposition, evidence selection or scoring.

| Year | Pool build seconds | Candidate pools | Evidence items |
| --- | --- | --- | --- |
| 2025 | 4.4441 | 2564 | 11306 |
| 2026 | 5.0115 | 2959 | 12306 |



| Year | Stage | New component dots | Reused CR1 scalar scores | Reused evidence vectors | New component strings | Support comparisons | New seconds | Total seconds* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | CR6-100 | 2881.4286 | 1033.1429 | 1567.5714 | 0.2857 | 4480.0714 | 0.0191 | 0.5416 |
| 2025 | CR7-100 | 2881.4286 | 1033.1429 | 1567.5714 | 0.2857 | 4480.0714 | 0.0193 | 0.5418 |
| 2025 | CR6-200 | 2881.4286 | 1033.1429 | 1567.5714 | 0.2857 | 6744.8571 | 0.0268 | 0.5493 |
| 2026 | CR6-100 | 3114.6429 | 1108.2143 | 1689.7857 | 0.2857 | 4615.9286 | 0.0187 | 0.6343 |
| 2026 | CR7-100 | 3114.6429 | 1108.2143 | 1689.7857 | 0.2857 | 4615.9286 | 0.0189 | 0.6344 |
| 2026 | CR6-200 | 3114.6429 | 1108.2143 | 1689.7857 | 0.2857 | 7161.0000 | 0.0265 | 0.6420 |



*Component-sum estimates use historical H2 search plus current vector preparation, evidence scoring and fusion. Pool construction is offline and separately tabulated. Model/index loading, historical reference recomputation, evaluation exports and diagnostic controls are excluded. Primary runs conservatively pay shared top-200 vector preparation, so K100 may be overcharged. Component vectors use the existing persistent embedding cache; reported new strings are relative to CR1, not necessarily neural cache misses on a rerun. Evidence encodings and NLI calls are zero. These single-machine timings are not controlled serving benchmarks.

## Evidence selection and noise

The pool audit finds 980 repeated candidate/question/item instances with positive old attribution missing from the new pool: {'structural_cap_or_deduplication': 974, 'query_seeded_path_excluded': 6}. Full texts and reasons are exported. The additive participial rule changes 49 repeated item roles, of which 6 become a selected component maximum. These are not globally unique claims.

| Year | K | Group | Matched components | Mean delta | Median delta | Improved | Same | Worsened | Changed node | Newly exposed | New support | Lost support | Pool-only mean delta | New/restructured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 100 | True | 59 | 0.0217 | 0.0000 | 20 | 37 | 2 | 22 | 22 | 2 | 0 | 0.0217 | 19 |
| 2025 | 100 | False | 2741 | 0.0044 | 0.0000 | 526 | 2145 | 70 | 596 | 566 | 19 | 0 | 0.0044 | 381 |
| 2025 | 200 | True | 80 | 0.0250 | 0.0000 | 30 | 48 | 2 | 32 | 32 | 5 | 0 | 0.0250 | 28 |
| 2025 | 200 | False | 5520 | 0.0034 | 0.0000 | 880 | 4550 | 90 | 970 | 933 | 29 | 0 | 0.0034 | 772 |
| 2026 | 100 | True | 57 | 0.0165 | 0.0000 | 18 | 35 | 4 | 22 | 22 | 1 | 0 | 0.0165 | 21 |
| 2026 | 100 | False | 2743 | 0.0040 | 0.0000 | 494 | 2176 | 73 | 567 | 537 | 23 | 0 | 0.0040 | 379 |
| 2026 | 200 | True | 81 | 0.0208 | 0.0000 | 29 | 48 | 4 | 33 | 33 | 4 | 0 | 0.0208 | 29 |
| 2026 | 200 | False | 5519 | 0.0035 | 0.0000 | 868 | 4549 | 102 | 970 | 933 | 36 | 0 | 0.0035 | 771 |



True/false groups mean canonical gold / absent from the answer key; the latter remain unjudged. The preceding table includes matched positive components only. Components are matched by kind and exact original span/text. New qualifiers and restructured conjunctions have null old support and delta, rather than invented zero baselines; removed old components are separately exported. Improved/same/worsened uses a 1e-7 numerical tolerance. “Newly exposed” means the selected normalized text was absent from the previous small bundle. Newly supported uses the fixed .5 threshold, not adjudicated entailment. Exclusions are tabulated separately: a larger exclusion value is a stronger violation proxy, not a support gain.

| Year | Gold | Matched exclusions (K100) | Mean violation delta | Larger violation | Smaller violation |
| --- | --- | --- | --- | --- | --- |
| 2025 | True | 0 | — | 0 | 0 |
| 2025 | False | 100 | 0.0015 | 17 | 5 |
| 2026 | True | 0 | — | 0 | 0 |
| 2026 | False | 100 | -0.0002 | 13 | 5 |



| Year | K | Gold | Pool min/Q1/median/Q3/max | rho(pool,max cosine) | rho(degree,score) | rho(items,score) | rho(attributable,score) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 100 | True | {'min': 3.0, 'q25': 15.5, 'median': 24.0, 'q75': 38.0, 'max': 53.0} | 0.0038 | -0.2728 | 0.0010 | -0.0345 |
| 2025 | 100 | False | {'min': 1.0, 'q25': 8.0, 'median': 15.0, 'q75': 28.0, 'max': 53.0} | 0.1596 | 0.0503 | 0.3441 | 0.3193 |
| 2025 | 200 | True | {'min': 2.0, 'q25': 12.5, 'median': 23.0, 'q75': 29.5, 'max': 53.0} | 0.0181 | -0.0683 | 0.1919 | 0.1924 |
| 2025 | 200 | False | {'min': 1.0, 'q25': 5.0, 'median': 11.0, 'q75': 20.0, 'max': 53.0} | 0.1373 | 0.1460 | 0.4588 | 0.4546 |
| 2026 | 100 | True | {'min': 3.0, 'q25': 17.75, 'median': 25.0, 'q75': 37.75, 'max': 53.0} | -0.0337 | -0.3268 | -0.0588 | -0.1168 |
| 2026 | 100 | False | {'min': 1.0, 'q25': 9.0, 'median': 15.0, 'q75': 28.0, 'max': 53.0} | 0.1445 | 0.0091 | 0.3309 | 0.2964 |
| 2026 | 200 | True | {'min': 3.0, 'q25': 13.0, 'median': 23.0, 'q75': 28.0, 'max': 53.0} | 0.0203 | -0.0467 | 0.2175 | 0.2144 |
| 2026 | 200 | False | {'min': 1.0, 'q25': 6.0, 'median': 12.0, 'q75': 21.0, 'max': 53.0} | 0.1447 | 0.1161 | 0.4324 | 0.4077 |



Correlations pool candidate/question observations within a cutoff and K; max-cosine correlation uses component observations. Repeated entities/components are dependent, so these are descriptive Spearman coefficients without significance claims. Correlation does not identify a causal size effect. Full distributions are retained. No pool-size penalty was added after viewing outcomes.

| Year | Diagnostic (K100) | Top10 | Canonical | Median |
| --- | --- | --- | --- | --- |
| 2025 | pool_only | 13 | 47 | 28.0000 |
| 2025 | without_qualifiers | 13 | 47 | 28.0000 |
| 2026 | pool_only | 13 | 50 | 26.0000 |
| 2026 | without_qualifiers | 13 | 50 | 26.0000 |



## Movement and retention

| Year | Stage | H2 bin | N | Improved | Same | Worsened | Entered10 | Left10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | CR6-100 | 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR6-100 | 11-50 | 16 | 14 | 0 | 2 | 6 | 0 |
| 2025 | CR6-100 | 51-100 | 12 | 9 | 0 | 3 | 4 | 0 |
| 2025 | CR6-100 | >100 | 16 | 0 | 16 | 0 | 0 | 0 |
| 2025 | CR7-100 | 1-10 | 3 | 1 | 2 | 0 | 0 | 0 |
| 2025 | CR7-100 | 11-50 | 16 | 15 | 0 | 1 | 6 | 0 |
| 2025 | CR7-100 | 51-100 | 12 | 8 | 1 | 3 | 1 | 0 |
| 2025 | CR7-100 | >100 | 16 | 0 | 16 | 0 | 0 | 0 |
| 2025 | CR6-200 | 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2025 | CR6-200 | 11-50 | 16 | 13 | 1 | 2 | 6 | 0 |
| 2025 | CR6-200 | 51-100 | 12 | 8 | 0 | 4 | 4 | 0 |
| 2025 | CR6-200 | >100 | 16 | 10 | 4 | 2 | 0 | 0 |
| 2026 | CR6-100 | 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR6-100 | 11-50 | 15 | 15 | 0 | 0 | 6 | 0 |
| 2026 | CR6-100 | 51-100 | 14 | 10 | 0 | 4 | 5 | 0 |
| 2026 | CR6-100 | >100 | 18 | 0 | 18 | 0 | 0 | 0 |
| 2026 | CR7-100 | 1-10 | 3 | 2 | 1 | 0 | 0 | 0 |
| 2026 | CR7-100 | 11-50 | 15 | 15 | 0 | 0 | 7 | 0 |
| 2026 | CR7-100 | 51-100 | 14 | 9 | 0 | 5 | 0 | 0 |
| 2026 | CR7-100 | >100 | 18 | 0 | 18 | 0 | 0 | 0 |
| 2026 | CR6-200 | 1-10 | 3 | 0 | 2 | 1 | 0 | 1 |
| 2026 | CR6-200 | 11-50 | 15 | 12 | 0 | 3 | 5 | 0 |
| 2026 | CR6-200 | 51-100 | 14 | 9 | 0 | 5 | 5 | 0 |
| 2026 | CR6-200 | >100 | 18 | 10 | 5 | 3 | 0 | 0 |



| Year | Stage | Reference | Prior hits | Preserved | Lost |
| --- | --- | --- | --- | --- | --- |
| 2025 | CR6-100 | H2 | 3 | 2 | 1 |
| 2025 | CR6-100 | CR0-100 | 6 | 4 | 2 |
| 2025 | CR6-100 | CR1-100 | 11 | 11 | 0 |
| 2025 | CR7-100 | H2 | 3 | 3 | 0 |
| 2025 | CR7-100 | CR0-100 | 6 | 5 | 1 |
| 2025 | CR7-100 | CR1-100 | 11 | 8 | 3 |
| 2025 | CR6-200 | H2 | 3 | 2 | 1 |
| 2025 | CR6-200 | CR0-100 | 6 | 4 | 2 |
| 2025 | CR6-200 | CR1-100 | 11 | 11 | 0 |
| 2026 | CR6-100 | H2 | 3 | 2 | 1 |
| 2026 | CR6-100 | CR0-100 | 5 | 3 | 2 |
| 2026 | CR6-100 | CR1-100 | 11 | 11 | 0 |
| 2026 | CR7-100 | H2 | 3 | 3 | 0 |
| 2026 | CR7-100 | CR0-100 | 5 | 5 | 0 |
| 2026 | CR7-100 | CR1-100 | 11 | 6 | 5 |
| 2026 | CR6-200 | H2 | 3 | 2 | 1 |
| 2026 | CR6-200 | CR0-100 | 5 | 2 | 3 |
| 2026 | CR6-200 | CR1-100 | 11 | 11 | 0 |



Reference hit sets overlap and must not be added. CR7 uses the unchanged RRF60 and average ranks for tied scores. It can preserve prior hits while also suppressing standalone gains.

| Year | Stage | Failure counts |
| --- | --- | --- |
| 2025 | CR6-100 | {'NOT_IN_H2_100': 16, 'NO_ATTRIBUTABLE_EVIDENCE': 5, 'SUPPORTED_BUT_RANKED_BELOW_10': 11, 'RETRIEVED_TOP10': 12, 'PARTIAL_REQUIREMENT_SUPPORT': 1, 'ANSWER_QUALIFIER_FAILURE': 1, 'REQUIREMENT_EVIDENCE_NOT_FOUND': 1} |
| 2025 | CR7-100 | {'NOT_IN_H2_100': 16, 'NO_ATTRIBUTABLE_EVIDENCE': 5, 'SUPPORTED_BUT_RANKED_BELOW_10': 14, 'RETRIEVED_TOP10': 10, 'PARTIAL_REQUIREMENT_SUPPORT': 1, 'ANSWER_QUALIFIER_FAILURE': 1} |
| 2025 | CR6-200 | {'SUPPORTED_BUT_RANKED_BELOW_10': 20, 'NO_ATTRIBUTABLE_EVIDENCE': 8, 'RETRIEVED_TOP10': 12, 'PARTIAL_REQUIREMENT_SUPPORT': 1, 'NOT_IN_H2_200': 4, 'ANSWER_QUALIFIER_FAILURE': 1, 'REQUIREMENT_EVIDENCE_NOT_FOUND': 1} |
| 2026 | CR6-100 | {'NOT_IN_H2_100': 18, 'NO_ATTRIBUTABLE_EVIDENCE': 5, 'SUPPORTED_BUT_RANKED_BELOW_10': 12, 'RETRIEVED_TOP10': 13, 'PARTIAL_REQUIREMENT_SUPPORT': 1, 'ANSWER_QUALIFIER_FAILURE': 1} |
| 2026 | CR7-100 | {'NOT_IN_H2_100': 18, 'NO_ATTRIBUTABLE_EVIDENCE': 5, 'SUPPORTED_BUT_RANKED_BELOW_10': 16, 'RETRIEVED_TOP10': 10, 'PARTIAL_REQUIREMENT_SUPPORT': 1} |
| 2026 | CR6-200 | {'SUPPORTED_BUT_RANKED_BELOW_10': 22, 'NO_ATTRIBUTABLE_EVIDENCE': 9, 'RETRIEVED_TOP10': 12, 'NOT_IN_H2_200': 5, 'PARTIAL_REQUIREMENT_SUPPORT': 1, 'ANSWER_QUALIFIER_FAILURE': 1} |



Failure precedence is pool absence, top-ten recovery, empty pool, no direct/name/winner attribution, weak qualifier, zero positive coverage, partial coverage, then supported-but-below-ten. NO_ATTRIBUTABLE_EVIDENCE means no strong direct/name/winner item; uncertain or mention-only evidence may still score. REQUIREMENT_EVIDENCE_NOT_FOUND and ANSWER_QUALIFIER_FAILURE mean low selected cosine support, not proven graph absence or scientific invalidity. DECOMPOSITION_FAILURE and ATTRIBUTION_FAILURE require qualitative inspection; no automated counts are fabricated from answer membership.

## Actual graph evidence and qualifier diagnostics

### Required Q1/MACE diagnostic

Which methods (by Feb 2026) are best suited for predicting interatomic potentials when balancing accuracy close to DFT with computational efficiency for systems of millions of atoms?

Ranks: H2=114, CR0-100=114, CR1-100=114, CR3-100=114, CR6-100=114, CR7-100=114, CR6-200=38.

Pool size 23 versus old bundle 6; CR6 coverage 1.0000; score 0.3687.

| Kind | Component | Old node | Old support | New node | New support | Attribution | New text | Selected evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | predicting interatomic potentials | clai_00604 | 0.5659 | clai_00602 | 0.6078 | DIRECT_SUBJECT_SUPPORT | True | MACE evaluates roughly ten times faster than prior force‑field models for invariant (L=0) architectures and 4–5 times faster for equivariant (L=1,2) architectures, while delivering lower prediction errors. |
| requirement | accuracy close to DFT | clai_00604 | 0.5499 | clai_00267 | 0.5974 | DIRECT_SUBJECT_SUPPORT | True | MACE and grACE packages provide excellent scaling with number of elements while achieving high accuracy. |
| requirement | computational efficiency | clai_00604 | 0.4946 | clai_00602 | 0.6522 | DIRECT_SUBJECT_SUPPORT | True | MACE evaluates roughly ten times faster than prior force‑field models for invariant (L=0) architectures and 4–5 times faster for equivariant (L=1,2) architectures, while delivering lower prediction errors. |
| requirement | systems of millions of atoms | clai_00604 | 0.4936 | clai_00267 | 0.5716 | DIRECT_SUBJECT_SUPPORT | True | MACE and grACE packages provide excellent scaling with number of elements while achieving high accuracy. |



Top three matches and source-link provenance are available in requirement_evidence_selection.json and candidate_evidence_pools.json. These are actual score inputs, not proof that the text satisfies the scientific condition.

Compare the efficiency and scale rows directly with their selected texts. A high similarity to an accuracy statement does not establish efficiency or million-atom scalability; only explicitly stated conditions count as textual evidence in a qualitative reading. Absence from the bounded eligible pool does not prove absence from the literature.

### Previously hidden evidence with improved gold rank: Q14/OMat24

What datasets and benchmarks are commonly used (by Feb 2026) to evaluate MLIPs and property prediction models for materials discovery?

Ranks: H2=71, CR0-100=90, CR1-100=82, CR3-100=83, CR6-100=28, CR7-100=58, CR6-200=47.

Pool size 9 versus old bundle 5; CR6 coverage 1.0000; score 0.2502.

| Kind | Component | Old node | Old support | New node | New support | Attribution | New text | Selected evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | evaluate MLIPs and property prediction models for materials discovery | clai_00107 | 0.1809 | clai_00106 | 0.5002 | EXPLICIT_NAME_SUPPORT | True | Molecular‑dynamics simulations using MatPES‑trained UMLIPs exhibit markedly higher stability, with median termination temperatures up to ~1,900 K, versus ~1,300–1,500 K for MPRelax and OMat24 models. |



Top three matches and source-link provenance are available in requirement_evidence_selection.json and candidate_evidence_pools.json. These are actual score inputs, not proof that the text satisfies the scientific condition.

### Different components select different evidence: Q1/EquiformerV2

Which methods (by Feb 2026) are best suited for predicting interatomic potentials when balancing accuracy close to DFT with computational efficiency for systems of millions of atoms?

Ranks: H2=42, CR0-100=75, CR1-100=23, CR3-100=24, CR6-100=24, CR7-100=26, CR6-200=35.

Pool size 24 versus old bundle 10; CR6 coverage 1.0000; score 0.3781.

| Kind | Component | Old node | Old support | New node | New support | Attribution | New text | Selected evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | predicting interatomic potentials | clai_00592 | 0.6193 | clai_00592 | 0.6193 | DIRECT_SUBJECT_SUPPORT | False | EquiformerV2 incorporates three architectural improvements: attention re-normalization, separable S2 activation, and separable layer normalization, replacing SO(3) convolutions with eSCN convolutions to efficiently incorporate higher-degree tensors. |
| requirement | accuracy close to DFT | clai_00590 | 0.6259 | clai_00589 | 0.6272 | DIRECT_SUBJECT_SUPPORT | True | EquiformerV2 trained on only OC22 (8.4M structures) achieves better results than GemNet-OC trained on both OC20 (130M structures) and OC22, demonstrating significantly better data efficiency. |
| requirement | computational efficiency | clai_00593 | 0.6252 | clai_00589 | 0.6498 | DIRECT_SUBJECT_SUPPORT | True | EquiformerV2 trained on only OC22 (8.4M structures) achieves better results than GemNet-OC trained on both OC20 (130M structures) and OC22, demonstrating significantly better data efficiency. |
| requirement | systems of millions of atoms | clai_00593 | 0.5506 | clai_00593 | 0.5506 | DIRECT_SUBJECT_SUPPORT | False | EquiformerV2 achieves the highest success rate in the AdsorbML algorithm and enables a 2x reduction in DFT calculations needed for computing adsorption energies. |



Top three matches and source-link provenance are available in requirement_evidence_selection.json and candidate_evidence_pools.json. These are actual score inputs, not proof that the text satisfies the scientific condition.

### Expanded-pool boost for an unjudged candidate: Q11/RMSE

Which methods (by Feb 2026) are best suited for integrating features from three or more distinct scales simultaneously within a unified framework?

Ranks: H2=11, CR0-100=39, CR1-100=74, CR3-100=34, CR6-100=33, CR7-100=17, CR6-200=50.

Pool size 7 versus old bundle 6; CR6 coverage 0.0000; score 0.1661.

| Kind | Component | Old node | Old support | New node | New support | Attribution | New text | Selected evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | integrating features from three or more distinct scales simultaneously within a unified framework | clai_00531 | 0.1679 | clai_00603 | 0.4181 | EXPLICIT_NAME_SUPPORT | True | MACE shows superior extrapolation ability, reducing energy RMSE by about 30 % at 1200 K and 40 % on dihedral‑slice energies for the 3BPA benchmark, and achieving lower errors on geometries far from the training distribution. |



Top three matches and source-link provenance are available in requirement_evidence_selection.json and candidate_evidence_pools.json. These are actual score inputs, not proof that the text satisfies the scientific condition.

This candidate is absent from the supplied answer key, so the boost is a noise warning to inspect, not a certified false positive.

### Remaining weak component support: Q1/CHGNet

Which methods (by Feb 2026) are best suited for predicting interatomic potentials when balancing accuracy close to DFT with computational efficiency for systems of millions of atoms?

Ranks: H2=88, CR0-100=100, CR1-100=97, CR3-100=95, CR6-100=97, CR7-100=95, CR6-200=169.

Pool size 20 versus old bundle 5; CR6 coverage 0.0000; score 0.0267.

| Kind | Component | Old node | Old support | New node | New support | Attribution | New text | Selected evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| core_task | predicting interatomic potentials | comp_00483 | 0.1639 | comp_00483 | 0.1639 | UNCERTAIN | False | MLIP-NEB relaxation framework |
| requirement | accuracy close to DFT | comp_00483 | 0.1448 | comp_00488 | 0.1644 | UNCERTAIN | True | Barrier classification threshold |
| requirement | computational efficiency | comp_00483 | 0.1725 | comp_00483 | 0.1725 | UNCERTAIN | False | MLIP-NEB relaxation framework |
| requirement | systems of millions of atoms | comp_00483 | 0.1441 | comp_00482 | 0.1510 | UNCERTAIN | True | M3GNet |



Top three matches and source-link provenance are available in requirement_evidence_selection.json and candidate_evidence_pools.json. These are actual score inputs, not proof that the text satisfies the scientific condition.

### Q12 answer-head qualifier control

| Year | Gold | Candidates | Mean without qualifiers | Mean CR6 | Top10 without qualifiers | Top10 CR6 |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | True | 7 | 0.4979 | 0.4466 | 5 | 4 |
| 2025 | False | 93 | 0.1626 | 0.1479 | 5 | 6 |
| 2026 | True | 8 | 0.4883 | 0.4448 | 4 | 4 |
| 2026 | False | 92 | 0.1740 | 0.1592 | 6 | 6 |



| Candidate | Gold | Without-qualifier rank | CR6 rank | Qualifier supports |
| --- | --- | --- | --- | --- |
| MLIP-NEB benchmarking workflow | False | 77 | 57 | [('ML-based', 0.181), ('interatomic potential', 0.1933)] |
| MLIP_framework_analysis | False | 71 | 52 | [('ML-based', 0.1735), ('interatomic potential', 0.2078)] |
| CHGNet: Pretrained universal neural network potential for charge-informed atomistic modeling | False | 53 | 70 | [('ML-based', 0.1472), ('interatomic potential', 0.1692)] |
| Atomic Cluster Expansion extended to scalar, vectorial, tensorial properties with magnetic moments and charge transfer | False | 72 | 55 | [('ML-based', 0.165), ('interatomic potential', 0.2104)] |
| Neural Equivariant Interatomic Potentials | False | 64 | 49 | [('ML-based', 0.1684), ('interatomic potential', 0.2176)] |
| Machine Learning Interatomic Potentials: A Review and Framework for Understanding MLIPs | False | 79 | 67 | [('ML-based', 0.1735), ('interatomic potential', 0.1868)] |



This paired control isolates the effect of including the extracted modifiers on this fixed pool/scorer. Benchmark membership is not a scientific label for semantic class; count/rank changes alone do not certify better class discrimination. Adding a component to a geometric mean can either raise or lower a score depending on its support, so qualifiers are not guaranteed penalties.

| H2 hit rescued by CR7 | Q | H2 | CR6 | CR7 |
| --- | --- | --- | --- | --- |
| Matbench Discovery | Q14 | 8 | 29 | 8 |



## Every canonical target

| Year | Q | Target | R3 | H2 | CR0 | CR1 | CR3 | CR6-100 | CR7-100 | CR6-200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | Q1 | ACE | 269 | 154 | 154 | 154 | 154 | 154 | 154 | 36 |
| 2025 | Q1 | CHGNet | 119 | 88 | 100 | 97 | 95 | 97 | 95 | 169 |
| 2025 | Q1 | EquiformerV2 | 23 | 42 | 75 | 23 | 24 | 24 | 26 | 35 |
| 2025 | Q1 | GAP | 206 | 47 | 93 | 11 | 15 | 16 | 21 | 21 |
| 2025 | Q1 | M3GNet | 101 | 26 | 22 | 14 | 11 | 14 | 11 | 17 |
| 2025 | Q1 | MACE | 333 | 114 | 114 | 114 | 114 | 114 | 114 | 38 |
| 2025 | Q1 | NequIP | 188 | 91 | 37 | 17 | 53 | 17 | 51 | 22 |
| 2025 | Q2 | D4FT | 10 | 12 | 2 | 11 | 4 | 7 | 3 | 10 |
| 2025 | Q2 | DeepH | 260 | 42 | 74 | 1 | 7 | 1 | 6 | 1 |
| 2025 | Q2 | HamGNN | 90 | 24 | 91 | 55 | 44 | 51 | 40 | 87 |
| 2025 | Q3 | DeepH-E3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2025 | Q4 | VGNN | 2292 | 65 | 39 | 1 | 17 | 1 | 16 | 1 |
| 2025 | Q6 | LiFlow | 8 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| 2025 | Q7 | symbolic regression | 159 | 186 | 186 | 186 | 186 | 186 | 186 | 27 |
| 2025 | Q8 | ConvLSTM | 138 | 25 | 28 | 1 | 1 | 1 | 1 | 1 |
| 2025 | Q9 | E(3)-equivariant GNNs | 2266 | 119 | 119 | 119 | 119 | 119 | 119 | 145 |
| 2025 | Q9 | NequIP | 196 | 111 | 111 | 111 | 111 | 111 | 111 | 32 |
| 2025 | Q10 | SchNet | 28 | 23 | 16 | 29 | 20 | 24 | 15 | 32 |
| 2025 | Q12 | CHGNet | 107 | 73 | 97 | 92 | 89 | 92 | 87 | 160 |
| 2025 | Q12 | DeePMD | 229 | 111 | 111 | 111 | 111 | 111 | 111 | 14 |
| 2025 | Q12 | EquiformerV2 | 24 | 32 | 82 | 22 | 16 | 21 | 12 | 32 |
| 2025 | Q12 | GAP | 215 | 50 | 89 | 23 | 33 | 16 | 23 | 21 |
| 2025 | Q12 | M3GNet | 97 | 29 | 8 | 7 | 7 | 7 | 8 | 8 |
| 2025 | Q12 | MACE | 334 | 104 | 104 | 104 | 104 | 104 | 104 | 45 |
| 2025 | Q12 | MatterSim | 130 | 35 | 68 | 2 | 4 | 1 | 4 | 1 |
| 2025 | Q12 | NequIP | 216 | 87 | 45 | 3 | 18 | 5 | 24 | 6 |
| 2025 | Q12 | Orb | 398 | 130 | 130 | 130 | 130 | 130 | 130 | 187 |
| 2025 | Q12 | SevenNet | 175 | 41 | 51 | 5 | 9 | 2 | 9 | 2 |
| 2025 | Q12 | eSEN | 2320 | 283 | 283 | 283 | 283 | 283 | 283 | 283 |
| 2025 | Q13 | CGCNN | 160 | 107 | 107 | 107 | 107 | 107 | 107 | 20 |
| 2025 | Q13 | CHGNet | 136 | 100 | 99 | 94 | 100 | 94 | 100 | 151 |
| 2025 | Q13 | EquiformerV2 | 24 | 57 | 79 | 29 | 53 | 20 | 46 | 31 |
| 2025 | Q13 | M3GNet | 111 | 55 | 23 | 1 | 5 | 3 | 7 | 3 |
| 2025 | Q13 | MACE | 253 | 106 | 106 | 106 | 106 | 106 | 106 | 44 |
| 2025 | Q13 | MEGNet | 185 | 185 | 185 | 185 | 185 | 185 | 185 | 180 |
| 2025 | Q13 | NequIP | 186 | 72 | 53 | 11 | 40 | 12 | 40 | 18 |
| 2025 | Q13 | SchNet | 34 | 36 | 8 | 67 | 63 | 17 | 19 | 27 |
| 2025 | Q14 | AFLOW | 260 | 66 | 94 | 5 | 17 | 5 | 18 | 7 |
| 2025 | Q14 | Alexandria | 3381 | 353 | 353 | 353 | 353 | 353 | 353 | 353 |
| 2025 | Q14 | GNoME | 126 | 41 | 30 | 24 | 28 | 12 | 14 | 22 |
| 2025 | Q14 | JARVIS-DFT | 547 | 541 | 541 | 541 | 541 | 541 | 541 | 541 |
| 2025 | Q14 | Matbench | 97 | 53 | 88 | 67 | 73 | 68 | 75 | 101 |
| 2025 | Q14 | Matbench Discovery | 2 | 8 | 1 | 28 | 8 | 29 | 8 | 48 |
| 2025 | Q14 | Materials Project | 106 | 35 | 51 | 15 | 15 | 17 | 17 | 27 |
| 2025 | Q14 | OMat24 | 475 | 71 | 90 | 82 | 83 | 28 | 58 | 47 |
| 2025 | Q14 | OQMD | 118 | 101 | 101 | 101 | 101 | 101 | 101 | 32 |
| 2025 | Q14 | WBM | 674 | 726 | 726 | 726 | 726 | 726 | 726 | 726 |
| 2026 | Q1 | ACE | 320 | 170 | 170 | 170 | 170 | 170 | 170 | 37 |
| 2026 | Q1 | CHGNet | 135 | 96 | 99 | 99 | 99 | 99 | 99 | 174 |
| 2026 | Q1 | EquiformerV2 | 27 | 32 | 90 | 25 | 19 | 25 | 20 | 36 |
| 2026 | Q1 | GAP | 241 | 57 | 55 | 15 | 25 | 20 | 32 | 24 |
| 2026 | Q1 | M3GNet | 114 | 29 | 19 | 17 | 13 | 17 | 13 | 19 |
| 2026 | Q1 | MACE | 439 | 128 | 128 | 128 | 128 | 128 | 128 | 39 |
| 2026 | Q1 | NequIP | 219 | 103 | 103 | 103 | 103 | 103 | 103 | 22 |
| 2026 | Q2 | D4FT | 12 | 14 | 3 | 11 | 4 | 9 | 3 | 12 |
| 2026 | Q2 | DeepH | 309 | 73 | 96 | 1 | 18 | 1 | 18 | 1 |
| 2026 | Q2 | HamGNN | 102 | 64 | 91 | 58 | 71 | 54 | 68 | 90 |
| 2026 | Q3 | DeepH-E3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 2026 | Q4 | VGNN | 2567 | 59 | 44 | 1 | 16 | 1 | 17 | 1 |
| 2026 | Q6 | LiFlow | 9 | 2 | 1 | 2 | 1 | 2 | 1 | 2 |
| 2026 | Q7 | symbolic regression | 184 | 225 | 225 | 225 | 225 | 225 | 225 | 225 |
| 2026 | Q8 | ConvLSTM | 158 | 29 | 30 | 1 | 1 | 1 | 1 | 1 |
| 2026 | Q9 | E(3)-equivariant GNNs | 2391 | 139 | 139 | 139 | 139 | 139 | 139 | 149 |
| 2026 | Q9 | NequIP | 232 | 111 | 111 | 111 | 111 | 111 | 111 | 34 |
| 2026 | Q10 | SchNet | 32 | 25 | 18 | 26 | 21 | 23 | 19 | 30 |
| 2026 | Q12 | CHGNet | 121 | 73 | 98 | 95 | 90 | 96 | 90 | 170 |
| 2026 | Q12 | DeePMD | 265 | 56 | 34 | 17 | 28 | 14 | 24 | 16 |
| 2026 | Q12 | EquiformerV2 | 28 | 32 | 93 | 24 | 20 | 23 | 14 | 30 |
| 2026 | Q12 | GAP | 250 | 50 | 68 | 25 | 41 | 21 | 31 | 26 |
| 2026 | Q12 | M3GNet | 109 | 31 | 11 | 10 | 8 | 10 | 9 | 10 |
| 2026 | Q12 | MACE | 448 | 110 | 110 | 110 | 110 | 110 | 110 | 44 |
| 2026 | Q12 | MatterSim | 148 | 37 | 74 | 2 | 4 | 1 | 3 | 1 |
| 2026 | Q12 | NequIP | 251 | 92 | 50 | 6 | 25 | 8 | 33 | 8 |
| 2026 | Q12 | Orb | 446 | 112 | 112 | 112 | 112 | 112 | 112 | 191 |
| 2026 | Q12 | PET | 404 | 233 | 233 | 233 | 233 | 233 | 233 | 233 |
| 2026 | Q12 | SevenNet | 200 | 42 | 55 | 8 | 12 | 4 | 8 | 4 |
| 2026 | Q12 | UMA | 2564 | 402 | 402 | 402 | 402 | 402 | 402 | 402 |
| 2026 | Q12 | eSEN | 266 | 116 | 116 | 116 | 116 | 116 | 116 | 42 |
| 2026 | Q13 | CGCNN | 179 | 130 | 130 | 130 | 130 | 130 | 130 | 20 |
| 2026 | Q13 | CHGNet | 150 | 86 | 99 | 96 | 96 | 96 | 95 | 161 |
| 2026 | Q13 | EquiformerV2 | 25 | 49 | 80 | 29 | 48 | 21 | 39 | 30 |
| 2026 | Q13 | M3GNet | 124 | 87 | 26 | 1 | 14 | 3 | 23 | 3 |
| 2026 | Q13 | MACE | 386 | 105 | 105 | 105 | 105 | 105 | 105 | 42 |
| 2026 | Q13 | MEGNet | 206 | 205 | 205 | 205 | 205 | 205 | 205 | 205 |
| 2026 | Q13 | NequIP | 207 | 79 | 54 | 12 | 42 | 13 | 44 | 18 |
| 2026 | Q13 | SchNet | 38 | 25 | 5 | 69 | 53 | 18 | 8 | 26 |
| 2026 | Q14 | AFLOW | 301 | 56 | 95 | 7 | 14 | 8 | 17 | 10 |
| 2026 | Q14 | Alexandria | 467 | 147 | 147 | 147 | 147 | 147 | 147 | 186 |
| 2026 | Q14 | GNoME | 143 | 43 | 32 | 28 | 34 | 15 | 21 | 23 |
| 2026 | Q14 | JARVIS-DFT | 611 | 620 | 620 | 620 | 620 | 620 | 620 | 620 |
| 2026 | Q14 | MPtrj | 154 | 157 | 157 | 157 | 157 | 157 | 157 | 154 |
| 2026 | Q14 | Matbench | 110 | 58 | 90 | 74 | 82 | 74 | 82 | 109 |
| 2026 | Q14 | Matbench Discovery | 2 | 4 | 3 | 25 | 1 | 27 | 1 | 42 |
| 2026 | Q14 | Materials Project | 117 | 34 | 55 | 17 | 13 | 19 | 18 | 27 |
| 2026 | Q14 | OMat24 | 124 | 41 | 53 | 24 | 27 | 7 | 9 | 9 |
| 2026 | Q14 | OQMD | 131 | 96 | 73 | 20 | 61 | 23 | 63 | 35 |
| 2026 | Q14 | WBM | 185 | 183 | 183 | 183 | 183 | 183 | 183 | 32 |



## Decisions

2025: CR6 versus CR1 changes micro Recall@10 by 0.0213 and median gold rank by -5.0000. CR7 recovers 10 canonical instances at ten. The oracle and retention tables quantify the remaining gap and losses; neither fusion nor K200 is presumed better.

2026: CR6 versus CR1 changes micro Recall@10 by 0.0400 and median gold rank by -4.5000. CR7 recovers 10 canonical instances at ten. The oracle and retention tables quantify the remaining gap and losses; neither fusion nor K200 is presumed better.

1. Recall@10: CR1 11/47 → CR6 12/47. The pure pool-access control recovers 13/47; it must be considered alongside the combined change.
2. Median gold rank: CR1 29.0000 → CR6 24.0000 (lower is better).
3. Matched positive components selecting a different node: 22/59 gold, 596/2741 unjudged. Newly exposed selected texts: 22 and 566 respectively.
4. Support increases: gold 20/59 = 0.3390; unjudged 526/2741 = 0.1919. Mean deltas are 0.0217 and 0.0044; these are model-support changes, not correctness annotations.
5. Q12 qualifiers: the paired control recovers 5 gold candidates at ten without qualifiers and 4 with qualifiers. Semantic class discrimination remains qualitatively unvalidated even if benchmark ranks improve.
6. Pool-size/score Spearman: gold 0.0010, unjudged 0.3441; degree/score -0.2728 and 0.0503. Full max-cosine and per-question diagnostics are exported; no corrective penalty was fitted.
7. CR6 moves 10 canonical targets from H2 ranks 11–100 into the top ten.
8. CR6 loses prior top-ten targets: {'H2': 1, 'CR0-100': 2, 'CR1-100': 0}. These reference sets overlap.
9. CR7 recovers 10/47 versus CR6 12/47; consult the separate preservation table for the retention tradeoff.
10. The primary oracle remains 31/47. Gap before: 20/47 = 0.4255; after CR6: 19/47 = 0.4043.
11. Failure categories describe operational missing candidates, attribution and support; they cannot determine which scientific mechanism is the dominant cause without independent evidence annotations.
12. Do not proceed to hierarchy approximation on this evidence alone; validate quality and attribution on unseen questions first.

Selection counts establish whether hidden candidate evidence was accessed; matched-component support statistics compare gold with unjudged candidates. Pool-only controls distinguish access from grammar changes, and the Q12 paired control isolates qualifier participation. Pool-size correlations are descriptive; where positive, they warn of max-pooling opportunity bias without establishing that size caused gains. Grammar changes were motivated by development-set failure traces and the user’s specification, then corrected using unrelated synthetic cases; they were not tuned against benchmark scores. No weights, caps or thresholds were selected by benchmark score. No informative misalignment null was run, which limits causal interpretation.

Remaining failures mix unavailable candidates, weak selected attribution/support and supported-but-low ranks. Operational categories cannot causally apportion missing graph facts, decomposition errors, attribution errors and semantic ranking. Cosine still cannot verify numerical scale, comparative direction beyond shallow patterns, conjunction truth or scientific applicability. Fuller evidence does not make those limitations disappear.

Author judgement: hierarchy approximation remains premature. Recovery and retention are assessed on reused development questions, with an oracle gap and no independent validation of evidence sufficiency. Small changes in recovered targets do not establish generalization or statistical significance. Keep this pass as a quality experiment and test the mechanisms on unseen questions before optimizing retrieval speed. No new neural reranker, H2 discovery change, hierarchy, parameter sweep or post-hoc size correction was added.

## Reproduction and verification

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv\Scripts\python.exe -m tkh_abstraction.cli evaluate-conditioned --config configs/conditioned.yaml
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/evaluation/conditioned_evidence_tests.xml
.\.venv\Scripts\python.exe scripts/report_conditioned.py
.\.venv\Scripts\python.exe scripts/package_submission.py
```

Full suite: 224 passed, no failures/errors/skips. Tests cover generic qualifiers and conjunctions, participial versus list attribution, wider deterministic deduplicated pools/provenance, component-specific selection, temporal safety, frozen aggregation, exclusions, zero baseline/path weights, exact prefixes and tails, deterministic fusion, no answer-specific rules, and ground-truth independence. The report audit verifies all old source/artifact/metric hashes, reference rankings/metrics, oracle dictionaries, all 14 statuses, source spans, temporal pool containment and score decomposition.

For unseen questions, use `--questions` and `--ground-truth` with separate output paths. An isolated run without copied baseline artifacts requires `conditioned_verify_history: false`; the full assessment report writer expects the supplied two-cutoff benchmark. Historical CR2/CR4 are never executed. Do not run old report writers against the extended workspace. Pools are cached in memory per snapshot and exported; existing embedding caches persist between runs.

Implementation: conditioned_decomposition.py, conditioned_evidence.py, conditioned_pipeline.py, additive CLI command, configs/conditioned.yaml, tests/test_conditioned.py and scripts/report_conditioned.py. Required machine-readable artifacts and test XML are alongside this report; protocol/checkpoints/audit are under conditioned/. Historical constraints metrics/report are archived under artifacts/baseline_constraints.

Ultralight local adaptation completed serial read-only Claude Opus 5 planning and Claude Sonnet 5 review through the official first-party subscription helper. The review prompted conservative generic attribution/conjunction safeguards and stronger pipeline tests; advice was verified rather than adopted blindly. Prompts, results and decisions are in notes/conditioned. Codex implemented, integrated and executed the work. No human scientific adjudication is claimed.
