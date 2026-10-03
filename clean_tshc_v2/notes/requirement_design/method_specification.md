# V2 method specification for report authorship

Present current V2 as a coherent entity-centric requirement-aware design. Do not read prior reports/results or describe a V3. All numbers must come only from the current frozen prediction/evaluation handoff. No invented citations or empirical performance claims.

## Data and identity

Use the current graph snapshot and question CSV. Original question is always preserved. Temporal visibility is publication-year only, with graph first_seen_year <= cutoff and edge/asserting-article checks. last_seen_year is not expiry; undated aliases are suppressed when last_seen_year exceeds an earlier cutoff. Month constraints cannot be resolved. Exact NFKC/casefold/whitespace/typographic-hyphen surface equality groups compatible types: method/technique/component/cited_work share approach; dataset/metric/article stay separate. Lowest member node ID defines canonical ID. Only graph aliases, no inferred fuzzy aliases or target-derived corrections.

## Candidate generation remains unchanged

Hard answer-family filtering applies across all entity channels. Existing BGE pinned model and V1 bounded member-node documents; canonical entity score max over members. Original and structured dense views are separate. BM25k1=1.5,b=.75 names/aliases/all indexed distinct evidence, positive overlaps only. Conservative exact bounded names: digits/CamelCase/allcaps or multiword length>=10. Evidence dense original/structured max,500 evidence seeds -> owners. Graph starts top200 evidence seeds, at most2 hops through an article only, degree80/fanout64, score decay .8^(length-1). No unrestricted diffusion/hierarchy beam. Graph paths are structural context, never entailment. Each of six ranking lists caps500 entities; union canonical IDs; RRF k60. Core source modules/entity intent/extractive parser hashes are unchanged in this task. Atomic ranking components do not alter the first-stage query strings.

A0: V1-like raw dense all nonauthor nodes canonicaldedup top500. A1: hardtyped max original+structured dense top500. A2 roundrobin dense original/structured,BM25,name. A3 adds evidence. A4 adds graph. A5 identicalfullunion RRF ordered. A6 genuine relevance CE over500 prefix; unchanged RRF tail. TypedRaw/ParsedOnly, WithoutEvidenceRRF/WithoutGraphRRF and global100/200/500 controls remain. A7 is predeclaredprimary, neverselectedfromresults. NoNLI and SingleEvidence controls isolate tiering and evidencek.

## Atomic requirements and evidence

Extractive parser already identifies contiguous core/requirements/exclusions with offsets; ranking-only splitter additionally separates commas/and/nor in non-core lists, retaining OR clauses intact. Literal source spans preserved. Core task essential; fallback fulloriginal only if no core. No invented facts or target taxonomy. This is heuristic language parsing, not a semantic oracle: phrases about a problem/undesirable condition can be extracted ambiguously; compounds can be split imperfectly.

For every entity in RRFtop500 and every component, max cosine ranks only legitimately attached candidate-specific evidence in its full index. Sort similarity descending, relationpriority (directclaim, explicitmention, typed evidence) and evidenceID ties. Deduplicate normalized text, selectk3. Exclude article fields and graph_evidence fields. Selectk1 independently with same deterministic rule for control. No evidence from unrelated entities. Evidence carries nodeID,text,paths,provenance and similarity.

Global CE uses FULL original question + identity-first compact query-dependent card. Genuine relevance model cross-encoder/ms-marco-MiniLM-L6-v2. Component CE uses 'Candidate <name>: <literal component>' and its identity-first evidence bundle. Modeltokenbudget512; prefixpack keeps identity first, retains whole evidence blocks while they fit, records included/omitted blocks and query truncation. Overlong trial tokenizer warnings are length checks, not overlong model inference. A bundle whose packing includes no evidence gets null componentlogit. Logits are relevance measurements, not probabilities.

Pinned NLI DeBERTav3small is supportproxy only. Premise is identity+up to3 selected evidenceblocks packed jointly with literal candidatehypothesis. One NLI pair per eligible packed bundle, not independent per-sentence aggregation. Hypotheses: '<name> is suitable for <core>.'; '<name> satisfies the condition: <requirement>.'; '<name> involves <excluded property>.' The last asserts a violation: swap entailment/contradiction for desired-status interpretation. Preserve both rawprobabilities and explicitpolarity. threshold .7 and otherprob <.3: SUPPORTED or CONTRADICTED; unclear/noevidence ->NOT_SUPPORTED_BY_AVAILABLE_EVIDENCE; existingevidence but unavailableNLI ->NOT_EVALUABLE. None of these are calibrated scientifictruth. If whole-block tokenpacking drops everypiece, unsupported; selected evidence alone is not proof it entered model.

## Conjunction ranking

Positive set includes core plus requirements. Exclusions influence tiers, not geometricrelevance (similarity to an excluded condition cannot represent negation).

Tier3 any CONTRADICTED component after exclusionpolarity.
Tier0 no contradiction and all positivecomponents SUPPORTED.
Tier1 no contradiction, at least one SUPPORTED positive component, not tier0.
Tier2 no contradiction, no confident support.

For each fixed question/component, scoreable entities get competition ranks:1+number of strictly higher logits. Ties share rank. Missing/unscorablelogit receives neutral (n_scoreable+1)/2, or1 if none scoreable. Thus missingevidence is not an automatic worst score; also can outrank weakobservedevidence. This is a fixed nonoptimized imputation assumption, explicitly a limitation. Requirementlogits across components are not added.

For positivecomponents i: RR_i=1/rank_i; S_req=exp(mean(log(RR_i+epsilon))),epsilon1e-12; S_weak=min(RR_i). Lexicographic tierascending, S_reqdescending,S_weakdescending, globalCEcompetitionrankascending,RRFpositionascending,entityID. No learned/fittedweights. Same500-set; unchangedtail. NoNLI forces tier2 and uses same primarycomponentrelevance, ignoring all NLI outputs; not an independently timed modelrun. SingleEvidence runs both models on k1 bundles independently. A7 metadata counts support/contradiction/unknown; every primaryfinalcandidate has trace; outsideprefix explicitly unscored with nullfeatures and untouchedRRFposition.

## Freeze, auditing, reproducibility

Two separate entrypoint processes predict.py/evaluate.py. Predictor reads graph, questions, fixedconfig, source, models only; no evaluator imports. Strict inputallowlist blocks oldoutputs, archives, reports, notes, targetannotations/mappings/gold/oracles and protectedprojectsource. All projectrelative readattempts logged; forbidden reads raise and no deniedread is allowed at successfulfreeze. Model files are hashed before loading; revisions must matchconfig. Pythonhooks cannot contain hostile nativecode; do not claim OSsecurity.

Prediction writes all rankings/traces and hashes config/source/inputs/modelmanifest/generatedpredictionartifacts beforefreeze+exit. Evaluator verifies every hash before import/goldread; runtimeguard refuses frozenfilewriting/removal/rename/chmod and all nonV2writes. Verify againafterevaluation. Gold mapping and positivecontrols isolated evaluationonly. Model weights frozen; no optimizer/fine-tuning/training, no benchmarkfit/gridsearch. Benchmark influenced earlier human design: DEVELOPMENTonly, not heldout.

Primaryseed42,CPU4threads; dense/requirementCE batch32, globalCE and NLI batch16; existing pinnedmodelrevisions inconfig/manifest. Caches contentaddressed with predictive-sourcefingerprint,config,index,intent/modelpins; old evaluatedpredictions never reused. A fullcachedreplay verifies identicalpredictionbytes; it is not an independentuncached neuralinference run. Syntheticfixedinputdeterminism tests cover dense/globalCE (notNLI unless later explicit test). Source/data/model/prediction hashes persisted.

## Metrics and limitations

Exact/alias mapping stays evaluation-only. Final directrecall denominator all TypeA targetoccurrences; mappedcandidate denominator onlyEXACT/ALIAS. Partial/absent/composite neverdirectcredit. Compositecomplete evidence supplementary. MRR firstdirecthit perTypeAquestion (misszero). Observed mean/median exclude misses, with missingcount/fixedpenalty (totalgraphnodes+1). Reranker candidate metrics measure BEFOREprefix, not unchangedtail in finalrankings. Fullunion reported viaA5. Movement A5->A6/A7 and A6->A7 including removedtop10; successes/failures are targetoccurrences, not uniqueentitycounts.

Failure precedence: temporal, uncertainmapping, noindex,typefilter, recoveredtop10, notunion,outsidepool, contradictionproxyflag, missingrequirementevidence, remainingrequirementrankerfailure. These are operational labels, not established causes or human scientificjudgments. Counterfactual harm/gain in rank does not adjudicate whether nongold entities are valid scientific alternatives.

Report requirementstatus distribution, fourtiers, missingevidence, selectedmulti-evidence and actuallypackedmulti-evidence counts, top3/top1statusdifferences. Diagnostics across all18questions, main identitymetrics TypeAonly. Efficiency logicalcounts+differentiated actualcached inferencepairs, walltimedescriptive; no productionlatency equivalence. NoNLI sharescomputedrelevance;100/200 reuseglobal500scores. Smallbenchmark, mappingceilings,sparsity,yearprecision,inferredroles,pretraineddomainmismatch,NLIcalibration,heuristicparser, fixedaggregation/imputation, singlehardware/seed remain material limitations.

## Repository and AI disclosure

Only V2 modified. User already copied priorV2 externally. Bulk deletion of oldV2directories was automatically blocked, so oldartifacts remain inactive/accessdenied; activephaseoutputs are artifacts/prediction and artifacts/evaluation within the same existingV2, not a newexperimentproject. Superseded top20verification code and unusedlegacycandidate/evaluator modules removed, not kept for backwardcompatibility. V1 started with two tracked reportchanges and four untrackedfiles; user approvedpreservingthem. Compare protectedhashes/diff tostartstate, notHEAD. rest_of_work also hashed in full (86,667files). NoGitHuboperations.

Codex implemented/integrated/executed/verified; Claude Opus5 planningconsultation, Claude Sonnet5 optionalcode review, Claude reportauthorship via officialsubscription Ultralighthelper. Reports have no humanexpertadjudication. Record actual helper models/errors honestly from JSON. Codex supplies measuredhandoff, checksfactualclaims and compilesLaTeX.

Assessment scope: include explicit paragraph that V2 is flat-retrieval companion to unchangedV1. No new hierarchy,P1-P6,T1-T5 compliance or hierarchybenefit claimed. Assessment matrix in notes/ASSESSMENT_ALIGNMENT.md; V1artifactexistence checked only. Concise3-5page companion covers V2method,evalvalidity,results/limitations and fourmoreweeks, with clear scope. AI_USAGE.md lists allcalls/adviceverification. Positive-only tier1 corrected beforefreeze/evaluation based on semantics and reviewer advice, nevergold.

Further parser limitation: pre-head scientific modifiers remain verbatim in original/globalCE and structured retrieval representations, but are not necessarily separate positive ranking components. The literal splitter can underrepresent that specificity. Do not claim complete semantic decomposition. No primary predictive policy was changed after reading current evaluation results.

Generic pre-evaluation bugfix: comma atomization now splits comma+whitespace lists, preserving thousands separators such as50,000. Unit test verifies both quantitative phrase and ordinarycomma-list splitting. All first-stage strings unchanged. Affectedrequirementinput caches miss because keys include rankingcomponents; unaffectedrawmodeloutputs maybecarriedforward. Interruptedruns produced nofreeze/nogoldevaluation and no empiricalselection.

## Ranking complexity (design bound, not a latency measurement)

For poolK<=500, m components including exclusions, M indexed evidence items, embeddingdimensiond, and L_c attached evidence items for candidatec: requirement dotproducts cost O(Mmd); component evidence selection sorts at most L_c rows, totaling O(m sum_c L_c log(max(2,L_c)))) in this implementation. Bundle inference makes at most Km relevance and Km NLI pairs per evidence-k arm, usually fewer due missing/packing, plusKglobalCE pairs shared acrossarms. Percomponent competition_ranks currently computes strict-higher counts over all distinct scores with a nested scan: O(mK^2) worst case, not an optimized O(mKlogK) ranking. Final lexicalkeysort O(KlogK); model cost depends on packedtokenlength andarchitecture. These are logical bounds; tests/runtime count actualpairs and walltime separately. Evidence matrix/premises reused only where cachesareexactcontentmatches; graph/index caches unchanged. The quadratic purePython rank routine is acceptableatK500but is a scaling limitation.
