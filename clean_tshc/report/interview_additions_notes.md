# Interview study layer added to `full_report.tex`

Local working-tree edit only. Nothing was staged, committed or pushed.

The original report text is unchanged: all 2,634 original lines are still present, byte-identical
and in the same order. 2,380 lines were inserted in 13 places. Original section, figure, table,
algorithm and equation numbers are the same as before (new figures are 10-22, new tables 23-26,
no new numbered equations).

Page counts: 45 pages before, 84 pages after. The page number printed in the footer is the PDF
page minus one (the title page is unnumbered).

To get the original back: `git checkout -- clean_tshc/report/full_report.tex clean_tshc/report/full_report.pdf`

## What was inserted in the body (blue "Interview intuition" boxes)

| Box | Next to | PDF page |
|---|---|---|
| Navigation box for the study layer | end of the abstract | 2 |
| Ward merge cost | Section 2.3, Equation 1 | 12 |
| Hyperedge entropy fragmentation | Section 2.4 | 12-13 |
| Variation of information | Section 2.5, Equation 3 | 13 |
| ARI versus VI (comparison table) | Section 2.5 | 14 |
| Empirical p-value with 100 nulls | Section 2.11, Equation 4 | 15 |
| Student-t interval with five seeds | Section 2.11 | 16 |
| Full objective J | Section 4.2, Equation 5 | 20 |
| Delta S | Section 4.3, Equation 6 | 21 |
| Delta F | Section 4.3, Equation 7 | 21 |
| Delta T | Section 4.3, Equations 8-10 | 22 |
| Coherence as length of the mean vector | Section 5.11 | 33 |

Each box covers: one-sentence meaning, what changes the quantity, what high/low means, why it is
normalised, and one likely interviewer misconception. Several include a toy arithmetic check.

## Appendix E: Interview Study Companion (PDF pages 52-83)

| Part | Content | PDF page |
|---|---|---|
| E.1 | How to use: three kinds of statement, memory aids, "draw the model in one minute" | 52 |
| E.2 | Thirteen diagrams (below) | 53-59 |
| E.3 | Mathematics-to-code map: `Merger.delta` listing, symbol table, concept/function/test table | 60-61 |
| E.4 | Design choices: why, cost, rejected alternative | 61-62 |
| E.5 | Numbers to recall | 63 |
| E.6 | 100 questions | 64-83 |

Diagrams:

| # | Figure | Topic | PDF page |
|---|---|---|---|
| 1 | 10 | P1-P6 map: by construction vs measured | 53 |
| 2 | 11 | Hierarchy cascade: singletons, 120, 40, 12 | 53 |
| 3 | 12 | Objective: three-way tension S / F / T | 54 |
| 4 | 13 | One worked merge (toy numbers) | 55 |
| 5 | 14 | Native hyperedge vs clique projection (5 nodes) | 55 |
| 6 | 15 | Hyperedge collapse, three cases | 56 |
| 7 | 16 | Temporal mechanism: regularisation vs matching | 56 |
| 8 | 17 | Event taxonomy | 57 |
| 9 | 18 | Evaluation circularity barrier | 57 |
| 10 | 19 | Label generation and held-out check | 58 |
| 11 | 20 | Flat vs beam retrieval | 58 |
| 12 | 21 | Why retrieval failed: evidence hit vs identity hit | 59 |
| 13 | 22 | Result dashboard | 59 |

Question groups:

| Group | Questions | PDF page |
|---|---|---|
| Task, data and problem formulation | Q1-Q10 | 64 |
| Hypergraphs and projection | Q11-Q19 | 66 |
| The formal objective | Q20-Q33 | 67 |
| Construction algorithm and implementation | Q34-Q48 | 70 |
| Variants and temporal treatment | Q49-Q57 | 73 |
| Labels and evaluation validity | Q58-Q72 | 75 |
| Retrieval and benchmark methodology | Q73-Q88 | 77 |
| Results, interpretation and defence | Q89-Q100 | 80 |

## Appendix F: Interview Emergency Sheet (PDF page 84, one page)

Objective, parameters, hierarchy, snapshots, model roles, P1-P6 matrix, key numbers, eight
"do not overclaim" warnings, eight one-line answers.

## How the 100 questions relate to `clean_tshc_interview_100_QA.md`

The answers come from that file, reshaped into: 30-second answer, "if they push further",
common trap, and where to find it. The numbering differs from the `.md` file because four pairs
were merged to make room for four questions the brief asked for:

| In the PDF | From the `.md` file |
|---|---|
| Q14 | md 14 + 15 (fragmentation and why divide by log r) |
| Q21 | md 22 + 23 (S and why divide by 4\|V\|) |
| Q32 | md 34, extended with the sensitivity study |
| Q52 | md 54 + 55 (Hungarian matching and why Jaccard) |
| Q79 | md 82 + 83 (flat baseline and the 5,480 documents) |
| Q96 | new: what claims can you not make |
| Q97 | new: why should we trust these results |
| Q98 | new: what would you change |
| Q99 | new: four more weeks |
| Q100 | md 100 (one-minute summary) |

Everything else keeps the `.md` order.

## Statements that go slightly beyond the report text

These are arithmetic consequences of what the report states, added because an interviewer may
ask. They are marked in the PDF where they appear.

- The normaliser `2 log n` for VI is a valid but not tight bound. The report states `VI <= log n`,
  so normalised `T` cannot exceed 0.5. This only rescales lambda.
- Toy examples (Ward check, fragmentation splits, the worked merge in Diagram 4) use made-up
  inputs and are labelled "toy" or "not data".
- "Not established by this implementation" is written wherever the clean implementation has no
  experiment: ARI as regulariser, spectral or two-stage comparisons, other pairwise encodings,
  other cluster representations, the arity-preserving null, optimality of k = 10.

## Build

From `clean_tshc/report/`:

    latexmk -pdf -interaction=nonstopmode -halt-on-error full_report.tex

Run it from PowerShell or cmd. In Git Bash, MiKTeX fails with a PATH error on this machine.
Result: 84 pages, no errors, no undefined references, 0 overfull boxes, 2 underfull boxes (the
same 2 as in the original). No new packages were added.
