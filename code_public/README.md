# Free Anchors for LLM Judge Panels — Reproduction Code

Code for the identifiability analysis of LLM judge panels: the gauge theorem
(T3), family-count identification (T1), the common-confusion floor
(Corollary C1(i)), the minimax lower bound for human anchors (T2), and
identification recovery from structural anchors (T4).

## What is here

Fifty-two scripts reproducing every measurement in the paper, from the gauge
theorem's synthetic checks to the two independently judged panels, together
with the judge labels we collected for the second panel.

## Environment

Every analysis runs on **CPU only**, with no GPU. Reproducing the results from
the released data needs no paid service either.

The one exception is the second panel: `e26` and `e27` query nine judges
through their providers' interfaces. Because the labels they collected are
released with this code, `e28` and every other analysis can be re-run without
contacting any provider. The collection cost 2,700 requests, 1.02M input and
65k output tokens.

```
python >= 3.11
pip install -r requirements.txt
```

`torch` appears in `e3_run.py` only inside a guarded check that asserts no GPU
is in use. It is optional; the script runs without it.

## Layout

Scripts expect to be run from a checkout with this structure:

```
<repo>/
  code_public/     this directory
  data/            input data (not distributed; see below)
  results/         JSON outputs, one file per experiment
  paper/figures/   figure output directory
```

Paths are resolved relative to the repository root
(`Path(__file__).resolve().parents[1]`); nothing is hard-coded to a local
machine. Run scripts from anywhere:

```
python code_public/e12_structural_anchor_realdata.py
```

Each script writes its measurements to `results/` as JSON, together with an
environment block recording seeds, direction budgets, `gpu_used`, and
`api_calls`. Several scripts also write figures to `paper/figures/`.

Re-running a script **overwrites** its own result file; `e2b_realdata_floor.py`
merges into an existing file instead of replacing it. Copy `results/` aside
first if you want to keep a previous run.

## Data

Input data is **not redistributed here**. Obtain each source directly:

| Directory | Source | Terms |
|---|---|---|
| `data/llmjudge_submissions/`, `data/llmjudge_gold/` | LLMJudge benchmark (33 judgment sets, human gold labels) | MIT |
| `data/trec_web_diversity/` | TREC Web Track diversity qrels (2009) | U.S. Government public domain |
| `data/trec_dl_qrels/` | TREC Deep Learning passage qrels | U.S. Government public domain |
| `data/umbrela/` | UMBRELA judgments | Apache-2.0 |
| `data/ntcir_intent2/` | NTCIR-10 INTENT-2 Document Ranking subtask | NTCIR research-use terms; **redistribution not permitted** — request access from the NTCIR organizers |
| `data/parry_et_al/` | Published assessor-disagreement data | as stated by the original authors |
| `data/beir_trec_covid/` | BEIR TREC-COVID queries, corpus and qrels, used to build the second panel | CC BY-SA 4.0 (TREC-COVID); download from the BEIR release and point `TRECCOVID_DIR` at it |
| `data/second_panel_treccovid/` | The item sample and the judge labels we collected | released with this code |

`data/family_mapping.json` assigns each of the 33 judges to a model family
(GPT / Llama / T5). It is our own annotation, with the evidence for each
assignment recorded inline, and is included with this code.

## Determinism

Every analysis is deterministic given the seeds recorded in the source files
and in the environment block of each result file. The judge queries were made
at temperature zero, but providers are not guaranteed to be bit-reproducible,
which is why the labels themselves are released. Direction budgets differ
between experiments and are recorded per run: the E2c, E6a, and E10 diameters
use 24,000 random gauge directions (6 seeds x 4,000), while E12, E14, and E15
use 18,000 (6 seeds x 3,000). Measurements taken under different budgets are
not directly comparable, because the diameter is a Monte Carlo lower bound.

## Script index

Synthetic verification of the theorems:

| Script | Purpose |
|---|---|
| `e1_synthetic_phase.py` | Phase diagram under known true parameters (E1: T1, T3, T4) |
| `e1_t1_strengthened.py` | Strengthened T1 test (N>=2000, 100 initialisations, 10 seeds) |
| `e1_radius_scaling.py` | Gauge radius against m = 2..50 |
| `e1_radius_decomposition.py` | Total radius vs non-negative-inverse (G0) radius |
| `e1_propF1_direct_check.py` | Direct check of Proposition F1 along its constructive direction |
| `e1_plot.py` | Figures for the E1 series |

Real-data experiments (LLMJudge, Contract A):

| Script | Purpose |
|---|---|
| `e2_experiment.py` | Family boundaries; within- vs cross-family agreement (E2) |
| `e2b_realdata_floor.py` | Common-confusion floor on real data |
| `e2c_realdata_L4.py` | Identification set diameter for m = 0..3 (E2c) |
| `e2c_diagnostics.py` | Set-vs-diameter separation diagnostics |
| `e12_structural_anchor_realdata.py` | Structural anchors on the real panel (E12) |
| `e14_e12_sensitivity.py` | E12 sensitivity to threshold and extraction seed (E14) |
| `e15_anchor_vertex_coverage.py` | Vertex coverage of the freely available anchors (E15) |
| `e5b_latent_class_gof.py` | Latent-class goodness of fit against reference bands (E5b) |
| `e9_criterion_and_unified_radius.py` | Binary criterion check and unified radius estimator (E9a) |
| `e9c_prototype_unified.py` | Diameter under three prototype constructions (E6a) |
| `e10_radius_at_fitting_K.py` | Diameter by fitting K (E10) |
| `e11_anchor_powerlaw_free.py` | Anchor budget curve, power law with free exponent |

Semi-synthetic and supporting analyses:

| Script | Purpose |
|---|---|
| `e3_experiment.py`, `e3_run.py`, `e3_profiles.py`, `e3_plots.py` | TREC-derived profile construction and analysis (E3) |
| `e3_floor_check.py` | Floor invariance in the number of families |
| `e3_convexhull_check.py`, `e3_dominance_verify.py`, `e3_dominance_crosscheck.py` | Panel domination; the crosscheck recomputes the result independently |
| `e4_reviewer_response.py` | Vertex scaling and domination-boundary tests (E4a, E4b) |
| `e5_conditional_independence.py` | Conditional independence across families (E5) |
| `e6_sensitivity_and_vossler.py` | Prototype and threshold sensitivity |
| `e7_intent_prior_sensitivity.py` | Sensitivity to the intent prior |
| `e8_vossler_contrast.py` | Constructive coexistence example for the two contracts (E8) |
| `e13_ntcir_measured_intent.py` | Interiority under measured intent probabilities, L = 5 (E13) |

Estimator quality and robustness:

| Script | Purpose |
|---|---|
| `e16_direction_budget_convergence.py` | Diameter against direction budget, 250 to 96,000 |
| `e19_worst_direction_optimisation.py` | Optimised rather than sampled direction; shows sampling is loose |
| `e20_optimised_reruns.py` | Partitions and contracts under the optimised estimator (its E12 rows are void; see the file header) |
| `e21_e12_optimised.py` | Anchor sweep under the optimised estimator, reusing E12's own pool |
| `e24_guarded_reverification.py` | Invertibility guard; verifies the m >= 1 values are unaffected |
| `e17_partition_robustness.py` | Four family partitions of the first panel |
| `e18_contract_diameter_contrast.py` | Contract A against contract B at m = 0 |
| `e29_e6a_e10_optimised.py` | Prototype constructions and fitting K on the optimised estimator |
| `e32_anchor_effect_on_interior.py` | Anchor effect measured on the interior prototypes, both panels |
| `e33_binary_criterion_real_data.py` | The arithmetic criterion tested on both real panels |
| `e34_anchor_gold_check.py` | Camera-ready: gold labels of the consensus-selected anchor pools on both panels |
| `e35_gold_confirmed_anchors.py` | Camera-ready: E32 first-panel sweep with gold-confirmed anchors only |
| `e36_gold_confirmed_anchors_second_panel.py` | Camera-ready: E32 second-panel sweep with gold-confirmed anchors only |

Second panel (independently judged, requires provider credentials):

| Script | Purpose |
|---|---|
| `e25_build_second_panel_sample.py` | Fixes the 300-item TREC-COVID sample, stratified by gold label |
| `e26_run_second_panel.py` | Queries nine judges from seven providers at temperature zero |
| `e27_retry_missing_labels.py` | Recovers labels lost to the output-token cap |
| `e28_second_panel_analysis.py` | Family-count analysis on the second panel |
| `e30_second_panel_anchors.py` | Anchor sweep on the second panel |
| `e31_second_panel_vertex_coverage.py` | Which vertices the second panel's free items reach |

Credentials for `e26`/`e27` are read from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
and `OPENROUTER_API_KEY` in the environment, or from a dotenv file named by
`API_KEYS_ENV`. The collected labels are released, so every analysis can be
reproduced without contacting any provider.

Figures:

| Script | Output |
|---|---|
| `fig_protocol_flow_v2.py` | Diagnostic protocol and anchor allocation |
| `fig_e2_agreement_v2.py` | Within- and cross-family agreement |
| `fig_intervention_contrast.py` | The three interventions on a common scale |

Earlier versions of both figure scripts exist in the working tree but are not
included here: the v1 protocol flowchart routed a failed Kruskal test to T4,
which is wrong because T4 presupposes that test. Only the corrected v2 scripts
are released.

Three further scripts are not included. `e1_gauge_convergence.py` contains a
syntax error, never executed, and supports no claim. `e22` and `e23` solved the
inner maximisation by linear programming without imposing invertibility, so
their optima are attained by the singular matrix `A = e_j 1'`, which is outside
the gauge group; `e24` carries the part of that work which survives, namely the
closed form `2(1 - eta)` at m = 0 and the guard check at m >= 1.

## Scope

The real-data conclusions come from a single LLM judge panel and from two
document ranking collections in Japanese and Chinese. Diameters are Monte
Carlo lower bounds. Numerical values should not be transferred to another
panel without rerunning the diagnostic.

## License

MIT. See `LICENSE`.

The license applies to the code in this directory only. The input datasets
listed above are governed by their own terms, and the NTCIR-10 INTENT-2 data
may not be redistributed.
