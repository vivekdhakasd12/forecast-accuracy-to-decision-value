# Exposé review: From Forecast Accuracy to Decision Value: Decision-Focused Learning for Day-Ahead Renewable Energy Commitment

Date: 2026-11-16 (thesis week 7 of 24). Scope: the exposé (`docs/Expose_Master's_Thesis.html` / `.pdf`), checked against the Case Study 1 code, results and report in this repo. No other repo files were modified.

How this was produced: six independent review lenses (claims vs evidence, DFL technical fit, literature, data, research design, codebase readiness). A separate verifier re-checked each lens's findings and tried to refute them. 63 findings were raised; none was refuted outright, about a third were downgraded or corrected, and the severities below are the verified ones. The raw findings with verdicts are in `expose_review_findings.json`. Literature entries marked "verify" were found by web search but not fully confirmed.

> **Update, 16 Nov 2026:** the scope stays exactly as in the approved exposé. Where this review suggests dropping or moving items (SPO+, the differentiable layer, drift, transfer, multi-zone), ignore that and use the full-scope plan in [THESIS_IN_PLAIN_WORDS.md](THESIS_IN_PLAIN_WORDS.md), sections 7 and 8. The technical and quality findings still apply.

## Verdict in brief

The thesis is sound and still feasible. The research question (how much of the gap to perfect foresight a decision-trained model closes, and what it gives up in calibration) is legitimate. The CS1 codebase is a strong, tested base for it: leakage-audited features, a two-stage HiGHS LP, and an oracle-based regret. As written, though, the exposé is not ready to execute. Three things matter most:

1. **The decision problem and settlement model are undefined.** The exposé describes a market-facing generator under asymmetric dual-price settlement. The code models a prosumer on a retail tariff (`dispatch.py:1-20`). Real German settlement (reBAP) is single-price, and the current LP becomes unbounded once real prices are used.
2. **The data and the timeline have slipped.** OPSD stops at 2020-09-30, no zone has imbalance prices, and none of the work planned for weeks 1-6 exists in the repo.
3. **The comparison design cannot yet detect the effect it is looking for.** CS1 effects sit at the noise floor (detectable effect about 6 EUR/day on 200 days). The baselines are weaker than a tuned commitment quantile, and with zero storage the "cheap" cost-weighted loss already is the exact decision loss.

SPO+ also does not apply as stated, because the forecast enters the constraint right-hand side and not the cost vector. Fix these in the next two weeks, narrow the scope to one zone and two regimes, and the thesis is very doable.

## What is strong

- **Infrastructure.** Leakage-safe features enforced at runtime (`features.py:53-58`) and tested adversarially (`tests/test_no_leakage.py:35-67`). The suite has 22 tests and passes in 9.5 s. HiGHS is called through `linprog(method='highs')` (`dispatch.py:202-203, 262`). Settlement is kept separate from commitment (`evaluate_commitment`, `dispatch.py:220-283`), and the oracle regret (`dispatch.py:286-292`) matches the definition used in DFL papers.
- **Exact loss and cheap gradients.** Settled cost is convex and piecewise linear in the committed schedule for a given realised PV, and HiGHS returns duals. Every DFL loss can therefore be the exact realised cost, and subgradients come almost for free.
- **Fast LP.** It solves in 1.6 ms with 1 scenario and 64 ms with 40 scenarios, so per-day solves inside training are affordable.
- **Model plumbing.** One `Forecaster` interface and registry (`models/base.py:32-104`, `pipeline.py:78-113`). The PyTorch LSTM already has a monotone quantile head (`neural.py:67-92`).
- **Staged design with a guaranteed thesis.** The plan in Sec. 5-6 is cost-weighted loss, then SPO+, then a differentiable layer, and Sec. 7 explicitly accepts a negative outcome.
- **Honest framing carried over from CS1.** The synthetic tariff and national-aggregate target are acknowledged (`report.tex:838, 903`), and the limitations chapter (`report.tex:826-875`) is effectively a ready-made fix list.
- **Most headline numbers trace exactly to `results/*.csv`** (see Claims check).
- **Paired daily design.** Daily regret differences have lag-1 autocorrelation of 0.03, so paired tests on all test days have good power.
- **Correct core references.** All 12 references in Sec. 8 exist and the lineage is right (Pinson 2007, Morales 2014, Gneiting & Katzfuss, CQR, ACI, SPO+, OptNet, cvxpylayers, Donti, Mandi survey).

## Critical issues

### C1. The decision problem and settlement model are undefined, and real prices break the current LP

**Problem.** Sec. 2 describes "a renewable generator trading in the European day-ahead market" that is short at a punitive price and long for little. The implemented model is a PV and battery prosumer: national DE capacity factor scaled to 2 MW (`policies.py:58-60`), synthetic 1.2 MW load (`policies.py:77-85`), TOU import 220/120 and feed-in 45, and fixed penalties of 300/90 (`configs/default.yaml:38-46`). Sec. 3 Objective 3 and Sec. 5 then swap in settled DA and imbalance prices. Four consequences follow:

- **Single-price settlement removes the asymmetric premise.** German reBAP is single-price, GB has used single cash-out since 2015 (P305), and the Nordics moved to single price on 2021-11-01. The deviation cost becomes (λ_imb − λ_DA) × deviation: linear, sign-random, and correlated with system state.
- **The LP becomes unbounded.** This was reproduced. With export_price = 250 or import_price_offpeak = −50, HiGHS returns status 10 (unbounded), because gp and gm have no upper bound (`dispatch.py:80-82`). The cached DE panel already holds 484 negative-price hours.
- **Failed days vanish silently.** A failed solve returns `total_cost = NaN` (`policies.py:133-134`). `summarise_policies` skips NaN in its mean while `n_days` still counts the row (`e2_dispatch.py:85`).
- **The target builds in an unrealistic error-price correlation.** The site's error is the national solar error (`opsd.py:215-222` × `policies.py:60`), which co-moves with reBAP. No single 2 MW plant faces this.

**What to do.** This week, write a one-page decision-model note and get the supervisor to sign it:
- Choose market-facing generator (BRP) plus optional battery, with commitment bounded in [0, Q_max]. Keep the prosumer LP only to reproduce CS1.
- Run two evaluation regimes. **Regime A (mechanism):** a stylised dual-price tariff and the penalty × storage grid. **Regime B (external validity):** DE-LU real DA price plus reBAP, with position limits.
- Add grid and position bounds and time-varying price vectors. Make `run_policy` raise on infeasible or unbounded solves.
- In Regime B, report results with real λ_imb and with λ_imb shuffled within month × hour. That separates forecast value from the national-aggregate correlation artefact.
- Never mix real DA prices with a synthetic imbalance penalty without labelling it.

### C2. The data cannot deliver Objective 3, and the data step is overdue

**Problem.** All four interim panels end 2020-09-30 23:00 UTC (the final OPSD release). The DE `price_eur_mwh` starts 2018-09-30 and is 65.2% NaN, because `opsd.py:36` never reads the pre-split DE_AT_LU price. CH and DK have no price at all. GB_GBN has spikes up to 999 that were never checked. No panel has an imbalance price. `data/raw/` is empty, there is no ENTSO-E, reBAP or Elexon loader, and `opsd.py:22-25` downloads from an unpinned `/latest/`. Sec. 6 put this work in weeks 3-4.

**What to do.** Make the data layer the only implementation work for weeks 7-8 (see the Data plan). The Sec. 5 fallback ("keep the synthetic tariff") does not rescue the Sec. 7 promise of "euros on settled market prices". Either deliver Regime B for DE or rewrite that promise.

### C3. Effect sizes sit at the noise floor, and there is no inference plan

**Problem.** Recomputed from `results/e2_daily.csv`, the headline stochastic-vs-point gain is 4.89 EUR/day (paired SD 30.9, n = 200, t = 2.24, CI [−9.2, −0.6]). The 80%-power minimum detectable effect is about 6.1 EUR/day at n = 200, 4.5 at n = 365 and 3.2 at n = 730. The "inversions" in the motivation (Sec. 2) are differences of 0.4-0.5 EUR/day, roughly 10× below detectability. CS1 ran no significance tests (`report.tex:752`), used 3 folds (`:840`) and one seed (`:862`). DFL training is seed-sensitive, so variation between seeds can exceed the effect.

**What to do.**
- Settle every test day, not a 200-day subsample.
- Make the delivery day the unit of inference. Use a Diebold-Mariano test with HAC (Newey-West) variance on paired daily cost differences, plus moving-block bootstrap CIs (block of about 7 days).
- Train each stochastic model on at least 5 seeds, average over seeds before the paired test, and report the seed spread separately.
- Pre-register at most four primary comparisons with a Holm correction. Use a Model Confidence Set to rank many configurations.
- Put the power calculation in the methods chapter.

### C4. The planned comparison is confounded and the baselines are too weak

**Problem.** Five issues:
- **(a) Model class.** CS1's best models are ridge and GBM, while gradient DFL needs a differentiable class, so loss is confounded with model class.
- **(b) Decision rule.** The baselines use 40-scenario SAA, whereas DFL usually plugs a point forecast into a deterministic LP.
- **(c) Tuning effort.** Baselines are "retained unchanged" with no hyperparameter search (`report.tex:752`), while DFL gets multi-week tuning phases (Sec. 6).
- **(d) Quantile chosen on test data.** The CS1 optimum q = 0.55 (149.06 EUR/day, `e2_quantile_sweep.csv`) was selected on the scored days (`e2_dispatch.py:92-121, 244`). It beats both stochastic (159.82) and point (164.71). Ridge[cqr] at 144.32 (`e2_by_model.csv`) is lower still. None of these is in the Sec. 5 baseline list.
- **(e) Wrong fractile in the code.** The newsvendor fractile is wrong (`policies.py:167-169` uses co = 90, giving 0.667; `report.tex:545, 563` repeat it). Over-committing import also costs the DA price, so the net-load fractile is (300 − p_DA)/390. The implied PV quantiles are 0.538 off-peak, 0.795 peak and 0.346 in the export regime, with a kink at g = 0. The empirical 0.55 matches the off-peak value, so the storage-based explanation in `report.tex:563` is probably wrong. The function only drives a plot line (`e2_dispatch.py:246`), but the theory it encodes matters (see the Recommended technical formulation).

**What to do.**
- Run a factorial comparison within one model class: loss {pinball, τ_t-weighted pinball, DFL} × model {linear, small MLP} × decision rule {deterministic plug-in, τ_t-quantile plug-in, SAA}, with identical features and folds.
- Give baselines and DFL the same tuning budget on the same validation block. Choose τ, hyperparameters and `common_factor` on validation days only.
- Show CS1 ridge+SAA and tuned-q ridge as external reference rows.
- Fix `newsvendor_quantile` to return per-hour, per-regime fractiles, and correct the `report.tex` explanation before reusing any text.

### C5. The timeline has slipped and the original plan leaves too little writing time

**Problem.** `find -newermt 2026-09-30` shows nothing new except supervisor email drafts: no data, no loaders, no baseline re-run, no DFL code. Sec. 6 has the literature in weeks 1-2, data in weeks 3-4 and baselines in weeks 5-6, all of which are now due. The riskiest component (the layer) sits in weeks 14-17, leaving only weeks 22-24 for writing and the defence. Sec. 6 says dates are "to be confirmed", so confirm the official start and end dates with the examination office.

**What to do.** Adopt the revised plan below. Freeze scope in writing this week. Set a kill criterion: if the gradient-based arm is not stable by the end of W14, stop and write up the cost-weighted and ERM results.

## Major issues

1. **SPO+ does not apply as stated (Sec. 3 Obj 2, Sec. 4, Sec. 5, Sec. 6 weeks 10-13).** PV enters only the equality right-hand side, `b = load[t] - pv[s,t]` (`dispatch.py:100-104`). The cost vector holds only known tariffs (`dispatch.py:190-200`). SPO+ and its Fisher-consistency result assume a predicted cost vector over a fixed feasible set (PyEPO supports only that case too). Drop SPO+ as a core arm. Keep it only if prices become the predicted quantity, which is a different question. Free weeks 10-13 for the ERM arm. Related point: because the uncertainty sits in the right-hand side, the optimal value is piecewise linear, not piecewise constant, in the prediction. The Sec. 5 statement that the "true gradient is zero almost everywhere" describes the cost-vector case. Restate the risk accurately.
2. **The motivation overreads r = 0.88 (Sec. 2).** Pearson(CRPS, regret) = 0.884 has CI [0.72, 0.95] and falls to 0.62 without Prophet (`report.tex:571, 777` say this; the exposé drops it). Pearson(MAE, regret) is 0.98 (0.94 without Prophet, since raw/CQR pairs share MAE). Regret mostly tracks point accuracy, and r = 0.88 is a strong correlation, so "diverge" is the wrong word. Rewrite: the gap DFL can close lies in choosing the commitment quantile and handling storage-coupled hours, not in "beating CRPS".
3. **The gap statement is overclaimed (Sec. 2, Sec. 4).** "Mainly on small or synthetic problems", "continues to evaluate models by scoring rule" and "not yet standard for day-ahead market commitment" are contradicted by value-oriented trading work since 2019 (Carriere & Kariniotakis 2019; Muñoz et al. 2020; Stratigakos et al. 2022; Wahdany et al. 2023; Zhang et al. 2024). The only energy-DFL paper cited, Yu et al. 2024, is about preventive dispatch. Reposition the contribution as: a strong calibrated two-stage baseline under a leakage-audited rolling-origin protocol, the calibration cost of decision training plus conformal repair, a controlled mechanism grid, and real-settlement robustness.
4. **The cost-weighted quantile loss is undefined, and the conditions under which DFL can win are not stated (Sec. 3, Sec. 5).** At battery = 0 the problem decouples into 24 newsvendors. Realised cost is then a scaled pinball loss at τ_t within each import/export regime, plus the kink at g = 0. τ_t-weighted pinball is therefore the exact DFL loss there, not a cheap approximation (Gneiting 2011; Ban & Rudin 2019). DFL can win strictly only through battery coupling, the kink, time-varying or uncertain prices correlated with the error, the terminal-SoC obligation, or misspecification. Write the loss down and pre-register "≈0 advantage at storage = 0" as a sanity check.
5. **Stage-2 recourse is anticipative.** `evaluate_commitment` solves one 24-hour LP with the full realised PV (`dispatch.py:240-263`), and the scenario recourse is full-path too. This inflates storage value, which is exactly where DFL headroom lives (`report.tex:793`). Add non-anticipative settlement (rolling re-solve, or at least a causal battery rule). Re-run the CS1 storage grid under both settlements in week 8, and train and evaluate under the same settlement.
6. **The scope is too large for 17.5 remaining weeks (Sec. 3, Sec. 6).** There is one main question with three moderators, five sub-questions, six objectives, three DFL arms, multiple zones, drift and transfer. Move SQ4 (drift), SQ5 (transfer) and multi-zone (Obj 4) to stretch or outlook. CS1's drift detector fired once in two years, and its thresholds were set after seeing the data (`report.tex:866-872`).
7. **The transfer claim overstates e5 (Sec. 3 SQ5, Sec. 7).** Zero-shot CH has R² = −6.03 and PICP90 = 0.24. Zero-shot beats a local model only up to 120/240/240 days (`e5_crossover.csv`). The test used ridge only, the targets are national series (not bidding zones), and there was no cost evaluation. Restate it as "pooled training with short local history transfers MAE; zero-shot intervals are invalid".
8. **The baselines cannot be "retained unchanged" (Sec. 5).** The decision problem changes (C1), and the timeline (weeks 5-6) itself says "re-establish". Remove "unchanged".
9. **The e2 numbers come from inconsistent samples and are not reproducible.** The sensitivity grid uses 90 days (`e2_dispatch.py:257`), by-model and the q-sweep 120 (`:269`, `:244`), and the summary 200. The Makefile `e2` target lacks `--sensitivity`, so `make all` never rebuilds `e2_sensitivity.csv`, and help and `.PHONY` still list e1..e4. The default cell shows 6.4% in the grid against 3.0% in the summary. Use one common day sample and add `--sensitivity` to `all`.
10. **The penalty × storage grid is noisy and invites multiple-comparison problems.** It has 16 cells × 90 days, its middle is non-monotone and "not interpretable" (`report.tex:844-846`), and the advantage ranges from 2.7% to 67.5%. Run every cell on all test days with paired CIs. Pre-specify monotone-trend tests across storage at fixed penalty; everything else is descriptive.
11. **The folds have no validation block.** `make_folds` builds train / 90-day calibration / 365-day test (`backtest.py:42-72`), giving 3 folds. DFL early stopping, τ selection and CQR calibration would share or leak into one slice. Use monthly or quarterly rolling origins with train / validation / conformal-calibration / test blocks.
12. **SQ3 (calibration) is not operationalised.** A point-output DFL model has no CRPS or PICP. Decide which DFL models output quantiles. Define coverage at the decision level as P(y ≤ committed quantile) against τ_t by hour and month, and apply ACI at that level. Train risk-neutral only, and leave CVaR as a footnote, since it was cost-dominated in CS1 (`e2_summary.csv`: 198.4 and 207.7 vs 164.7).
13. **The differentiable-layer arm is underscoped.** At S = 40, T = 24 the LP has 4,848 variables and 1,920 equalities per day (`dispatch.py:153-186`). The scenario generator uses sort, interp at random uniforms and clip, with a fixed `common_factor = 0.75` (`policies.py:29-55`). No smoothing remedy and no compute budget are named. If you keep it, use a deterministic-equivalent layer or at most 8 fixed-quantile scenarios with a quadratic regulariser (Wilder et al. 2019). Benchmark one forward/backward pass in week 8.
14. **Imbalance settlement design is missing from the literature and methods.** Single vs dual pricing and price uncertainty at the gate (Zugno et al. 2013; Stratigakos et al. 2021) are absent. Add a short subsection and state the settlement rule for each zone.
15. **Gate and delivery day are in UTC.** `gate_hour: 12 # (UTC)` (`configs/default.yaml:13`), with origin normalised in UTC (`features.py:29-35`). The EPEX gate is 12:00 CET/CEST (11:00 or 10:00 UTC), so the features read 1-2 hours past the gate. Local delivery days have 23 or 25 hours, but `features.py:167` and `e2_dispatch.py:41` force 24. Move to Europe/Berlin and allow T ∈ {23, 24, 25}.
16. **Settlement granularity.** reBAP and actuals are quarter-hourly, and SDAC moved to a 15-minute MTU from 2025-10-01. End the study window on 2025-09-30. Settle each hourly commitment flat over four quarter-hours against quarter-hour actuals and reBAP, without averaging reBAP to hourly.
17. **The study period determines the price regime.** OPSD-era DE DA averages 35.8 EUR/MWh, while 2021-22 crisis prices would dominate any pooled euro figure. Fix the split before training, report results by regime, and add normalised regret alongside EUR.
18. **The zone set needs replacing.** GB left ENTSO-E in June 2021 and prices in GBP. DK consists of DK1 and DK2 and switched to single price in Nov 2021. CH is outside SDAC. If portability survives as a stretch goal, use DE-LU, NL and DK1.
19. **Decision-time price leakage is an unguarded hazard.** Realised D+1 DA prices clear after the gate. Separate decision-time price information from settlement prices, and add a test that poisons post-gate prices and imbalance prices and asserts the schedules are unchanged.
20. **The interfaces are not DFL-ready.**
    - `Forecaster.fit(X, y)` has no decision context (`base.py:93-94`).
    - The backtest catches any TypeError and retries silently (`backtest.py:84-91`, also `drift/policies.py:40-46`), and passes `history=X` including the test fold (`backtest.py:126`).
    - LSTM batches are random rows, not days (`neural.py:221-228`).
    - The k-to-MW conversion, night clamp and site scaling are numpy-only (`base.py:62-79`, `policies.py:58-60`).
    - The LP exists only as hand-built scipy sparse matrices, and cvxpy/cvxpylayers are not installed.

    Add a `decision_ctx` argument, a day-batcher and differentiable post-processing. Build one cvxpy model with an equivalence test against HiGHS.
21. **There is no version control and no lockfile.** `git status` returns "not a git repository". pyproject.toml has only lower bounds while the venv has numpy 2.4.6, pandas 3.0.5 and torch 2.14.0. Run `git init`, tag `cs1-final`, and add a lockfile today.

## Minor issues and corrections

- **"7% worse CRPS was cheaper to act on" (Sec. 2).** This is gbm_quantile_k raw vs CQR: regret 161.74 vs 162.13, a 0.39 EUR/day difference, with identical MAE. It is a recalibration of one model and statistically indistinguishable. Drop it, or save per-model daily regrets and use a pair whose CI excludes zero.
- **"Twenty model variants".** These are ten models × {raw, CQR}, and several regret pairs differ by under 1%. Report r at the model level too (n = 10: r = 0.93, CI [0.73, 0.98]).
- **ACI "repaired" coverage.** ACI's worst month is 0.857/0.874/0.879, still below 0.90. Write "improved from 83% to 86-88%". `report.tex:771` and `:883` say 81% / 0.904, which no CSV supports, while `:664` and `e3_aci.csv` give 83% / 0.903. Fix these before reusing the text. Pre-register a conditional-coverage target.
- **HTML `<title>`** reads "Exposé · Case Study 2" (`Expose_Master's_Thesis.html:5`).
- **"German generation data"** is the national OPSD aggregate. Say so.
- **Yu et al. (2024) entry.** The title is truncated and the fifth author is missing. Correct entry: Yu, Q., Li, Z., Han, X., Ju, P., & Shahidehpour, M. (2024). End-to-end learning for stochastic preventive dispatch of renewables-rich power systems in abnormal weather conditions. *Renewable Energy*, 234, 121107. doi:10.1016/j.renene.2024.121107. Also remove the "verified against Crossref" sentence (Sec. 8).
- **Conference references** use arXiv IDs instead of proceedings: Agrawal 2019, Amos & Kolter 2017 (PMLR 70, 136-145), Donti 2017 (NeurIPS 30, 5484-5494), Romano 2019, Gibbs & Candès 2021. Keep JAIR vol. 80 for Mandi et al. Use Zotero with Better BibTeX from now on.
- **SQ1 "share of the gap closed"** has no estimator. Define it as a ratio of mean daily regrets with a block-bootstrap CI, always reported next to the absolute EUR/day difference, with the best same-class two-stage baseline as the reference.
- **Settled vs estimated imbalance prices.** Quality-assured reBAP arrives about 20 working days after month end. Use it for evaluation only, never as a feature (lag of at least 2 days if used at all), and record download dates.
- **No NWP or TSO forecast input** (`report.tex:903`). A weak base forecaster can inflate the measured DFL gain. Add the ENTSO-E 14.1.D DA solar forecast as both a feature and a baseline.
- **DE price splice at 2018-10-01** (`opsd.py:36`). This becomes moot once the ENTSO-E period starts in 2019. If you splice, add a regime flag.
- **Common random numbers.** One RNG is consumed sequentially by the stochastic and CVaR policies (`e2_dispatch.py:52-64`), so there are no common random numbers. Seed per (date, model) and share scenarios across policies. Add a regression test that freezes the CS1 CSVs.
- **Tautological test.** `tests/test_no_leakage.py:141-143` asserts `(origins - span <= origins).all()`, which is always true. Add a fold-isolation test for `run_backtest`.
- **GBM hooks.** GBM `fit` has no `sample_weight` or custom objective (`gbm.py:110, 161`). Add both for a τ_t-weighted GBM.
- **Hygiene.**
  - `presentation/colab_bundle` duplicates `src` and the folder holds about 38 MB of media. Move it out of the tree.
  - `_CAPACITY_REPORTED` trimming is skipped on cached loads (`opsd.py:109-111, 202-209`).
  - `scale` and `ref_mae` are unused (`backtest.py:121-122`).
  - `OMP_NUM_THREADS = 1` (`__init__.py:23-26`) will slow torch training. Run it in a separate process.
- **Normalisation leak.** The 99.9th-percentile normalisation for transfer zones is computed over the full sample including test (`opsd.py:219`). Compute it on train folds only.

## Claims check: Case Study 1 numbers cited in the exposé

| Claim (exposé) | Source | Status |
|---|---|---|
| CRPS vs regret correlation r = 0.88 | `e2_by_model.csv` (0.884) | Exact, but needs CI [0.72, 0.95]. 0.62 without Prophet. MAE correlates more (0.94-0.98). Reword "diverge". |
| Twenty model variants | `e1_leaderboard.csv`, `e2_by_model.csv` | Misleading: 10 models × {raw, CQR}, near-duplicate pairs. |
| 7% worse CRPS was cheaper to act on | `e2_by_model.csv` gbm_quantile_k raw/CQR | CRPS gap exact (7.45%). Cost gap 0.39 EUR/day is noise. Drop. |
| Stochastic advantage 3 to 67% | `e2_sensitivity.csv` (2.73 to 67.49) | Rounded fairly, but 90 days, one model, not rebuilt by `make all`, no CIs, non-monotone. The base case is 3.0% (`e2_summary.csv`). |
| Risk-averse policies halve shortfall but cost more | `e2_summary.csv` | Exact and significant (0.45 to 0.21-0.22 MWh/day; CVaR +33.7 [17, 50], fixed quantile +43.0 [28, 58] EUR/day). |
| Split conformal worst month 83%, repaired by ACI | `e3_aci.csv` | 83% exact. "Repaired" overstated (ACI 0.86-0.88). `report.tex:771, 883` carry stale 81% / 0.904. |
| Score-trained model transfers across bidding zones | `e5_transfer_mean.csv`, `e5_crossover.csv` | Overstated: national series, MAE only, zero-shot CH R² = −6, no cost evaluation. |
| Leakage-safe features verified by an adversarial test | `tests/test_no_leakage.py:35` | Exact. Covers feature construction only. |
| HiGHS two-stage LP | `dispatch.py:202-203, 262` | Exact. |
| Rolling-origin backtest with held-out conformal block | `backtest.py:42-72`, `report.tex:313` | Exact. 3 folds, no validation block. |
| One-command reproduction | `Makefile` | Partly true: `make all` runs test plus e1-e5 but omits the sensitivity grid. |
| Prosumer model framed as a generator in the day-ahead market | `dispatch.py:1-20`, `report.tex:319-325` | Wrong framing: CS1 is a prosumer on a retail tariff. |

## Recommended technical formulation

**Problem statement (generator / BRP, price-taker).** For delivery day d with features x_d available at the local gate (12:00 Europe/Berlin, D−1), commit an hourly schedule q_d ∈ [0, Q_max]^T, with T ∈ {23, 24, 25}. Realised PV y_d (quarter-hourly) and an optional battery with non-anticipative recourse determine the settled cost C(q_d; y_d, λ_d). Regret is R_d = C(q_d) − C(q*_d), where the PV-oracle q*_d knows y_d but faces the same price information as the policy. All costs are settled ex post on the same prices.

- **Regime A (mechanism, core).** Stylised dual price: short at λ_DA + π_s, long at λ_DA − π_l. Use a grid over (π_s, π_l) × battery {0, 1, 2, 4 MWh}. The newsvendor structure is intact, so the DFL question is well-posed.
- **Regime B (external validity, core for DE only).** Real DE-LU DA price plus reBAP (single price), with position limits. Report raw, normalised and λ_imb-shuffled results.

**Hypotheses (pre-register).**
- **H1:** On held-out days, the best DFL arm reduces mean daily regret against the best tuned two-stage baseline of the same model class (DM/HAC test, alpha = 0.05).
- **H2:** At battery = 0 in Regime A the advantage is ≈ 0, because τ_t-pinball is the exact decision loss.
- **H3:** The advantage increases with storage size and with penalty asymmetry (pre-specified trend test).
- **H4:** Decision-trained quantile models under-cover at the decision level. ACI on top restores coverage while keeping at least X% of the cost advantage (X fixed in advance).

**Baselines (all tuned on validation only).**
- B0: PV-oracle (lower bound).
- B1: climatology and persistence.
- B2: point model with deterministic plug-in.
- B3: quantile model at analytic per-hour, per-regime τ_t, plus a validation-tuned τ_t vector.
- B4: CQR + ACI at τ_t.
- B5: SAA stochastic program with tuned `common_factor` and τ_t-consistent marginals.
- B6: ENTSO-E TSO DA solar forecast plug-in.
- External rows: CS1 ridge+SAA and tuned-q ridge.

**DFL arms, in order.**
1. **D1: τ_t-weighted pinball (core, weeks 10-11).** Exact DFL at battery = 0. Use it in ridge/linear, a small MLP and LightGBM (via objective or weights).
2. **D2: linear decision ERM through the settlement LP (core, main result, weeks 12-14).** Use q_d,t = θ_hᵀ x_d,t, or a linear forecast fed into a fixed deterministic dispatch, and minimise Σ_d C(q_d; y_d) + λ‖θ‖² as one convex LP/QP in HiGHS (about 84k recourse variables, benchmark first). This gives the globally optimal linear decision rule and an internal validity reference. It follows Ban & Rudin (2019), extended with battery recourse.
3. **D3: subgradient training of a small MLP using settlement duals for ∂C/∂q (core if D2 shows headroom, weeks 14-15).** No differentiable layer is needed when the model outputs the decision.
4. **D4: regularised differentiable layer (stretch, time-boxed).** Use a deterministic-equivalent or at most 8 fixed-quantile scenarios, with a quadratic term (Wilder 2019) or perturbation (Berthet 2020), via cvxpylayers or PyEPO. Validate on a toy problem first.
5. **SPO+ (out).** Cite it as inapplicable to right-hand-side uncertainty. It is valid only for a price-prediction variant, which would be a different thesis.

**Core vs stretch.**
- Core: DE only; Regimes A and B; D1 and D2 (D3 if time allows); H1-H4; all test days; 5 seeds; non-anticipative settlement.
- Stretch: D4, drift on injected change points, and zero-shot transfer to one zone (DK1 or NL) under Regime A.

## Data plan

- **Main zone:** DE-LU.
- **Main period:** 2019-01-01 to 2025-09-30 (hourly DA throughout; ends before the 15-minute MTU).
- **Sources:**
  - ENTSO-E Transparency via pinned entsoe-py with a free token: actual generation 16.1.B/C (quarter-hourly), load 6.1.A, DA solar forecast 14.1.D, DA price 12.1.D.
  - Imbalance price: reBAP from netztransparenz.de (quarter-hourly, quality-assured), cross-checked against ENTSO-E 17.1.G.
- **Split:**
  - Train 2019-01 to 2022-12.
  - Validation and conformal calibration in 2023 H1 (separate blocks).
  - Test 2023-07 to 2025-09, with monthly rolling origins.
  - Report by regime: pre-crisis, crisis 2021-22, post-crisis.
- **Target framing:** a portfolio BRP holding a small share of the national fleet, as a price-taker. Run a sensitivity check on one TSO control-area series (e.g. TenneT DE or 50Hertz) to weaken the national error-price correlation.
- **Pipeline:** cache raw responses with a hash manifest and use the canonical column names (`solar_mw`, `load_mw`, `price_eur_mwh`, `imbalance_price_eur_mwh`). Add tests for UTC conventions, DST and quarter-hour completeness.
- **CS1 reproduction:** OPSD pinned to the 2020-10-06 release, 2015-2020, unchanged prosumer LP. Use it only to reproduce CS1.
- **Contrast zones (stretch only):** DK1 (Energinet, single price from 2021-11-01) and NL (TenneT). Drop CH. GB is optional via Elexon, in GBP.
- **Fallback.** If the ENTSO-E or reBAP rebuild is not done by the end of W8, stay on OPSD DE. Run Regime A on 2015-2020, and Regime B on DE-LU DA plus reBAP for 2018-10 to 2020-09 only (short, with COVID inside the test year, labelled as limited). Any zone without imbalance data is used for forecasting and portability only, under the stylised tariff, labelled as stylised.
- **Before any training:** report the empirical distribution of (λ_imb − λ_DA) and its sign-conditional means by year and hour. That is the real penalty ratio, and it tells you early whether Regime B has DFL headroom.

## Missing literature to add

**Value-oriented forecasting and DFL for renewable trading (lead references):**
- Carriere, T., & Kariniotakis, G. (2019). An integrated approach for value-oriented energy forecasting and data-driven decision-making: Application to renewable energy trading. *IEEE Transactions on Smart Grid*, 10(6), 6933-6944. doi:10.1109/TSG.2019.2914379
- Muñoz, M. A., Morales, J. M., & Pineda, S. (2020). Feature-driven improvement of renewable energy forecasting and trading. *IEEE Transactions on Power Systems*, 35(5), 3753-3763. doi:10.1109/TPWRS.2020.2975246
- Stratigakos, A., Camal, S., Michiorri, A., & Kariniotakis, G. (2022). Prescriptive trees for integrated forecasting and optimization applied in trading of renewable energy. *IEEE Transactions on Power Systems*, 37(6), 4696-4708. doi:10.1109/TPWRS.2022.3152667
- Stratigakos, A., Michiorri, A., & Kariniotakis, G. (2021). A value-oriented price forecasting approach to optimize trading of renewable generation. *2021 IEEE Madrid PowerTech*.
- Wahdany, D., Schmitt, C., & Cremer, J. L. (2023). More than accuracy: End-to-end wind power forecasting that optimises the energy system. *Electric Power Systems Research*, 221, 109384. doi:10.1016/j.epsr.2023.109384
- Zhang, Y., Jia, M., Wen, H., Bian, Y., & Shi, Y. (2024). Toward value-oriented renewable energy forecasting: An iterative learning approach. *IEEE Transactions on Smart Grid*. doi:10.1109/TSG.2024.3503554
- Zhang, Y., Wen, H., Bian, Y., & Shi, Y. (2024). Improving sequential market coordination via value-oriented renewable energy forecasting. arXiv:2405.09004.
- Zhang, J., Wang, Y., & Hug, G. (2022). Cost-oriented load forecasting. *Electric Power Systems Research*, 205, 107723. doi:10.1016/j.epsr.2021.107723
- Li, R., et al. (2024). Decision-oriented learning for future power system decision-making under uncertainty. arXiv:2401.03680. Use as the field review.

**Imbalance settlement and price uncertainty:**
- Zugno, M., Jónsson, T., & Pinson, P. (2013). Trading wind energy on the basis of probabilistic forecasts both of wind generation and of market quantities. *Wind Energy*, 16(6), 909-926.

**Newsvendor and contextual optimisation theory:**
- Gneiting, T. (2011). Quantiles as optimal point forecasts. *International Journal of Forecasting*, 27(2), 197-207.
- Ban, G.-Y., & Rudin, C. (2019). The big data newsvendor: Practical insights from machine learning. *Operations Research*, 67(1), 90-108.
- Bertsimas, D., & Kallus, N. (2020). From predictive to prescriptive analytics. *Management Science*, 66(3), 1025-1044.
- Sadana, U., et al. (2025). A survey of contextual optimization methods for decision-making under uncertainty. *European Journal of Operational Research*, 320(2), 271-289.

**Gradients through LPs and tooling:**
- Wilder, B., Dilkina, B., & Tambe, M. (2019). Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. *AAAI 33*, 1658-1665.
- Mandi, J., & Guns, T. (2020). Interior point solving for LP-based prediction+optimisation. *NeurIPS 33*.
- Berthet, Q., et al. (2020). Learning with differentiable perturbed optimizers. *NeurIPS 33*.
- Vlastelica, M., Paulus, A., Musil, V., Martius, G., & Rolínek, M. (2020). Differentiation of blackbox combinatorial solvers. *ICLR 2020*.
- Shah, S., Wang, K., Wilder, B., Perrault, A., & Tambe, M. (2022). Decision-focused learning without decision-making: Learning locally optimized decision losses. *NeurIPS 35*.
- Tang, B., & Khalil, E. B. (2024). PyEPO: A PyTorch-based end-to-end predict-then-optimize library for linear and integer programming. *Mathematical Programming Computation*, 16(3), 297-335. doi:10.1007/s12532-024-00255-x

**Storage-focused DFL (needed if the battery stays central):**
- Sang, L., Xu, Y., Long, H., Hu, Q., & Sun, H. Electricity price prediction for energy storage system arbitrage: A decision-focused approach. arXiv:2305.00362. Verify the IEEE TSG venue on Xplore.
- Smets & Delarue (2025), DFL price forecasting for storage in the day-ahead market (TU Delft repository). Verify the details.
- Tapia, Castellano, Mallada & Dvorkin, end-to-end storage arbitrage, arXiv:2512.06600.

**Methods for uncertain constraint parameters (if you discuss SPO+ alternatives):**
- Hu, Lee & Lee, Predict+Optimize with unknown constraint parameters (AAAI 2023; NeurIPS 2023). Verify the exact citations.

**Statistical inference:**
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217-1241.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453-497.

## Revised plan from week 7

| Week (approx. start) | Work | Deliverable / gate |
|---|---|---|
| W7 (16 Nov) | `git init` + tag `cs1-final` + lockfile. Decision-model note (C1), hypotheses H1-H4, evaluation protocol, scope freeze. Rewrite Sec. 2/4 motivation and gap. Check ENTSO-E token and reBAP download. | Supervisor sign-off in writing |
| W8 (23 Nov) | ENTSO-E + reBAP loaders, local-time gate, DST, quarter-hour settlement. Generator LP with bounds and price vectors, NaN-safe. Non-anticipative settlement option. Benchmark D2 problem size. | Data tests green. Fallback decision at end of W8. |
| W9 (30 Nov) | New fold design (train/val/calib/test, monthly origins). Stats module (DM/HAC, block bootstrap, Holm, MCS). Re-run B0-B6 in Regimes A and B, 5 seeds, all test days. Spread analysis of (λ_imb − λ_DA). | Baseline table with CIs. Methods chapter started. |
| W10-11 (7 Dec) | D1 τ_t-pinball (linear, MLP, LightGBM). Battery = 0 equivalence check (H2). | First H1/H2 result |
| W12-13 (21 Dec, holidays) | D2 linear decision ERM. Buffer for the holiday weeks. | Main DFL result on Regime A |
| W14 (4 Jan) | D2 on Regime B. Start D3 (dual subgradients). | Kill/continue decision for gradient arms |
| W15 (11 Jan) | D3 small MLP, 5 seeds | D3 result or documented negative result |
| W16-17 (18 Jan) | Penalty × storage grid, all days, trend tests (H3). Decision-level calibration audit + ACI (H4). | All core results |
| W18 (1 Feb) | Regime B robustness: shuffled-λ control, by-regime reporting, TSO control-area sensitivity | Robustness section |
| W19 (8 Feb) | Buffer, or one stretch item: D4 layer, drift, or DK1 transfer | Optional |
| W20-23 (15 Feb) | Writing: results, discussion, intro/related work, final figures from one `make` target | Full draft to supervisor by end of W22 |
| W24 (15 Mar) | Revisions, submission, defence prep | Submission |

## First two weeks: concrete next actions

1. [ ] Confirm the official start and submission dates with the examination office.
2. [ ] `git init`, commit, tag `cs1-final`. Generate a lockfile. Move `presentation/` media and `colab_bundle` out of the tree.
3. [ ] Write the one-page decision-model and evaluation note: generator/BRP, Regimes A and B, regret and oracle definition, H1-H4, primary endpoint, power calculation (MDE 3.2 EUR/day at n = 730). Send it to the supervisor with the open questions below.
4. [ ] Rewrite the exposé Sec. 2 motivation (r with CI and without Prophet, drop the "7% cheaper" inversion), the Sec. 4 gap paragraph (cite Carriere, Muñoz, Stratigakos, Wahdany, Zhang), and the SQ5 transfer wording. Fix the Yu et al. entry. This becomes the draft of thesis chapters 1-2.
5. [ ] Fix `newsvendor_quantile` (`policies.py:167-169`) to per-hour, per-regime τ_t. Correct `report.tex:545, 563, 771, 883`.
6. [ ] Get an ENTSO-E API token. Write `data/entsoe.py` and `data/imbalance.py` (reBAP) with caching and a hash manifest. Pin the OPSD URL to the 2020-10-06 release.
7. [ ] Switch the gate and delivery day to Europe/Berlin, support T ∈ {23, 24, 25}, and settle at quarter-hour level.
8. [ ] Rewrite the LP as generator plus battery: bounds on q, gp, gm, up and dn, price vectors as data, and `run_policy` raising on non-optimal status. Add tests with negative and single-price inputs.
9. [ ] Add non-anticipative settlement. Re-run the CS1 storage grid under anticipative and non-anticipative settlement and compare.
10. [ ] Replace `make_folds` with rolling origins and four blocks. Add a fold-isolation test and a decision-layer price-poisoning test. Fix the tautological assertion (`test_no_leakage.py:141-143`).
11. [ ] Build the stats module (DM/HAC, moving-block bootstrap, Holm). Seed scenarios per (date, model) with common random numbers. Put `--sensitivity` in `make all`.
12. [ ] Benchmark the D2 ERM LP size (one fold, battery 4 MWh) in HiGHS.

## Open questions for the supervisor

1. May the scope be narrowed to DE-LU, with drift (SQ4), transfer (SQ5) and multi-zone (Obj 4) moved to stretch or outlook, even though they are in the approved exposé? Does that need a formal amendment?
2. Generator/BRP framing or prosumer: which does the supervisor consider the thesis object? Is the two-regime design (stylised dual-price mechanism plus real single-price DE robustness) acceptable for Objective 3?
3. Is replacing SPO+ with linear decision ERM plus dual-subgradient training acceptable as the "full DFL" arm, with the differentiable layer as stretch?
4. Is a well-predicted null result (no DFL advantage at storage = 0 and over a tuned τ_t baseline) an acceptable main finding if H1 fails?
5. Is a national-aggregate target with a portfolio-BRP framing acceptable, or is a site-level or control-area series required?
6. What statistical standard does the supervisor expect: DM/HAC with block bootstrap and Holm, and how many seeds?
7. What are the confirmed start and submission dates, and is a full draft due before submission (proposed end of W22)?
8. May the CS1 report text and figures be reused in the thesis after the corrections, and how should overlap with the graded case study be declared?
