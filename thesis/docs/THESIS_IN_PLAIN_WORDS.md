# My thesis in plain words

Title (official): *From Forecast Accuracy to Decision Value: Decision-Focused Learning for Day-Ahead Renewable Energy Commitment*

Title (honest): **Should a solar forecast be graded on accuracy, or on how much money it saves?**

---

## 1. The situation

Imagine you own a solar farm in Germany.

- **Every day at noon** you have to promise the grid how much electricity you will deliver **in each hour of tomorrow**. This promise is called the *commitment* or *schedule*.
- Tomorrow comes. The sun does what it wants.
- **Promised too much and delivered less?** You are *short*. You pay an expensive penalty.
- **Delivered more than you promised?** You are *long*. The extra is bought for very little.

Being short costs more than being long, so the smart move is **not** to promise your best guess. You promise **a bit less** than your best guess. How much less depends on how harsh the penalty is.

If you also have a **battery**, you can cover a shortfall from it. That changes how much you should promise.

## 2. How everyone does it today (two separate steps)

1. **Forecast:** train a model to predict tomorrow's solar output as **accurately** as possible.
2. **Decide:** hand that forecast to an optimiser, a maths program that picks the promise that costs the least.

The forecaster never learns what its mistakes **cost**. It only learns how **wrong** it was.

## 3. What Case Study 1 found

I built exactly this two-step pipeline on real German solar data, then checked how much money each forecast actually lost when you acted on it.

- More accurate forecasts were **usually** cheaper, but **not always**.
- Using the full range of possible outcomes ("probably 50 MW, but between 30 and 70") instead of one number saved anywhere from **3% to 67%** of the avoidable cost, depending on the penalty and battery size.
- Simply promising a well-chosen "a bit less than the best guess" level was one of the cheapest strategies overall.

In short: **accuracy and money are related, but they are not the same thing.**

## 4. What the thesis does

Train the forecaster with **money as its score** instead of accuracy.

> "Learn from your euro bill, not from your error."

This idea is called **decision-focused learning (DFL)**. The model is trained so that its forecasts produce the cheapest decisions, even if they are slightly less accurate.

## 5. The questions, simply

1. **Does it save money?** Compare the money-trained model with the best normal approach.
2. **When does it help?** With a small or large battery, and with a mild or harsh penalty.
3. **What does it break?** A money-trained forecast may stop being *honest* about uncertainty: its "90% range" might not contain the truth 90% of the time. Measure that.

## 6. The twist that makes this interesting

With **no battery**, there is a well-known shortcut. You train a normal forecaster to predict the right "a bit less" level directly. This is called a *quantile*, and the training loss is the *pinball loss*. That shortcut is **already** the perfect money-trained model.

So the expected result is:

- **No battery → no gain from DFL.** This is a good sanity check: if the code says otherwise, something is wrong.
- **With a battery → DFL might win**, because hours are now linked: what you promise at 10:00 affects what the battery can do at 15:00. The simple shortcut cannot see that.

That gives the thesis a clear, testable story: **DFL only pays off when storage links the hours together. How much does it pay?**

If DFL turns out not to beat the shortcut, that is still a valid result: "for this market, the simple trick is enough."

## 7. The plan: everything in the exposé, each part kept as small as it can be

The scope is the approved exposé. Nothing gets dropped. What keeps it manageable is
doing every item in its **simplest acceptable form**, and noticing that Case Study 1
already contains the code for half of it.

| Exposé item | Simplest acceptable version | Reuses Case Study 1? |
|---|---|---|
| Real prices (Obj. 3) | Germany: real day-ahead price (ENTSO-E) + real imbalance price (reBAP). Other zones keep the synthetic tariff, which Sec. 5 of the exposé already allows, and the limitation is reported. | Partly (data layer) |
| Several zones (Obj. 4) | Germany plus two zones already in the data (e.g. Denmark, Switzerland) | Yes (`transfer.py`, multi-zone loader) |
| Penalty x storage grid (Obj. 4) | A small grid: 3 penalty levels x 3 battery sizes | Yes (`e2 --sensitivity`) |
| Baselines | Ridge point forecast, ridge quantile + conformal, perfect-foresight oracle | Yes |
| DFL method 1: cost-weighted quantile loss | Pinball loss at the cost-optimal level, weighted by what each hour's mistake costs | New, but easy |
| DFL method 2: SPO+ | Implemented in the form that fits this problem. The textbook SPO+ assumes the forecast is a *price*, and here it is *solar output*. Finding and testing the right adaptation is part of answering sub-question 2. If it only works partly, that is a reportable result. | New |
| DFL method 3: differentiable layer (Obj. 1) | The dispatch LP rebuilt in `cvxpylayers` with a small smoothing term, a single scenario (deterministic version) and a small model. Time-boxed to two weeks. | New |
| Calibration audit + conformal on top (Obj. 5) | Reuse the conformal and adaptive conformal code on the decision-trained models | Yes (`models/conformal.py`) |
| Drift (sub-question 4) | Rerun the Case Study 1 drift experiment with a decision-trained model | Yes (`drift/`, `e3`) |
| Transfer (sub-question 5) | Rerun the Case Study 1 transfer experiment with a decision-trained model | Yes (`transfer.py`, `e5`) |
| Reproducible release (Obj. 6) | GitHub repo, tests, `make all` | Already done |

Quality fixes from the review still apply. These don't change the scope, they make the results hold up:

- **Dispatch model:** the dispatch LP must survive real prices (bounds on every variable, no silently dropped days).
- **Statistics:** compare on every test day, use a few random seeds, and use a paired significance test.
- **Exposé wording:** don't overstate the Case Study 1 numbers when they are reused in the thesis text.

## 8. Timeline (today is week 7 of 24)

Weeks 1 to 6 of the exposé plan slipped, so the remaining work is compressed. The order stays the same as the exposé.

| Weeks | Do | Done when |
|---|---|---|
| 7-8 | Data: ENTSO-E prices + reBAP for Germany. Make the dispatch LP safe for real prices. | Price data loaded, tests pass |
| 9 | Re-run the two-stage baselines on the new data (all zones, all test days) | Baseline results table |
| 10-11 | DFL 1: cost-weighted quantile loss | First DFL vs baseline comparison |
| 12-13 | DFL 2: SPO+ (adapted) | SPO+ results, or a documented reason it falls short |
| 14-15 | DFL 3: differentiable layer (time-boxed) | End-to-end model results |
| 16 | Calibration audit + conformal on top of the DFL models | Cost vs honesty trade-off table |
| 17 | Drift + transfer reruns with the best DFL model | Robustness and portability results |
| 18 | Penalty x storage grid + statistics on everything | Final results |
| 19 | Buffer (something will be late) | |
| 20-24 | Write, revise, submit, prepare the defence | Submitted thesis |

The rule that protects the thesis is the one the exposé already states: each phase produces a
result on its own. If the differentiable layer misbehaves in weeks 14-15, stop on time, report
what happened, and move on. A documented "this was unstable, here is why" still answers the
question.

## 9. Words you will hear (cheat sheet)

| Word | Meaning |
|---|---|
| Day-ahead market | Where you promise tomorrow's delivery, today at noon |
| Commitment / schedule | The promise: MW per hour for tomorrow |
| Imbalance | The difference between what you promised and what you delivered |
| Imbalance price / penalty | What that difference costs you |
| Regret | Your cost minus the cost of someone who knew the future. Lower is better. |
| Quantile | "The level that the truth stays below X% of the time" |
| Pinball loss | The training score that teaches a model to predict a quantile |
| CRPS | A score for how good a whole forecast range is |
| Calibration | Whether the stated "90% range" really holds 90% of the time |
| Conformal prediction | A cheap add-on that repairs calibration |
| DFL (decision-focused learning) | Training a forecaster on the cost of the decision, not on accuracy |
| Two-stage | Forecast first, then decide separately (the normal way) |
| Linear program (LP) | The maths problem the optimiser solves to pick the cheapest promise |
| Leakage | Accidentally letting the model see the future. Cheating, even if unintentional. |

The detailed technical review is in [expose_review.md](expose_review.md). This file is the one to re-read when you are lost.
