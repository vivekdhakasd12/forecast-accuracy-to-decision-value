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

## 7. The simple version of the plan

Keep it small. Germany only, solar only.

| Step | What | In plain words |
|---|---|---|
| A | **Baseline 1: normal forecast** | Case Study 1's best model (ridge regression), promise = best guess |
| B | **Baseline 2: smart shortcut** | Same model, trained to predict the "a bit less" level (quantile) |
| C | **Money-trained model** | Train directly on the euro cost, including the battery |
| D | **Compare** | Across a few battery sizes and penalty levels, on many test days, with proper statistics |
| E | **Honesty check** | Is the money-trained model still well calibrated? Can a conformal correction fix it? |
| F | **Bonus (only if time)** | Real German market prices, other countries, the fancy "differentiable layer" method |

What gets dropped or made optional compared to the exposé:

- **SPO+:** it does not fit this problem, because the solar forecast is not a price.
- **The differentiable optimisation layer:** complicated and fragile. Bonus only.
- **Drift and other countries:** bonus only.

These cuts need a short email to Prof. Jalali. A narrower scope is normal, but it should be agreed in writing.

## 8. Timeline (today is week 7 of 24)

| Weeks | Do |
|---|---|
| 7-8 | Git and GitHub set up. Email the supervisor about the simpler scope. Fix the small dispatch-model issues found in the review. |
| 9-10 | Baselines A and B, with proper statistics (many test days, several random seeds) |
| 11-14 | Money-trained model C (start with a linear model, which is the simplest) |
| 15-17 | Comparison D and honesty check E |
| 18-19 | Bonus F, or buffer if things are late |
| 20-24 | Write, revise, submit, prepare the defence |

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
