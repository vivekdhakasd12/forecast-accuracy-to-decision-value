# Speaker script: Case Study 1 presentation

Target length **about 13 to 14 minutes** for 13 slides, at a calm pace of roughly
130 words per minute. The time after each slide title is a guide, not a rule.

This file is also the source of the speaker notes inside the .pptx. Edit it here,
then rebuild the deck with `.venv/bin/python presentation/build_slides.py`, and the
notes update to match.

**Delivery tips**

- Do not read the slides aloud. The slide shows the number; you explain why it matters.
- Pause for one second after each big orange number. Let it land.
- When a slide has a chart, point at the part you are talking about.
- If you are short of time, the easiest slides to shorten are 7 and 9.

---

## Slide 1 · Title (0:20)

Hello, everyone. My name is Devendra Singh Dhakad. Today I will present my case study for Topic 8, AI for Clean Energy.

The title is: Probabilistic Day-Ahead Solar Forecasting and Risk-Aware Battery Dispatch under Concept Drift. My supervisor is Professor Jalali.

## Slide 2 · Why the mean is the wrong answer (1:15)

Let me start with the problem.

In the European day-ahead market, a solar site has to make a promise one day early. At twelve o'clock today, it must commit to a schedule for every hour of tomorrow. So the forecast horizon is between 12 and 35 hours.

If the site delivers less than it promised, it pays 300 euros per megawatt-hour. If it delivers more, it gets back only 90 euros. So being short is more than three times as expensive as being long.

This asymmetry changes the right answer. When the two mistakes have different costs, the best commitment is not the average forecast. It is a quantile of the forecast distribution.

A normal point forecast cannot give you that, because it gives only one number. The decision needs the whole shape.

That is why this study measures every forecast twice: once with the usual scoring rules, and once in euros.

## Slide 3 · Five research questions (0:45)

So I asked five research questions.

First: how good can a forecast be without any weather data, and are its uncertainty ranges honest?

Second: does a better forecast actually lead to a better decision, measured in euros?

Third: when the data changes over time, is it better to retrain when a detector sees a change, or on a fixed calendar?

Fourth: what does the model really depend on?

And fifth: a new solar site has only a few days of data. Should it train its own model, or borrow one from another country?

## Slide 4 · Data and the day-ahead protocol (1:00)

For the data, I used Open Power System Data. It is a free, open dataset of hourly German solar generation. After cleaning, I had about 33,000 hourly records. That is 1,383 complete delivery days, from 2016 to 2020.

One important choice: I used no weather forecasts at all. A small operator often cannot pay for a weather feed. But the part of solar power that we can predict exactly is astronomy: the position of the sun. And astronomy is free.

So I calculate the sun's position and the clear-sky power with physics formulas. The machine learning only has to learn what is left, which is mostly the clouds.

The diagram shows the timeline. Everything to the left of the orange line is data we are allowed to use. Nothing to the right may be used as an input.

## Slide 5 · Pipeline, and the one safety property (1:15)

On the left you can see the full pipeline, from the raw data at the top to the results at the bottom.

The most important rule in the whole system is the orange box: no feature may look past the gate.

This is easy to get wrong. The most obvious feature for solar forecasting is the value from exactly 24 hours ago. But with a noon gate, that value does not exist yet for any hour after midday. If you use it, the model looks very good in testing, but it could never work in real life.

So I do not use any fixed lag. Each feature goes back to the most recent value that really exists at the gate.

And I did not just trust my code. I wrote a test that replaces all the data after the gate with nonsense values, and then checks that the features do not change at all. This test found two real bugs that a normal code review had missed.

## Slide 6 · RQ1: sharp intervals are not valid intervals (1:15)

Now the results. The first finding is about uncertainty.

A good forecast should be able to say: I am 90 percent sure the real value will be inside this range. I tested whether that is actually true.

For one model, the answer was very bad. Its 90 percent range contained the real value only 22.5 percent of the time.

What makes this dangerous is that the same model also had the narrowest ranges of all. So it looked the most precise, but it was the most wrong. Only the interval score shows this problem. Coverage alone does not.

The fix is called conformal calibration. It adjusts the ranges using data the model has not seen, and it needs no assumptions about the distribution. For this model, coverage rose from 22 to 63 percent, and the interval score improved by 57 percent.

In the chart, each arrow is one model, before and after calibration. They all move up, towards the 90 percent line.

## Slide 7 · RQ1: the same failure, up close (0:45)

This is what that problem looks like in practice. These are five real days. The black line is what actually happened. The blue bands are the forecast ranges.

On the first day, the real output was higher than the range. On the fourth day, an unexpected cloudy afternoon pushed it far below.

Two misses in five days is clearly worse than the one in ten that a 90 percent range promises.

This slide also shows the limit of working without weather data. The model knows the shape of tomorrow very well. It does not know tomorrow's weather.

## Slide 8 · RQ2: forecast skill against decision value (1:15)

The second question: does a better forecast give a better decision?

To test this, I feed every forecast into an optimisation model of a solar site with a battery. It fixes the schedule the day before, and then pays for its mistakes when the real sun comes out. I compare that cost with the cost of a perfect forecast. The difference is called regret.

The chart shows forecast quality on one axis and cost on the other. The correlation is 0.884, which is high. So a better forecast usually does help.

But not always. Some pairs go the other way. One model is 7 percent worse on forecast score, but it is cheaper to use.

I want to be honest about this number too. It is high partly because Prophet is so bad on both axes that it stretches the scale.

The table shows one more surprise. The careful, risk-averse strategies lost money. They reduced the shortfalls, but they paid more for that than they saved.

## Slide 9 · RQ2: when the distribution is worth having (1:00)

At the standard settings, using the full forecast distribution saved only about 3 percent. So I asked: when is it worth more?

The answer depends on two things: how expensive the penalty is, and how big the battery is. Across this grid, the benefit goes from 3 percent all the way up to 67 percent. So a single number for it means nothing unless you state both settings.

Here I was also wrong about something. I expected a big battery to replace a good forecast, because a battery can fix mistakes afterwards. At high penalties, the opposite is true. A bigger battery makes the forecast more valuable, from 18 to 35 percent.

The reason is that the battery gives the optimiser room to act on the forecast. So storage and good forecasting work together. They are complements, not substitutes.

## Slide 10 · RQ3: drift, retraining, and conditional validity (1:15)

The third question is about change over time.

On the left, I compare three retraining strategies with the same budget. Retraining when a drift detector fires gave the best accuracy, and it needed only two retrains. The calendar strategy needed seven.

But at first, the detectors did not work at all. They found zero changes. The reason was scale. These detectors look at absolute differences, and my error values are very small, around 0.04. I tested this directly: the detector sees a jump from 0 to 10, but it misses a jump from 0.02 to 0.06, even though that is three times bigger. After I standardised the data, it worked.

On the right is the same idea for uncertainty. Normal conformal calibration was correct on average over the whole year. But in its worst month, it covered only 83 percent. The adaptive version raised the worst month to 88 percent, and its ranges were even narrower.

An operator does not live the average of the year. They live one month at a time.

## Slide 11 · RQ4: attribution is not necessity (1:00)

The fourth question: what does the model really depend on?

I used two methods. SHAP tells us how much the model uses each group of features. Ablation removes one group completely, retrains the model, and measures the damage.

The two methods give almost opposite answers. According to SHAP, solar geometry is the most important group, with 67 percent. But when I removed it completely, the model did not get worse. It got slightly better. The reason is that the same information is also inside the calendar and history features.

The recent cloud information has only 8.5 percent in SHAP. But it is the one group the model cannot live without.

So SHAP tells you what the model used. It does not tell you what the problem needs. If you want to decide which data to collect, you need ablation.

## Slide 12 · RQ5: a new site with days of history (1:15)

The last question. Imagine a new solar site. It has only a few days of data, but it needs a forecast tomorrow.

I trained a model on Germany and tested it on Switzerland, Denmark and Great Britain. Together they cover nine degrees of latitude.

The red dashed line is the German model used directly, with no local data at all. The blue line is a model trained only on local data. The local model needed 120 to 240 days before it could beat the borrowed German model.

The best strategy is the orange line: combine the German data with the local data. With only 7 days of local data, it beat a local model trained on 480 days, at two of the three sites.

This works because of how the features are built. The target is normalised, and the sun's position is recalculated for each site. So what transfers is how the weather affects the output, not anything specific to Germany.

## Slide 13 · What the study concludes (1:15)

To conclude, five lessons.

First: check your uncertainty ranges before you trust them. They can be very wrong, and calibration fixes this cheaply.

Second: being correct on average is not enough. The worst month matters.

Third: judge a forecast by the decision it supports. Its value was between 3 and 67 percent, and the most careful strategy actually lost money.

Fourth: a new site should borrow before it collects. And storage and forecasting work together.

And fifth: simple baselines matter. One of my models was silently receiving empty input for every test case. It still produced normal-looking forecasts. I found the problem only because it lost to a simple baseline.

The limitations are in the box and in the report. The biggest one is that I used no weather forecasts, and that limits the accuracy.

Thank you. I am happy to take your questions.

---

## Likely questions, with short answers

These are the questions I would expect from an examiner, based on what the report
itself admits. Keep each answer short, then stop and let them ask the next one.

**"Why does a simple linear model beat gradient boosting and the LSTM?"**
The features already do most of the work. Physics gives the daily shape, the history gives persistence, and the cloud statistics give the current state. What is left is tomorrow's clouds, and no model can see those without weather data. Also, with only three test folds, the top models are a group, not a clear ranking.

**"Why didn't you use weather forecasts?"**
It was a deliberate choice. A small operator often cannot buy them, and I wanted to show what is possible without them. Weather forecasts would definitely improve accuracy. That is the first item in my future work.

**"How do you know there is no data leakage?"**
I did not rely on reading the code. The test replaces all data after the gate with nonsense values and checks that the features stay exactly the same. It found two real bugs, and both are fixed.

**"Isn't the correlation of 0.884 inflated?"**
Yes, partly, and the report says so. Prophet is very bad on both axes and stretches the scale. Inside the set, some pairs still go the opposite way, so a forecast score cannot fully replace measuring the decision.

**"Why only three folds?"**
The clean data covers about four years, and each fold needs a year of training, a calibration period and a test year. I report the standard error across folds, and I treat the top models as a cluster because of it.

**"The brief mentions TCN, TFT and metaheuristics. Why are they missing?"**
TCN and TFT are real gaps. My LSTM only matched the simple baseline, so a larger sequence model on this amount of data would probably not help much, but I did not prove that. Metaheuristics were a deliberate choice: the dispatch problem is a linear program, and the solver finds the exact best answer in under a second. A metaheuristic could only approximate it.

**"Are the euro figures real market prices?"**
No. They come from a realistic tariff that I defined. They show the size of the effect and how it works, not the exact money a real site would make.

**"Why was the stochastic strategy only 3 percent better?"**
That is only at the standard settings. The sensitivity study shows it ranges from 3 to 67 percent, depending on the penalty and the battery size.

**"Why normalise by the 99.9th percentile in the transfer study, and not by capacity?"**
Only Germany's capacity data matches its generation data. Switzerland, Denmark and Great Britain report more generation than their installed capacity, which is physically impossible. So their capacity numbers cannot be used as a denominator.

**"What would you do next?"**
Add weather forecasts. Add uncertainty in load and price. Use real market prices. Test on single sites instead of a whole country. And run the transfer study across all European countries, to find which sites are best to borrow from.

---

## Voiceover for the demo video (75 seconds)

Speak this over `demo_decision_loop.mp4`. It is about 150 words, which is roughly
70 seconds, so the timings below leave you a little silence. Do not rush to fill it.
Letting the animation run on its own for a few seconds is good.

**0:00** Title card. Say nothing. Let it sit.

**0:05** This is the decision the project is really about, repeated once a day.

**0:14** On the left is tomorrow's forecast. It is a distribution, not a single line. The dark band is the 50 percent range, the pale band is 90 percent.

**0:26** The black line is what actually happened, arriving hour by hour. Some days it stays inside the bands. Some days it does not.

**0:38** On the right is the bill. This is the extra cost against a perfect forecast. Four strategies pay four different amounts for the same weather.

**0:50** At the bottom is the running total. Single days are noisy. Over many days, the strategies separate.

**1:00** The green line commits against the whole distribution. It stays below the orange one, which commits against a single number. Both stay below the two risk-averse strategies, which cut the shortfalls and still pay more.

**1:11** End card. After 80 days, that is the gap. The same result as in the report, replayed one day at a time.
