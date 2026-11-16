# How to record the "how it works" demo video

A walkthrough of the code and the pipeline, about **4 minutes**. This is a
different video from the results animation (`demo_decision_loop.mp4`). That one
shows *what the project found*. This one shows *how it is built*.

The real pipeline takes 45 minutes, so recording it live is not an option.
`demo_pipeline.py` runs the same code path on one slice of data and narrates
itself, in about 70 seconds at recording pace.

---

## Before you record

- Make the terminal font large, around 18pt. What is readable on your screen is
  too small in a video.
- Set the terminal to about 110 columns wide, so the demo's lines do not wrap.
- Close everything else. No notifications, no other tabs.
- Do a dry run first: `PYTHONPATH=src:. .venv/bin/python presentation/demo_pipeline.py --pace 0`
- Record with **Cmd+Shift+5** on macOS, or QuickTime, **File → New Screen Recording**.
- Record the whole thing in one take if you can. If you make a mistake, pause,
  and record that section again separately. You can cut the parts together later.

---

## Shot list

### Shot 1. The repository (0:00 to 0:25)
**On screen:** your editor, with the project tree open on the left.

> This is the project. The `src` folder holds the library, split by what each part
> does: data, models, evaluation, decision, drift and explainability. The
> `experiments` folder holds five scripts, one for each research question. And
> `tests` holds the checks that keep the whole thing honest.
>
> Let me show you the single most important piece of code in it.

### Shot 2. The gate (0:25 to 1:05)
**On screen:** open `src/riskaware_pv/data/features.py` and scroll to `_gated_shift`.

> In the day-ahead market, we commit at twelve o'clock for the whole of tomorrow.
> So the most obvious feature, the value from 24 hours ago, does not exist yet for
> any hour after midday. Using it would make the model look good in testing, but
> it could never work in reality.
>
> This function is the fix. Instead of a fixed lag, it works out the horizon, and
> then steps back whole days until it reaches a value that genuinely exists at the
> gate. The `hour_offset` is folded in before the rounding, and that detail matters.
> A neighbour-hour feature that adds one hour after rounding steps one hour past
> the gate, and that was a real bug in this project.

### Shot 3. The test that guards it (1:05 to 1:30)
**On screen:** open `tests/test_no_leakage.py`, at `test_features_ignore_everything_after_the_gate`.

> I did not want to rely on reading my own code, so this test attacks it. It takes
> the data, replaces everything after a chosen gate with a nonsense value, rebuilds
> all the features, and then checks that every earlier row is exactly the same as
> before.
>
> If any feature reached into the future, it would pick up the poison and change.
> This test found two real bugs.

### Shot 4. Run the core (1:30 to 2:50)
**On screen:** terminal, full width. Run:

```bash
PYTHONPATH=src:. .venv/bin/python presentation/demo_pipeline.py --pace 2.5
```

Mostly let it speak for itself. Say these four things as the matching stage appears:

> **At stage 2:** Notice there is no weather data anywhere. The sun's position is
> pure physics, so I calculate it instead of downloading it.
>
> **At stage 4:** Here it runs the real test suite, and it passes. Now watch this.
> It puts the old bug back on purpose, and the guard catches it immediately, naming
> the two features that were contaminated.
>
> **At stage 6:** This is interesting. On this one slice, calibration makes things
> worse. That is not a mistake in the demo. It calibrated on summer and tested on
> autumn, and those two seasons are not exchangeable. It is the same weakness I
> document in the report, showing up live.
>
> **At stage 7:** And here the forecast becomes a decision. The optimiser commits a
> schedule, reality arrives, and we pay the difference against a perfect forecast.

### Shot 5. What comes out (2:50 to 3:10)
**On screen:** `results/` folder, then `figures/`, scrolling slowly.

> Every stage writes its results to disk, so nothing has to be recomputed to redraw
> a figure. These 25 tables and 17 figures are what the report is built from. No
> number in the report is typed by hand.

### Shot 6. The whole suite (3:10 to 3:30)
**On screen:** terminal. Run `make test`.

> And this is the safety net: 22 tests. They cover the leakage rule, the economics
> of the optimiser, the scoring rules, and two bugs that were serious enough to be
> worth a permanent test each.
>
> The full pipeline reproduces with one command, `make all`, in about 45 minutes.

---

## Putting it together

Save each shot as its own file, then join them:

```bash
printf "file 'shot1.mov'\nfile 'shot2.mov'\nfile 'shot3.mov'\n" > parts.txt
ffmpeg -f concat -safe 0 -i parts.txt -c copy how_it_works.mp4
```

If the clips were recorded at different sizes, `-c copy` will fail. Re-encode instead:

```bash
ffmpeg -f concat -safe 0 -i parts.txt -vf scale=1920:1080 -c:v libx264 -crf 20 how_it_works.mp4
```

To put the results animation on the end:

```bash
printf "file 'how_it_works.mp4'\nfile 'demo_decision_loop.mp4'\n" > all.txt
ffmpeg -f concat -safe 0 -i all.txt -c copy full_demo.mp4
```
