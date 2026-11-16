# Recording the Colab demo, with no voiceover

The notebook's text does the explaining, so your job is to **run it, then scroll
slowly enough that a viewer can read**. That one idea drives everything below.

Two files are needed. Both are in this folder:

- `Case_Study_1_Demo.ipynb`
- `colab_bundle.zip`  (4.4 MB)

---

## Part 1. Get it running in Colab

1. Go to **colab.research.google.com**. Sign in with your Google account.
2. A box appears. Click the **Upload** tab, then **Browse**, and choose
   `Case_Study_1_Demo.ipynb`.
   If no box appears, use **File → Upload notebook**.
3. The notebook opens. Click **Connect** at the top right and wait for the green tick.
4. Run the **first code cell only**: click it, then press **Shift+Enter**.
5. It prints `Choose colab_bundle.zip ...` and shows a **Choose Files** button.
   Click it and pick `colab_bundle.zip`. Wait for the upload to finish.
6. When it prints `ready · gate hour 12 · quantiles [...]`, the setup worked.
7. Now run everything: **Runtime → Run all**. It takes about a minute.
8. Scroll to the bottom and check the last code cell says `21 passed`.

**If step 5 fails**, the most common cause is picking the notebook file again
instead of the zip. Re-run the cell and choose the `.zip`.

**If a cell errors**, use **Runtime → Restart session**, then **Run all** again.
The upload only has to happen once per session.

---

## Part 2. Make the browser ready for recording

Do this **after** the notebook has finished running, and before you record.

1. **Zoom in.** Press **Cmd and +** two or three times, until the body text is
   clearly readable. Around 150% is right. Text that looks fine to you now will be
   too small in the finished video.
2. **Hide the bookmarks bar**: **Cmd+Shift+B**.
3. **Keep the left sidebar open** in Colab. Click the bullet-list icon on the far
   left to show the **Table of contents**. In a silent video this is valuable: it
   shows the viewer where they are.
4. Close every other browser tab. Tabs from other sites look untidy.
5. Turn on Do Not Disturb: click the clock in the top right of the Mac menu bar,
   then **Focus → Do Not Disturb**.
6. Scroll back to the very top of the notebook.

---

## Part 3. Record

1. Press **Cmd+Shift+5**.
2. Choose **Record Selected Portion** and drag a box around the browser window.
   Leave out the Mac menu bar and the dock.
3. Click **Options**. Set **Microphone** to **None**, since there is no voiceover.
4. Click **Record**.
5. Stop with the small stop icon in the menu bar, or **Cmd+Ctrl+Esc**.
6. The file appears on your Desktop as a `.mov`.

---

## Part 4. What to do while recording

This is the whole technique: **scroll slowly, and stop at every heading.**

Use two fingers on the trackpad, or the arrow keys for a steadier speed. Do not
use the scroll wheel in big jumps, it is hard to read and looks rushed.

At every section heading, **stop scrolling and count to five in your head** before
moving on. If a block of text is long, count to eight. You have already read this
notebook, so it will feel far too slow to you. It will be about right for someone
seeing it for the first time.

Suggested pauses:

| Where | Stop and count |
|---|---|
| Title, at the very top | 6 |
| Section 1, the data | 5 |
| Section 2, the clear-sky plot | 6, let them look at the curve |
| Section 3, the gate, and the code block | 8, this one is dense |
| Section 4, the guard, both cells | 8, this is the most important moment |
| Section 5, the models | 5 |
| Section 6, the calibration table | 8, the result is surprising |
| Section 6, the fan chart | 6 |
| Section 7, the decision tables | 8 |
| Section 8, the three other questions | 10, there are four tables |
| Section 9 and the conclusion | 8 |

That gives roughly **five to six minutes**. That is a sensible length for a silent
read-along video.

### Two moments worth pausing on longer

**Section 4, the guard.** The first cell prints `none. The gate holds.` The next
one puts the original bug back and prints `CAUGHT: 2 features changed`. Stop on
that output for a good eight seconds. It is the single best thing in the video.

**Section 6, the calibration table.** The coverage gets *worse*, and the text
underneath explains why. Give the viewer time to read the number, be puzzled, and
then read the explanation. Rushing this makes it look like a mistake.

---

## Part 5. Optional, a title at the start

A silent video benefits from a still title for the first few seconds.

Easiest way: before recording, add a text cell at the very top of the notebook.
Click the first cell, press **Cmd+M then A** to insert a cell above, click
**+ Text**, and paste:

```
# Case Study 1, Topic 8: AI for Clean Energy
### Probabilistic day-ahead solar forecasting and risk-aware battery dispatch
Devendra Singh Dhakad · 100004684 · Supervisor: Prof. Dr. Mehrdad Jalali
```

Hold on it for six seconds at the start of the recording.

---

## Part 6. Tidy the file afterwards

The recording is usually fine as it is. Two things you may want:

**Trim the dead space** at the start and end. Open the `.mov` in QuickTime, then
**Edit → Trim**, drag the yellow handles, click **Trim**, then **File → Save**.

**Make the file smaller** if it is large, since screen recordings often are:

```bash
ffmpeg -i ~/Desktop/recording.mov -vf scale=1920:-2 -c:v libx264 -crf 23 -an colab_demo.mp4
```

The `-an` drops the audio track, which is empty anyway.

**Speed up a slow section**, for example if the Run all took too long on camera:

```bash
ffmpeg -i colab_demo.mp4 -filter:v "setpts=0.5*PTS" -an colab_demo_fast.mp4
```

`0.5` means twice as fast.

---

## Checklist before you press record

- [ ] Notebook fully run, last cell shows `21 passed`
- [ ] Browser zoomed to about 150%
- [ ] Bookmarks bar hidden, other tabs closed
- [ ] Table of contents sidebar open
- [ ] Do Not Disturb on
- [ ] **Microphone set to None** in the Cmd+Shift+5 options
- [ ] Scrolled back to the very top
