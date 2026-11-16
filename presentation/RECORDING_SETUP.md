# Setting up VS Code and recording the demo

Step by step, from opening the folder to exporting the video.

---

## Part 1. Open the project in VS Code

1. Open VS Code. **File → Open Folder**, choose `case-study-1`, click Open.
2. If a bar appears saying "Do you trust the authors", click **Yes, I trust the authors**.
3. Install the Python extension if you do not have it. Click the Extensions icon in
   the left bar (four squares), search `Python`, install the one by Microsoft.
4. Point VS Code at the project's virtual environment, or nothing will run:
   - Press **Cmd+Shift+P**, type `Python: Select Interpreter`, press Enter.
   - Choose the one ending in `.venv/bin/python`. If it is not listed, click
     **Enter interpreter path** and paste `.venv/bin/python`.
   - The bottom right of the window should now show `.venv`.
5. Open a terminal inside VS Code with **Ctrl+`** (the key above Tab).
6. Check it works:

   ```bash
   PYTHONPATH=src:. .venv/bin/python -m pytest tests -q
   ```

   You should see `22 passed`.

---

## Part 2. Make VS Code look good on camera

Your normal settings are too small to read in a video. Change these before recording.

1. **Cmd+Shift+P**, type `Preferences: Open Settings (UI)`, Enter.
2. Set these, searching for each by name:

   | Setting | Value | Why |
   |---|---|---|
   | `Editor: Font Size` | `18` | readable when the video is scaled down |
   | `Terminal › Integrated: Font Size` | `17` | same |
   | `Editor: Minimap Enabled` | off | removes clutter on the right |
   | `Breadcrumbs: Enabled` | off | removes the path strip |
   | `Editor: Render Whitespace` | `none` | cleaner |
   | `Workbench: Color Theme` | `Light+` or `Default Dark+` | either, but stay consistent |

3. Press **Cmd+B** to hide the sidebar when you are not pointing at the file tree.
4. Zen Mode (**Cmd+K** then **Z**) removes everything except the editor. Useful for
   code close-ups. Press Escape twice to leave.
5. Close every tab you do not need. Open tabs from other projects look untidy and
   are the most common thing people forget.

---

## Part 3. Prepare the screen

- Quit Slack, Mail, WhatsApp and anything else that can pop up a notification.
- Turn on Do Not Disturb: click the clock in the top right, then **Focus → Do Not Disturb**.
- Hide desktop icons if your desktop is messy. In the terminal:

  ```bash
  defaults write com.apple.finder CreateDesktop false && killall Finder
  ```

  Put them back afterwards with `true` instead of `false`.
- Set your display to 1920x1080 if you can, so the recording is exactly 1080p.

---

## Part 4. Record

macOS has a built-in recorder. You do not need any extra software.

1. Press **Cmd+Shift+5**.
2. Choose **Record Selected Portion** and drag a box around the VS Code window,
   or **Record Entire Screen** if your display is already 1080p.
3. Click **Options** and set **Microphone** to your microphone. If you leave this
   on None you will record silent video.
4. Click **Record**. Stop from the menu bar icon, or **Cmd+Ctrl+Esc**.
5. The file lands on your Desktop as a `.mov`.

**Use a headset microphone if you have one.** Laptop microphones pick up the fan
and the room. Record in the quietest room you have, with the door shut.

Record each section as a separate file. If you fumble a sentence you only redo
that section, not the whole video.

---

## Part 5. What to record, in order

Full narration for every shot is in `demo_video_guide.md`. This is the shot list.

| # | On screen | Say | Time |
|---|---|---|---|
| 1 | File tree, `PROJECT_TOUR.md` open | the folder layout, what goes where | 0:25 |
| 2 | `src/riskaware_pv/data/features.py`, at `_gated_shift` | the gate, and why a fixed 24 hour lag is illegal | 0:40 |
| 3 | `tests/test_no_leakage.py` | the test that poisons the future and checks nothing moved | 0:25 |
| 4 | Terminal, run the demo below | let it narrate itself, add four comments | 1:20 |
| 5 | `figures/` and `results/` folders | everything the report is built from | 0:20 |
| 6 | Terminal, `make test` | 22 tests, then `make all` reproduces everything | 0:20 |

The command for shot 4:

```bash
PYTHONPATH=src:. .venv/bin/python presentation/demo_pipeline.py --pace 2.5
```

Optionally finish with `demo_decision_loop.mp4`, the 75 second results animation.
Its voiceover is at the end of `speaker_script.md`.

---

## Part 6. Join the clips

Rename your recordings `shot1.mov`, `shot2.mov` and so on, put them in one folder,
then:

```bash
printf "file 'shot1.mov'\nfile 'shot2.mov'\nfile 'shot3.mov'\nfile 'shot4.mov'\nfile 'shot5.mov'\nfile 'shot6.mov'\n" > parts.txt
ffmpeg -f concat -safe 0 -i parts.txt -c copy how_it_works.mp4
```

If that fails because the clips are different sizes, re-encode instead:

```bash
ffmpeg -f concat -safe 0 -i parts.txt -vf scale=1920:1080 -c:v libx264 -crf 20 -c:a aac how_it_works.mp4
```

To put the results animation on the end:

```bash
printf "file 'how_it_works.mp4'\nfile 'demo_decision_loop.mp4'\n" > all.txt
ffmpeg -f concat -safe 0 -i all.txt -c copy full_demo.mp4
```

Check the result plays and the audio is in sync before you submit it.

---

## Before you hit record, a checklist

- [ ] `.venv` selected as the interpreter, bottom right of VS Code
- [ ] Font sizes raised, minimap and breadcrumbs off
- [ ] Only project tabs open
- [ ] Do Not Disturb on
- [ ] Microphone selected in the Cmd+Shift+5 Options menu
- [ ] Dry run of `demo_pipeline.py --pace 0` completed without errors
- [ ] You have read `demo_video_guide.md` once, out loud
