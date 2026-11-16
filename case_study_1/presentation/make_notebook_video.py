"""Render the demo notebook as a silent walkthrough video.

A hand-made screen recording of a notebook is hard to get right: the scrolling is
uneven, the cursor wanders, and the pace is set by whoever is holding the
trackpad, who has already read the thing. This renders the same walkthrough
deterministically instead.

It executes the notebook, exports it to HTML, screenshots the whole page as one
tall image with headless Chrome, and then steps down that image one screen at a
time. Each screen is held for a duration proportional to how much text is on it,
so dense sections get more reading time than sparse ones.

    .venv/bin/python presentation/make_notebook_video.py
    .venv/bin/python presentation/make_notebook_video.py --minutes 4 --skip-execute

Output: presentation/demo_notebook_walkthrough.mp4 (1920x1080, 25 fps, silent)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "case_study_1" / "presentation" / "Case_Study_1_Demo.ipynb"
OUT = ROOT / "case_study_1" / "presentation" / "demo_notebook_walkthrough.mp4"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

W, H, FPS = 1920, 1080, 25
OVERLAP = 130          # px of the previous screen kept, so nothing is cut mid-line
TRANSITION = 0.55      # seconds of scroll between one screen and the next
HEAD_TAIL = 2.5        # seconds held on the first and last screen, on top of its own hold


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def execute_notebook(tmp: Path) -> Path:
    import nbformat
    from nbclient import NotebookClient

    print("executing the notebook ...")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="cs1",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    out = tmp / "executed.ipynb"
    nbformat.write(nb, out)
    return out


def to_html(ipynb: Path, tmp: Path) -> Path:
    print("exporting to HTML ...")
    run([sys.executable, "-m", "nbconvert", "--to", "html", "--template", "classic",
         "--output-dir", str(tmp), "--output", "page", str(ipynb)])
    return tmp / "page.html"


def screenshot(html: Path, tmp: Path) -> Image.Image:
    """One tall screenshot of the whole page, at video width."""
    print("screenshotting the page with headless Chrome ...")
    shot = tmp / "tall.png"
    # A device scale factor renders text at video width crisply, rather than
    # capturing narrow and upscaling it afterwards, which goes soft.
    scale = W / 1500
    subprocess.run(
        [str(CHROME), "--headless", "--disable-gpu", "--hide-scrollbars",
         f"--force-device-scale-factor={scale:.4f}", "--window-size=1500,24000",
         f"--screenshot={shot}", str(html)],
        capture_output=True, text=True,
    )
    if not shot.exists():
        raise SystemExit("Chrome produced no screenshot")
    im = Image.open(shot).convert("RGB")
    a = np.asarray(im)
    rows = np.where((a < 250).any(axis=(1, 2)))[0]
    im = im.crop((0, 0, im.width, int(rows.max()) + 60))
    if im.width != W:
        im = im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
    print(f"   page is {im.width} x {im.height}px")
    return im


def plan(page_h: int, minutes: float) -> list[tuple[int, float]]:
    """Scroll positions and how long to hold each, weighted by how much text is there."""
    step = H - OVERLAP
    stops = list(range(0, max(page_h - H, 1), step)) + [max(page_h - H, 0)]
    stops = sorted(set(stops))
    return stops


def build(minutes: float, skip_execute: bool) -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ipynb = NOTEBOOK if skip_execute else execute_notebook(tmp)
        page = screenshot(to_html(ipynb, tmp), tmp)

    arr = np.asarray(page)
    stops = plan(page.height, minutes)

    # Hold time follows the amount of ink on each screen, so a dense table gets
    # longer than a heading on its own.
    ink = np.array([float((arr[y:y + H] < 200).mean()) for y in stops])
    weight = ink / ink.sum()
    budget = minutes * 60 - len(stops) * TRANSITION - 2 * HEAD_TAIL
    holds = np.clip(weight * budget, 3.5, 30.0)
    holds[0] += HEAD_TAIL
    holds[-1] += HEAD_TAIL
    total = holds.sum() + len(stops) * TRANSITION

    print(f"{len(stops)} screens, {total/60:.1f} min "
          f"(holds {holds.min():.0f}s to {holds.max():.0f}s)")

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
         "-crf", "20", "-pix_fmt", "yuv420p", str(OUT)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    def frame(y: int) -> bytes:
        y = max(0, min(int(y), page.height - H))
        return arr[y:y + H].tobytes()

    try:
        for i, y in enumerate(stops):
            for _ in range(int(holds[i] * FPS)):
                proc.stdin.write(frame(y))
            if i + 1 < len(stops):
                y0, y1 = y, stops[i + 1]
                n = int(TRANSITION * FPS)
                for f in range(n):
                    t = (f + 1) / n
                    t = t * t * (3 - 2 * t)          # ease in and out
                    proc.stdin.write(frame(y0 + (y1 - y0) * t))
            print(f"   screen {i+1}/{len(stops)}", end="\r")
    finally:
        proc.stdin.close()
        proc.wait()
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=5.0, help="target length")
    ap.add_argument("--skip-execute", action="store_true",
                    help="use the notebook as saved, without re-running it")
    a = ap.parse_args()
    build(a.minutes, a.skip_execute)
