"""Package the minimum the Colab demo needs into one small zip.

The full dataset is 124 MB and takes minutes to download. The demo does not need
it: the design matrix is already built and cached at 4.5 MB, and the aggregate
result tables are a few hundred kilobytes. Bundling those instead means the
notebook starts working within seconds of being opened.

    .venv/bin/python presentation/make_colab_bundle.py

Output: presentation/colab_bundle.zip
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "case_study_1" / "presentation" / "colab_bundle.zip"

INCLUDE_TREES = ["src", "configs", "tests"]
INCLUDE_GLOBS = [
    "data/processed/design_solar_g12.parquet",      # the main design matrix
    "case_study_1/results/*.csv",                                # aggregate tables, small
]
# The four per-country matrices for the transfer study add 15 MB and 58 seconds
# of refitting. The notebook reads that result from results/e5_transfer_mean.csv
# instead, so they are only bundled on request.
TRANSFER_GLOB = "data/processed/design_*_quantile.parquet"
SKIP = {"__pycache__", ".pytest_cache", ".DS_Store"}


def main(with_transfer: bool = False) -> None:
    files: list[Path] = []
    for tree in INCLUDE_TREES:
        for p in (ROOT / tree).rglob("*"):
            if p.is_file() and not any(s in p.parts or p.name == s for s in SKIP):
                files.append(p)
    for pattern in INCLUDE_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    if with_transfer:
        files.extend(sorted(ROOT.glob(TRANSFER_GLOB)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(set(files)):
            z.write(f, f.relative_to(ROOT))

    mb = OUT.stat().st_size / 1e6
    print(f"{OUT.name}: {len(set(files))} files, {mb:.1f} MB")
    for tree in INCLUDE_TREES:
        n = sum(1 for f in set(files) if str(f.relative_to(ROOT)).startswith(tree))
        print(f"   {tree:<10} {n:>3} files")
    n = sum(1 for f in set(files) if str(f.relative_to(ROOT)).startswith("data"))
    print(f"   {'data':<10} {n:>3} files (cached design matrices, no raw download needed)")
    n = sum(1 for f in set(files) if str(f.relative_to(ROOT)).startswith("case_study_1/results"))
    print(f"   {'results':<10} {n:>3} files (aggregate tables)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-transfer", action="store_true",
                    help="also bundle the per-country matrices (+15 MB)")
    main(ap.parse_args().with_transfer)
