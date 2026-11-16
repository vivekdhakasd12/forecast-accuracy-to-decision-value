# Report build

`report.tex` is the single source for `report.pdf`. It reuses the LaTeX preamble,
title-page layout and typographic settings of the Case Study 2 report so the two
documents are visually consistent; only the title, the case-study number and the
PDF metadata differ.

## Build

```bash
tectonic -X compile report.tex --outdir .
```

Tectonic is used rather than a full TeX Live install: it fetches only the
packages the document needs and produces the same output on any machine. The
document uses `fontspec`, so it must be compiled with a XeTeX- or LuaTeX-based
engine — `pdflatex` will fail.

## Figures

- The report pulls figures from the `case_study_1/figures/` directory, which is written by the five experiment
  scripts in `case_study_1/experiments/`. Regenerate them with `make cs1-e1`
  through `make cs1-e5` from the repository root.
  There is deliberately only one copy: keeping a second set beside the report
  meant the two could disagree about what the results were.
- `figures/pipeline_architecture.pdf` and `figures/day_ahead_protocol.pdf` are
  schematics drawn by `diagrams.py` in this directory:

  ```bash
  ../../.venv/bin/python diagrams.py   # run from case_study_1/report/
  ```

  They are vector PDFs; everything else is a 200 dpi raster, which at the 13 cm
  text measure works out to roughly 430 dpi on the page.

## References

Every entry in Chapter 9 was resolved against a live bibliographic source
(Crossref, OpenAlex or arXiv) rather than written from memory. The DOIs in the
reference list are the ones those lookups returned.
