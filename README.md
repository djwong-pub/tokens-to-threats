# Tokens to Threats

### A From-Scratch Textbook for LLMs, Agents, and Security

**Curated by Wong Ding Jie. Compiled by AI agents.**

> This is a compilation for me to learn all about AI. Feel free to learn from it too!

A 32-phase textbook that goes from matrix arithmetic to a running agent server and then to a
measurable security experiment. Every derivation is written out. Every number is either derived
in the text or produced by code in this package, on a machine with no GPU.

**first edition**, typeset October 18, 2026. `textbook.pdf` sha256 `948af3653a283dbd…`

| | |
|---|---|
| Book | `textbook.pdf`, 780 pages, 32 chapters in 9 parts, 6 appendices, glossary as index |
| Figures | 329 diagrams, all generated from the scripts shipped here |
| Code | 38 self-asserting scripts (`code/`), 61 symbolic and numeric checks of the book's maths |
| Reading | 132 verified papers, listed with links in `references/READING.md` (PDFs not included) |
| Hardware | CPU only. No figure in the book is presented as a hardware benchmark |

## What this package contains

```
textbook.pdf            the book
code/                   the runnable suite: 32 phase directories + harness + shared modules
code/RESULTS.md         results table for every script (assertions, measured numbers)
code/data/              the small hand-written corpus the n-gram phases train on
textbook/samples/       one sample paper, checked by the phase-32 script
references/READING.md    the 132-paper reading list with links
requirements.txt        numpy, matplotlib, torch + the interpreter version used
MANIFEST.sha256         hash of every file in this package
RIGHTS.md               the terms. Read this one before reusing anything.
```

## Run the suite

No GPU, no network, no dataset download. Every script ends in numeric assertions rather than a
picture, so a green run means the number printed in the book is the number your machine
produces.

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python code/harness.py            # all 38 scripts, about 2.5 minutes
./.venv/bin/python code/harness.py phase10    # one phase
```

Expected final line: `[harness] 38/38 passed`. `torch` is used only as a cross-check oracle in 6
of the 32 phase directories; the rest need numpy and matplotlib. One script
(`code/phase19_server/mini_server.py`) opens an HTTP server on `127.0.0.1` with an ephemeral port
and talks to itself; nothing else contacts any host.

## Getting the cited papers

`references/READING.md` lists all 132 with title, first author, year, venue and link. The
binaries are not redistributed here; each link goes to the paper's own host. For the 125 arXiv
entries the two useful URL shapes are:

```bash
# abstract page (and "View PDF" from there)
https://arxiv.org/abs/2302.10147
# the PDF directly
https://arxiv.org/pdf/2302.10147
```

The remaining 7 are on PLOS, PMLR, JMLR and USENIX; their links resolve to the publisher page.

## What is deliberately not here

* **The 132 papers.** Third-party copyrighted. `references/READING.md` gives the link for each.
* **The LaTeX source of the book.** The PDF is the deliverable. `textbook/samples/` is the one
  manuscript file included, because `code/phase32_writing/paper_skeleton_check.py` checks it.
* **The figure generators.** `figures_src/` holds the scripts that draw the 329 diagrams into the
  book; figure captions in the chapters name their generator, and those names are not shipped.
  Nothing in this package imports them, which the export's import-closure check proves on every
  build. The gap between the paths the book names and the paths it ships is measured in the
  source repository, not hidden.
* **The build and QA machinery** of the project that produced this (`qa/`, `notes/`, raw harness
  logs). Raw logs carry the build machine's paths.

## Provenance and affiliation

Set in the Springer Nature Monographs template (`SNmono.cls`), which is not included and whose
presence here would mean nothing: **this book is self-published and has no affiliation with
Springer Nature.** The LaTeX class was used for layout only, and the title page says so.

Figures are drawn by this project's own matplotlib scripts; no figure is taken from a paper. The
fonts embedded in the PDF (DejaVu, STIX, Computer Modern and derivations) keep their own terms,
which permit embedding in a document.

## Rights

All rights reserved by Wong Ding Jie. Read `RIGHTS.md`. Short version: read it, run it, link it,
and quote short passages with attribution to the extent your jurisdiction's own exceptions allow.
**No licence is granted** to rehost, redistribute, adapt or sell any of it, and commercial use of
the text, the figures, the PDF or the code requires prior written permission. If you need the code
under a permissive licence for a course or a repo, ask through the repository issue tracker.
