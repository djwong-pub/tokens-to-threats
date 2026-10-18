"""Phase 32 — Start paper writing early: verify the filled-in sample document.

The chapter's artifact is a three-page working draft of the executed Top-K
injection experiment (ch. 30), written in the plan's eight-section structure
(Problem, Hypothesis, Background, Method, Experimental setup, Results,
Limitations, Conclusion). This script checks the artifact the same way the
book checks everything else: structurally, against the real run's numbers,
and against the no-placeholder rule.

Checks (self-test mode, the default)
  sections-complete  all eight section headings present, in the plan's order
  numbers-real       the headline numbers of the executed run appear in the
                     draft (the draft must cite the log, not a memory)
  limitations-first  the Limitations section names the underpowered ablation
                     and the scripted-policy bound (the two honest readings)
  no-filler          no work-in-progress or filler markers anywhere (the
                     marker families are composed at runtime, below)
  length-band        the draft sits in the three-page band (word count)

Run:  ./.venv/bin/python code/phase32_writing/paper_skeleton_check.py
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SAMPLE = os.path.join(ROOT, "textbook", "samples",
                      "sample-paper-topk-injection.tex")

SECTIONS = ["Problem", "Hypothesis", "Background", "Method",
            "Experimental setup", "Results", "Limitations", "Conclusion"]


def tex_prose(path: str) -> str:
    """Strip unescaped % comments; keep verbatim-free body text."""
    out = []
    for line in open(path):
        line = re.sub(r"(?<!\\)%.*", "", line)
        out.append(line)
    return "\n".join(out)


def main() -> int:
    assert os.path.exists(SAMPLE), f"sample artifact missing: {SAMPLE}"
    src = tex_prose(SAMPLE)

    # ---------------------------------------------------- sections-complete
    pos = -1
    for name in SECTIONS:
        m = re.search(r"\\section\{" + re.escape(name) + r"\}", src)
        assert m, f"missing section: {name}"
        assert m.start() > pos, f"section out of order: {name}"
        pos = m.start()
    print(f"PASS sections-complete (all 8 sections present in the plan's "
          f"order, {', '.join(SECTIONS)})")

    # ----------------------------------------------------------- numbers-real
    required = ["0.583", "0.333", "3.142", "-0.507", "2.306", "3.977",
                "1.343", "0.836", "28.5", "171.6",
                "[2,2,3,3,4,4,5,6,7,9,11,15]"]
    missing = [r for r in required if r not in src]
    assert not missing, f"draft missing run numbers: {missing}"
    print(f"PASS numbers-real (all {len(required)} headline numbers of the "
          f"executed run appear in the draft, the ASR curve, both CIs, the "
          f"resolvability threshold, cost endpoints, the rank list)")

    # -------------------------------------------------------- limitations-first
    lim = src.split(r"\section{Limitations}", 1)[1]
    lim = lim.split(r"\section{Conclusion}", 1)[0]
    for needle, label in (("62", "the n~62 power requirement"),
                          ("upper bound", "the scripted-policy exposure bound"),
                          ("constructed", "the constructed-corpus scope")):
        assert needle in lim, f"Limitations missing {label}"
    print("PASS limitations-first (Limitations names the underpowered "
          "ablation's n requirement, the scripted-policy upper bound, and the "
          "constructed-corpus scope)")

    # ----------------------------------------------------------- no-filler
    marker_families = "|".join(("TO" + "DO", "T" + "BD", "FIX" + "ME",
                                "X" + "XX", "lo" + "rem"))
    bad = re.findall(r"\b(" + marker_families + r")\b|\?\?\?", src,
                     re.IGNORECASE)
    assert not bad, f"filler markers found: {bad[:5]}"
    print("PASS no-filler (0 work-in-progress or filler markers in the "
          "draft, marker families checked at runtime)")

    # ------------------------------------------------------------ length-band
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", src)
    n_words = len(words)
    assert 800 <= n_words <= 2200, \
        f"draft word count {n_words} outside the three-page band"
    print(f"PASS length-band ({n_words} words, the three-page band of "
          f"800-2200 holds)")

    print(f"METRIC n_words={n_words} n_sections=8 n_numbers={len(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
