# Rights notice

**Copyright © Wong Ding Jie. All rights reserved.**

## The terms, in one paragraph

**All rights reserved.** This package is published for **read-only public display**: you may read,
download, run and link to it. **No licence is granted**, express or implied, to reproduce,
redistribute, rehost, adapt, sublicense or sell the text, the figures, the PDF or the code beyond
what your jurisdiction's own exceptions allow. **Commercial use requires prior written permission**
from Wong Ding Jie.

## What this means for you

You may, without asking:

* read the book, in any format you already own;
* link to this repository or to the PDF here;
* run the code locally, for your own study and teaching;
* quote short passages with attribution, to the extent the exceptions in your own jurisdiction
  allow (fair use, fair dealing, quotation for criticism or review, and so on);
* print a copy for your own personal use.

You may not, without prior written permission:

* sell, bundle for sale, or place this book or the code behind a paywall;
* republish the text, the figures or the PDF as your own edition, in whole or substantially;
* use the work or its materials in a commercial product or paid course;
* train or evaluate a commercial model on the text of this book.

If you want to do any of those, ask through the repository issue tracker. Attribution is
appreciated even where it is not required.

## What is **not** covered by this notice

This notice applies only to material authored by Wong Ding Jie. Third-party items keep their own
terms and are not re-licensed here:

| Item | Status | Terms |
|---|---|---|
| The 132 cited papers | **not included** in this package | remain with their authors and publishers; links in `references/READING.md` |
| Fonts embedded in the PDF (DejaVu, STIX, Computer Modern and derivations) | subset-embedded by the typesetter | their own font licences, which permit embedding in a document |
| `numpy`, `matplotlib`, `torch` | dependencies, not vendored | BSD-style / PSF-derived / Apache-2.0 family; installed from PyPI by you |
| `SNmono.cls` (Springer Nature Monographs LaTeX class) | **not included** | used for layout only. No affiliation with or endorsement by Springer Nature |

## Assertions, not promises

The code in `code/` is offered as-is, with no warranty of any kind, to the extent permitted by
law. Its assertions verify arithmetic and behaviour on a CPU-only reference machine; they are
measurements of one build, and `code/RESULTS.md` records which. Where the book says a number was
**measured**, it means measured by this code; where it says **analytic**, it means derived on
paper; **simulated** marks an estimate no available machine could run. That labelling is part of
the text, not a disclaimer added afterwards.

## Security content

Attack material in this book and its code is restricted to local, synthetic, sandboxed,
CTF-style demonstrations against software the reader runs, against models the reader hosts. The
synthetic flags and credentials in the testbeds (`FLAG-SYNTH-*`, the fake triage notes) are
generated for the exercises. Nothing here is an operational guide against systems you do not own.

## Authorship

Curated by Wong Ding Jie. Compiled by AI agents. The rights above are asserted by Wong Ding Jie.
