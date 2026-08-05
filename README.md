# Spatial Action Review

A dashboard for checking whether an AI model's *answer* about a microscope image
can be trusted as permission for the *action* it takes on that image.

In an automated electron-microscopy workflow, a vision-language model does two
things at once. It answers a question a scientist can read — *"are there
mitochondria here?"* — and it returns a set of points the workflow acts on:
seeding a segmentation, choosing where to look next, flagging a region for review.
A person supervising the run usually sees only the answer. The points are what
actually happen.

The two can disagree. The model can answer correctly and still put its points
somewhere useless, and nobody reading the answer would know. This dashboard puts
both halves side by side on the same image region, so a supervisor can see the
disagreement and decide what to do about it.

Companion to *Spatial Action Review: A Visual Analytics Dashboard for Auditing
Language-to-Action Hand-offs in Electron Microscopy* (VAxAutoSci @ IEEE VIS).

![The dashboard](docs/screenshot.png)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

It opens at <http://localhost:8501>. Nothing else to set up — the audit data ships
with the app.

## What you are looking at

**Answer–action ledger** (top left). Every image region falls into one of four
boxes, depending on whether the answer was right and whether the points were
usable:

| | points usable | points not usable |
| --- | --- | --- |
| **answer correct** | aligned pass | **silent failure** |
| **answer wrong** | action-only pass | blocked |

**Silent failure** is the case that matters. The answer looks fine, so a
supervisor reading answers waves it through, but the action underneath is wrong.
Click any box to work through its regions.

**Risk map** (below the ledger). The same silent-failure rate broken down by
question type and dataset, so you can see *where* the problem concentrates rather
than only how big it is. Click a cell to narrow the queue to that region; click it
again to widen it.

**Image-region audit** (right). One region at a time: the microscope image with
green crosses on the real mitochondria and red rings where the model pointed, the
question it was asked, the options, its answer, the expected answer, and every
score. This is where you find out *why* a region failed.

**Routing decision** (bottom right). The audit ends in a choice, not a number:
accept the action, send the region for human inspection, hold it for a stricter
setting, or flag the model for revision. Each decision is saved with the settings
it was made under, and the **Review log** tab exports them all.

**The gate slider** (sidebar). A region's points "pass" when they cover enough of
the real mitochondria. You choose how much is enough. Everything on the page
recomputes as you move it, because how strict you are is a safety decision, not a
display preference.

**Stray actions.** The gate only asks whether the model found the objects. It
cannot see the opposite mistake — extra points landing on nothing, each one an
operation the workflow would carry out on empty image. Those are counted
separately and flagged in red.

## The data

Three sources, chosen in the sidebar:

| Source | What it is |
| --- | --- |
| **Case study (541)** | The run followed end to end in the paper: one adapted Qwen3-VL checkpoint over 541 image regions. |
| **SFT-variant audit (753)** | Five Qwen3-VL variants over the same 753 regions, for comparing how the picture changes after each round of fine-tuning. |
| **Your own run** | Upload your own results file and the dashboard audits it the same way. The views are not tied to these particular runs. |

4,306 records and 753 microscope crops, all included in the repository. No
database, no external service.

## Does it match the paper?

Yes, and that is checked rather than claimed:

```bash
python -m pytest tests -q      # 87 passed
```

The tests hold the dashboard to the published numbers — the headline figures, all
four ledger counts, all fifteen risk-map cells, every model variant with its
confidence intervals — each transcribed from the manuscript. If a change to the
code or the data moves any of them, the tests fail.

To rebuild the data files from the original analysis outputs:

```bash
python tools/build_release.py            # rebuild
python tools/build_release.py --check    # verify
```

There is also a set of pre-publication checks — that the licence is verbatim, that
nothing in the tree can fabricate a number, that no local paths or credentials are
committed. One of them fails on purpose until the author fields in `CITATION.cff`
are filled in:

```bash
python -m pytest -m release      # run before making the repository public
```

The builder refuses to produce a data file it cannot vouch for. Mismatched record
counts, scores that disagree between sources, or overlays that would draw more
points than a region was actually scored on all stop the build rather than
reaching the dashboard. Where a measurement genuinely is not available, the
dashboard says so instead of showing a zero, and it never draws a microscope image
it does not have.

## Deploying it

The data is ~39 MB, so the repository is self-contained and deploys as-is to
[Streamlit Community Cloud](https://share.streamlit.io) — point it at this repo
and `app.py`. See [DEPLOY.md](DEPLOY.md), including how to deploy from a **private**
repository.

Re-querying a live model is optional and stays off unless configured; see
[RUN_ON_GPU.md](RUN_ON_GPU.md).

## Files

```
app.py                    the page
sar/                      loading, scoring, rendering, sign-off, model client
tools/build_release.py    rebuilds the data files
tests/                    the checks against the paper
release/                  the audit records and the crop images
```

- [DEPLOY.md](DEPLOY.md) — publishing the app, public or private.
- [PROVENANCE.md](PROVENANCE.md) — where every number comes from, and what is
  deliberately left out.
- [RUN_ON_GPU.md](RUN_ON_GPU.md) — regenerating the data, serving a model.
- [app_implementation.md](app_implementation.md) — the full implementation record.

## Scope: what is here and what is not

This repository is the **audit** half of the work, and it is complete on its own
terms — everything needed to reproduce every number in the paper is here.

**Included.** The dashboard; the 4,306 audit records and 753 crops; the scoring and
reliability analysis that turn per-record model outputs into the reported measures;
the builder that assembles the records; the tests that hold all of it to the
published values.

**Not included.** The code that adapted the Qwen3-VL checkpoints (the supervised
and reward-based fine-tuning recipes) and the resulting adapter weights. That work
is reported separately and is not part of this study. The audit does not depend on
it: the records already contain each model's answers and predicted points, so the
reliability measures, the ledger, the risk map and every figure reproduce from what
is here without retraining anything.

If you want to audit a different model, you do not need our training code either —
export your own run in the same paired answer–action schema and load it through
**Your own run**. The required fields are listed in [PROVENANCE.md](PROVENANCE.md).

## Licence and citation

The code, documentation and analysis in this repository are licensed under the
**GNU Affero General Public License v3.0 or later** (`AGPL-3.0-or-later`) — see
[LICENSE](LICENSE). In
short: use it, study it, modify it and share it freely, including for research and
teaching; if you distribute a modified version, or run one as a network service,
you must make your source available under the same licence.

To cite it, see [CITATION.cff](CITATION.cff), or cite the paper directly.

The microscope crops in `release/images/` derive from the public **Lucchi**, **VNC**
and **MitoEM** datasets. Those carry their own terms, which are not superseded by
the licence above; the datasets are cited in the paper and recorded in
[PROVENANCE.md](PROVENANCE.md).
