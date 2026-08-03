# Provenance and release scope

Everything the dashboard displays comes from `release/`, which is built by
`tools/build_release.py` from artifacts in the analysis workspace. No number is
re-derived by hand and no reported value is transcribed into the app.

## The two audit builds

The manuscript reports two audit builds that share one scoring definition and
differ only in their exported record sets (Section 4). The release carries both,
and the dashboard loads either into the same views.

### `case_study` — Qwen3-VL Staged GRPO, 541 records (default)

The run traced end to end in the paper: the teaser figure, the answer-action
ledger (Table 1), the task-by-dataset risk map (Table 2), and the global signals
in Section 4.1. GRPO is applied to the Staged SFT adapter with a grounding
reward.

Source: **`plan/inspector_data.json`** — the same file the static prototype
serves. The builder verifies that it declares the Staged GRPO condition, that it
holds exactly 541 records, and that every exported `quadrant` label agrees with
the τ = 0.5 gate recomputed from `answer_correct` and `obj_recall`.

Every one of the 541 records in this build has its crop image released, so it
never falls back to a placeholder.

Fields the 541-record export does not carry: **`point_f1`** and **`pim_prec`**.
These are joined from **`outputs/percrop_staged_grpo.csv`**, produced by
re-running `03_score_and_join.py` for this same condition. They are *not* taken
from `outputs/percrop.csv`, whose rows for those columns belong to other model
conditions.

The join is refused unless the score file agrees with the export on
`answer_correct`, `n_gt`, `n_pred` and `obj_recall` (tolerance 5e-4) for every
record. On the 541 overlapping records there are zero disagreements, and the
joined values reproduce the manuscript's `pim_prec_given_correct` of **0.471**
exactly. Only these two columns are added; no value already in the export is
replaced. If the score file is absent the columns stay empty and the app says so.

Question text, MCQ options, and the six records whose centroid list is empty are
cross-checked against `outputs/paired_set.jsonl`. Those six genuinely have
`n_gt = 0` (image regions with no labelled mitochondrion), which both sources
agree on.

### `sft_variants` — Qwen3-VL, 753 records × 5 conditions

The re-audit table (Table 3): zero-shot, Perception SFT, Grounding SFT, Staged
SFT, Joint SFT, each scored on the same 753 paired records.

| Field group | Source |
| --- | --- |
| answer correctness, object recall, point F1, point-in-mask precision, count abs. error, expected letter/answer, model answer snippet, task, dataset | `outputs/percrop.csv` |
| question text, MCQ options, ground-truth centroids, source image path, probe id | `outputs/paired_set.jsonl` |
| parsed predicted point coordinates | `outputs/preds_qwen3vl_{zero_shot,perception_sft,grounding_only,staged_sft,joint_sft}.jsonl` |

All 753 crop images for this build are now released, so every record has its
image and the audit queue's image filter is inert for it. Aggregates (ledger, risk
map, summary tiles, re-audit table) are computed over all 753 records regardless;
the image filter only ever narrowed the single-record queue, and it remains
available for uploaded runs, which may be partial.

## Metrics the dashboard derives rather than reads

Two per-record columns are computed in the app from the released coordinates,
because no shipped export carries them:

- **hit rate of points** (`points on target / points emitted`) and the
  **stray-point count**. These make visible an error the paper's action gate
  cannot see: the gate is a recall threshold, so a record can cover its objects
  and still emit points that correspond to nothing.
- **point F1**, but only where the export omits it. Where the export has it, the
  scored value is shown and labelled *as scored in the run*.

Both use centroid matching at a normalised distance of 0.1 — the same rule and,
where scipy is present, the same optimal assignment as `03_score_and_join.py`.
They are labelled *computed here from the coordinates* wherever they appear.

Derived point F1 agrees exactly with the exported column on **97.5%** of the
753-record Joint SFT rows. Derived object recall agrees on only 79%, because the
offline scorer prefers point-in-mask hits when label masks are reachable and
falls back to centroids otherwise. **Object recall is therefore always read from
the export, never recomputed** — the derived value is not a substitute.

**Point-in-mask precision cannot be derived at all**: it needs the label masks,
which are not redistributed. Both shipped sources now carry the values computed
from those masks (see the case-study section above and `percrop.csv` for the
SFT variants); a source that does not carry them is reported as missing them
rather than showing a zero.

## Shared conventions

- Dataset ids are mapped to the paper's names — `lucchi_plus` → Lucchi,
  `vnc_drosophila` → VNC, `mitoem_human` → EM-H — and the raw id stays on every
  record and is shown in the audit panel.
- The **Location** row folds direct-location and marked-region probes together,
  as the manuscript's risk-map table does. The single-option (K = 1) dynamic
  marked-region records are a subset of that row.
- The trust-gap interval is a nonparametric percentile bootstrap with
  `B = 2000` resamples and seed 0 via `numpy.random.default_rng`, matching
  Section 3.5. Resamples that lose one of the two answer strata are skipped, not
  imputed.
- Probe widths are variable. The manifest records the option-count distribution
  and the record-wise uniform-choice baseline per build (26.3% over 541 records,
  26.4% over 753), so the answer-side rates are always read next to their
  baseline.

## Integrity checks enforced at build time

The builder aborts rather than emitting a questionable release if:

- `inspector_data.json` does not declare Staged GRPO or does not hold 541 records;
- an exported quadrant label disagrees with the τ = 0.5 gate;
- an SFT condition does not have exactly 753 rows in `percrop.csv`;
- a record's parsed predicted-point count disagrees with `n_pred`;
- a record's ground-truth centroid count disagrees with `n_gt`, or `n_gt`
  disagrees between `inspector_data.json` and `paired_set.jsonl`;
- a source crop is not truly greyscale (which would make the 8-bit conversion
  lossy).

All 4,306 released records pass these checks.

## Conditions present in the workspace but not in the release

The manuscript reports only the Qwen3-VL builds above, and both are released in
full. The workspace also retains figure-level exports for other runs — Gemma3,
InternVL, Grounding-only GRPO, Joint GRPO — but their per-record rows are absent
from `outputs/percrop.csv` (14, 11, 0 and 0 rows respectively). They are not
referenced by the submitted paper and are not in the release.

To add a condition later, restore its per-record rows and prediction JSONL and
extend `SFT_CONDITIONS` in `tools/build_release.py`. The 753-row assertion will
refuse anything incomplete.

## The static prototype

`plan/oversight_inspector.html` reads `inspector_data.json` from its own
directory and shows the same Staged GRPO case study this app defaults to.

It previously fell back to a `DATA` constant embedded in the page — the output of
a synthetic generator — whenever the fetch failed, which included any `file://`
open. Both have been removed: the generator is gone from
`plan/04_reliability_analysis.py`, which now exits if `--percrop` is missing
instead of fabricating a table, and the page has no embedded data at all. If the
export cannot be loaded the page states that and stops.

Serve the directory before opening it:

```bash
cd plan && python3 -m http.server 8099
# http://localhost:8099/oversight_inspector.html
```
