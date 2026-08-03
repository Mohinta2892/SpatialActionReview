"""Spatial Action Review — live oversight dashboard for language-to-action hand-offs.

Layout follows plan/oversight_inspector.html: an answer-action ledger and a
task-by-dataset risk map on the left, a single-record image-region audit on the
right. Selecting a ledger cell or a risk-map cell filters the audit queue.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sar import ingest as ingest_mod
from sar import theme, ui
from sar.ingest import IngestError
from sar.data import (
    QUADRANT_BLURB,
    QUADRANTS,
    QUADRANT_LABEL,
    build_records,
    condition_table,
    load_release,
    risk_map,
    summarise,
    with_gate,
)
from sar.render import crop_tile
from sar import review as review_mod
from sar import scoring as scoring_mod
from sar import serve as serve_mod

APP_DIR = Path(__file__).resolve().parent
RELEASE_DIR = APP_DIR / "release"
IMAGE_DIR = RELEASE_DIR / "images"
PAPER_TAU = 0.5

st.set_page_config(
    page_title="Spatial Action Review — Autonomous EM Oversight",
    page_icon="◎",
    layout="wide",
)


# ------------------------------------------------------------------ caching

@st.cache_data(show_spinner="Loading audit record set…")
def _load():
    return load_release(RELEASE_DIR)


@st.cache_data(show_spinner=False)
def _scored(build: str, condition: str, tau: float) -> pd.DataFrame:
    df, _ = _load()
    return with_gate(build_records(df, build, condition), tau).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _summary(build: str, condition: str, tau: float):
    return summarise(_scored(build, condition, tau))


@st.cache_data(show_spinner=False)
def _risk(build: str, condition: str, tau: float) -> pd.DataFrame:
    return risk_map(_scored(build, condition, tau))


@st.cache_data(show_spinner=False)
def _tau_sweep(build: str, condition: str) -> pd.DataFrame:
    df, _ = _load()
    sub = build_records(df, build, condition)
    rows = []
    for tau in [round(0.05 * i, 2) for i in range(1, 20)]:
        scored = with_gate(sub, tau)
        correct = scored["answer_correct"] == 1
        reliable = scored["action_reliable"] == 1
        rows.append({
            "τ": tau,
            "aligned-pass rate": float((correct & reliable).mean()),
            "silent-failure rate": float((correct & ~reliable).mean()),
        })
    return pd.DataFrame(rows)


def _tau_sweep_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Threshold sweep for a record set already in memory (an uploaded run)."""
    rows = []
    for tau in [round(0.05 * i, 2) for i in range(1, 20)]:
        gated = with_gate(scored, tau)
        correct = gated["answer_correct"] == 1
        reliable = gated["action_reliable"] == 1
        rows.append({
            "τ": tau,
            "aligned-pass rate": float((correct & reliable).mean()),
            "silent-failure rate": float((correct & ~reliable).mean()),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _conditions_at(build: str, conditions: tuple[str, ...], tau: float) -> pd.DataFrame:
    df, _ = _load()
    return condition_table(df, build, list(conditions), tau)


df_all, manifest = _load()


# ------------------------------------------------------------------ state

BUILDS = manifest["builds"]
DEFAULT_BUILD = manifest["default_build"]
UPLOAD_BUILD = "uploaded"
BUILD_OPTIONS = [*manifest["build_order"], UPLOAD_BUILD]


@st.cache_data(show_spinner="Reading uploaded run…", max_entries=3)
def _ingest_upload(json_bytes: bytes, zip_bytes: bytes | None):
    """Normalise an uploaded run. Cached on the file contents, not the widget."""
    image_dir = None
    if zip_bytes:
        # Keyed by content hash so re-running the same upload reuses the extraction
        # and two different uploads never collide.
        digest = hashlib.sha256(zip_bytes).hexdigest()[:16]
        image_dir = Path(tempfile.gettempdir()) / f"sar_upload_{digest}"
        if not image_dir.exists():
            ingest_mod.extract_images(zip_bytes, image_dir)
    payload = json.loads(json_bytes.decode("utf-8"))
    ing = ingest_mod.ingest(payload, image_dir)
    return ing.records, {
        "model": ing.model, "condition": ing.condition, "tau_declared": ing.tau_declared,
        "n_images": ing.n_images, "missing_fields": ing.missing_fields,
        "warnings": ing.warnings,
        "image_dir": str(image_dir) if image_dir else "",
    }


def _init_state() -> None:
    st.session_state.setdefault("build", DEFAULT_BUILD)
    st.session_state.setdefault("condition", BUILDS[DEFAULT_BUILD]["conditions"][0])
    st.session_state.setdefault("tau", PAPER_TAU)
    st.session_state.setdefault("quadrant", "silent_failure")
    st.session_state.setdefault("region", None)  # (task, dataset) or None
    st.session_state.setdefault("cursor", 0)
    st.session_state.setdefault("pending_qid", None)
    st.session_state.setdefault("palette", theme.DEFAULT)
    # The audit queue defaults to records whose crop image ships with the app,
    # so the crop view always opens on real evidence. Aggregates are unaffected:
    # the ledger, risk map, and summary always use every record in the build.
    st.session_state.setdefault("images_only", True)
    st.session_state.setdefault(review_mod.STATE_KEY, {})
    st.session_state.setdefault("route_note", "")
    st.session_state.setdefault("live", {})       # question_id -> LiveResult
    st.session_state.setdefault("endpoint_url", serve_mod.Endpoint.from_env().base_url)
    st.session_state.setdefault("endpoint_model", serve_mod.Endpoint.from_env().model)


_init_state()


# Every one of these runs as a widget callback. Streamlit forbids assigning to
# st.session_state[k] for a widget-keyed k *after* that widget has been created
# in the current run, so the sidebar controls must mutate state from callbacks
# (which execute before the rerun instantiates any widget) rather than from an
# `if st.button(...):` body further down the page.

def _reset_queue() -> None:
    st.session_state.cursor = 0


def _select_build(name: str) -> None:
    """Switch audit source, resetting the condition to that source's first one."""
    st.session_state.build = name
    chosen = name
    if chosen in BUILDS:
        st.session_state.condition = BUILDS[chosen]["conditions"][0]
    st.session_state.region = None
    st.session_state.pending_qid = None
    _reset_queue()


def _reset_config() -> None:
    st.session_state.build = DEFAULT_BUILD
    st.session_state.condition = BUILDS[DEFAULT_BUILD]["conditions"][0]
    st.session_state.tau = PAPER_TAU
    st.session_state.quadrant = "silent_failure"
    st.session_state.region = None
    st.session_state.images_only = True
    st.session_state.pending_qid = None
    _reset_queue()


def _select_quadrant(quadrant: str) -> None:
    st.session_state.quadrant = quadrant
    _reset_queue()


def _select_region(task: str, dataset: str) -> None:
    current = st.session_state.region
    st.session_state.region = None if current == (task, dataset) else (task, dataset)
    _reset_queue()


def _set_palette(name: str) -> None:
    st.session_state.palette = name


def _sign_off(row, source: str, tau: float, route: str) -> None:
    """Record the supervisor's routing decision for the current record."""
    review_mod.record(
        st.session_state[review_mod.STATE_KEY],
        row=row, source=source, tau=tau, route=route,
        note=st.session_state.get("route_note", ""),
    )
    st.session_state.route_note = ""


def _clear_sign_off(question_id: str) -> None:
    review_mod.clear(st.session_state[review_mod.STATE_KEY], question_id)


def _clear_all_sign_offs() -> None:
    st.session_state[review_mod.STATE_KEY] = {}


def _reask(row, image_path, tau: float, endpoint) -> None:
    """Send the current record back to the served model and keep the result aside."""
    st.session_state.live[str(row["question_id"])] = serve_mod.reask(
        endpoint, row, image_path, tau
    )


def _open_example(quadrant: str, question_id: str) -> None:
    """Jump the audit queue to a pinned worked example."""
    st.session_state.tau = PAPER_TAU
    st.session_state.quadrant = quadrant
    st.session_state.region = None
    st.session_state.images_only = True
    st.session_state.pending_qid = question_id
    _reset_queue()


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.markdown("#### Run configuration")
    st.caption("Audit source")
    for name in BUILD_OPTIONS:
        label = BUILDS[name]["short_label"] if name in BUILDS else "Your own run (upload)"
        st.button(
            label,
            key=f"src_{name}",
            width="stretch",
            type="primary" if st.session_state.build == name else "secondary",
            on_click=_select_build, args=(name,),
            help="The two shipped record sets share one scoring definition and differ only in "
                 "their exported records, which is why the counts differ. The third option "
                 "audits any run that exports the same paired record.",
        )

    if st.session_state.build == UPLOAD_BUILD:
        st.caption(
            "The dashboard audits any vision-language run that exports the paired "
            "answer-action record. Drop in an inspector_data.json and, optionally, a zip of its "
            "crop images named crop_id.png."
        )
        st.file_uploader(
            "inspector_data.json",
            type=["json"],
            key="upload_json",
            on_change=_reset_queue,
            help="The schema emitted by 04_reliability_analysis.py. Required per record: "
                 "crop_id, dataset, task, answer_correct, obj_recall, n_gt, n_pred.",
        )
        st.file_uploader(
            "crop images (.zip, optional)",
            type=["zip"],
            key="upload_zip",
            on_change=_reset_queue,
            help="Entries named <crop_id>.png. Records without a matching image keep every "
                 "numeric and textual field and draw their points on a blank grid.",
        )
        with st.expander("Try it without a run of your own"):
            st.markdown(
                "Any export in this schema works. The two record sets shipped with the app are "
                "themselves valid uploads, which is how the loader is regression-tested."
            )
        build_meta = None
    else:
        build_meta = BUILDS[st.session_state.build]
        st.caption(build_meta["role"])

        if len(build_meta["conditions"]) > 1:
            st.selectbox(
                "Model condition",
                options=build_meta["conditions"],
                key="condition",
                on_change=_reset_queue,
                help="Each condition is scored on the same 753 paired records.",
            )
        else:
            st.caption(f"Condition · **{build_meta['conditions'][0]}**")

    st.slider(
        "Action-reliability gate τ",
        min_value=0.05, max_value=0.95, step=0.05,
        key="tau",
        on_change=_reset_queue,
        help="Rτ(c) = 1[obj_recall(c) ≥ τ]. A region's point action passes the gate when its "
             "points cover at least this fraction of the labelled mitochondria.",
    )
    st.caption(
        f"The gate is the pass condition for a point action: a region **passes** when its "
        f"points cover at least **{100 * st.session_state.tau:.0f}%** of the labelled "
        "mitochondria in that region, and **fails** otherwise. Everything on the page is "
        "recomputed from this one setting."
    )
    st.button("Reset run configuration", width="stretch", on_click=_reset_config)

    if build_meta is not None:
        st.divider()
        st.markdown("#### Worked examples")
        st.caption(
            "Pinned records for each behavioural state of this build, at τ = 0.50, spread across "
            "task families. Every one has its crop image."
        )
        for quadrant in QUADRANTS:
            group = build_meta["demo_cases"].get(quadrant)
            if not group or not group["cases"]:
                continue
            with st.expander(
                f"{QUADRANT_LABEL[quadrant]} · {len(group['cases'])}",
                expanded=(quadrant == "silent_failure"),
            ):
                st.caption(group["note"])
                for i, case in enumerate(group["cases"]):
                    st.button(
                        case["label"],
                        key=f"ex_{quadrant}_{i}",
                        width="stretch",
                        type=("primary" if st.session_state.pending_qid == case["question_id"]
                              else "secondary"),
                        on_click=_open_example,
                        args=(quadrant, case["question_id"]),
                        help=f"{case['question_id']} · {case['n_gt']} GT object(s), "
                             f"{case['n_pred']} predicted point(s)",
                    )

    st.divider()
    n_build = build_meta["n_records"] if build_meta else None
    n_img = build_meta["images_released"] if build_meta else None
    fully_covered = build_meta is not None and n_img >= n_build
    # Rendered unconditionally: Streamlit purges the session state of a keyed
    # widget that is skipped on a run, so hiding this on a fully covered build
    # would silently reset it to False on the way back.
    st.checkbox(
        "Audit queue: records with a crop image only",
        key="images_only",
        on_change=_reset_queue,
        disabled=fully_covered,
        help=(f"All {n_build} records here have a crop image, so this filter changes nothing."
              if fully_covered else
              "Records with no crop image keep every numeric and textual field and draw their "
              "points on a labelled blank grid. The ledger, risk map, and summary always use "
              "every record either way."),
    )

    with st.expander("Release scope"):
        st.markdown(f"**{manifest['n_records_total']:,} records** across two shipped sources.")
        for name in manifest["build_order"]:
            b = BUILDS[name]
            st.markdown(
                f"**{b['short_label']}** — {b['n_records']} records, "
                f"{len(b['conditions'])} condition(s), {b['images_released']} crop images  \n"
                f"`{b['source']}`"
            )
            if b["absent_record_fields"]:
                st.caption(
                    "not yet in this export, and shown as soon as it is: "
                    + ", ".join(b["absent_record_fields"])
                )
        if build_meta is not None:
            opt = build_meta["option_stats"]
            st.markdown(
                f"**Probe widths (this build)** — options per question K = "
                f"{min(int(k) for k in opt['k_distribution'])}–"
                f"{max(int(k) for k in opt['k_distribution'])}; record-wise uniform-choice "
                f"baseline {100 * opt['uniform_choice_baseline']:.1f}%; "
                f"{opt['n_single_option']} single-option (dynamic marked-region) records."
            )
        for key, note in manifest["notes"].items():
            st.caption(note)



build = st.session_state.build
tau = float(st.session_state.tau)
palette = st.session_state.palette
ui.inject_css(ui.css(palette))

# Appearance sits at the top of the workspace, not buried in configuration.
spacer, tb_label, *tb_cols = st.columns([7, 1.5, 1, 1])
tb_label.html(ui.toolbar_label("appearance"))
for col, name in zip(tb_cols, theme.PALETTES):
    col.button(
        theme.LABELS[name],
        key=f"pal_{name}",
        width="stretch",
        type="primary" if palette == name else "secondary",
        on_click=_set_palette,
        args=(name,),
    )
upload_meta: dict | None = None
if build == UPLOAD_BUILD:
    json_file = st.session_state.get("upload_json")
    zip_file = st.session_state.get("upload_zip")
    if json_file is None:
        st.html(ui.upload_prompt_html())
        st.stop()
    try:
        records, upload_meta = _ingest_upload(
            json_file.getvalue(), zip_file.getvalue() if zip_file else None
        )
    except (IngestError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        st.error(f"That file cannot be audited: {exc}")
        st.stop()

    model = upload_meta["model"]
    condition = upload_meta["condition"]
    build_label = "Your own run"
    image_dir = Path(upload_meta["image_dir"]) if upload_meta["image_dir"] else None
    scored = with_gate(records, tau).reset_index(drop=True)
    summary = summarise(scored)
    risk = risk_map(scored)
else:
    build_meta = BUILDS[build]
    condition = st.session_state.condition
    model = manifest["model"]
    build_label = build_meta["short_label"]
    image_dir = IMAGE_DIR
    scored = _scored(build, condition, tau)
    summary = _summary(build, condition, tau)
    risk = _risk(build, condition, tau)

st.html(ui.header_html(
    model=model, condition=condition, build_label=build_label, tau=tau,
    summary=summary, signed_off=len(st.session_state[review_mod.STATE_KEY]),
))

if upload_meta is not None:
    st.success(
        f"Audited **{len(scored):,}** records from your file — {model} / {condition} — "
        f"with **{upload_meta['n_images']}** matching crop image(s). Every state below is "
        "recomputed from `answer_correct` and `obj_recall` at τ = "
        f"{tau:.2f}; the file's own quadrant labels are not used."
    )
    if upload_meta["missing_fields"]:
        st.info(
            "Fields absent from every record in your file, shown as *not exported in this "
            "build* rather than as zeros: **" + "**, **".join(upload_meta["missing_fields"]) + "**."
        )
    for note in upload_meta["warnings"]:
        st.warning(note)

# Handles the Live-model tab needs, set while rendering the audit panel below.
current_record = None
current_image_path = None
queue_empty = True

left, right = st.columns([1.02, 1.0], gap="large")


# ------------------------------------------------------------------ left: ledger + risk map

with left:
    with st.container(border=True):
        st.html(ui.card_title(
            "Answer–action ledger",
            "Answer correctness A(c) against spatial-action reliability Rτ(c) on the same image "
            "region. <b>Silent failure</b> is the oversight hazard: the answer clears, the action "
            "does not. Select a cell to audit its records.",
        ))
        counts = scored["quadrant"].value_counts()
        rates = {q: int(counts.get(q, 0)) / len(scored) for q in QUADRANTS}

        head = st.columns([0.34, 1, 1])
        head[1].html('<div class="sar-rm-head">action reliable</div>')
        head[2].html('<div class="sar-rm-head">action unreliable</div>')
        for row_label, quads in (("A = 1", ("trustworthy", "silent_failure")),
                                 ("A = 0", ("lucky", "honest"))):
            cols = st.columns([0.34, 1, 1])
            cols[0].html(f'<div class="sar-rm-label">{row_label}</div>')
            for col, quad in zip(cols[1:], quads):
                with col:
                    st.button(
                        ui.ledger_button_label(quad, rates[quad], int(counts.get(quad, 0))),
                        key=f"q_{quad}",
                        type="primary" if st.session_state.quadrant == quad else "secondary",
                        on_click=_select_quadrant, args=(quad,),
                        width="stretch",
                    )

        stray_note = ui.stray_summary_html(summary)
        if stray_note:
            st.html(stray_note)

    with st.container(border=True):
        st.html(ui.card_title(
            "Risk map · silent-failure rate by workflow region",
            "Task × dataset regions where accepting the language answer would not guarantee a "
            "reliable point action. Darker cells need direct action review. Select a cell to "
            "restrict the audit queue to that region; select it again to clear.",
        ))
        # Prefer the paper's ordering, then append anything the source uses that
        # the paper does not, so an uploaded run with its own task or dataset
        # vocabulary still gets a complete risk map.
        def _axis(column: str, preferred: list[str]) -> list[str]:
            present = list(dict.fromkeys(risk[column].astype(str)))
            return ([v for v in preferred if v in present]
                    + [v for v in present if v not in preferred])

        datasets = _axis("dataset", manifest["dataset_order"])
        tasks = _axis("task", manifest["task_order"])
        lookup = {(str(r.task), str(r.dataset)): r for r in risk.itertuples()}
        max_rate = float(risk["silent_failure_rate"].max()) if len(risk) else 0.0

        head = st.columns([0.62] + [1] * len(datasets))
        for col, dataset in zip(head[1:], datasets):
            col.html(f'<div class="sar-rm-head">{dataset}</div>')

        styles: list[str] = []
        for task in tasks:
            cols = st.columns([0.62] + [1] * len(datasets))
            cols[0].html(f'<div class="sar-rm-label">{task}</div>')
            for col, dataset in zip(cols[1:], datasets):
                cell = lookup.get((task, dataset))
                with col:
                    if cell is None:
                        st.html('<div class="sar-rm-label" style="justify-content:center">—</div>')
                        continue
                    key = f"rm_{task}_{dataset}"
                    selected = st.session_state.region == (task, dataset)
                    styles.append(ui.risk_cell_style(key, cell.silent_failure_rate, max_rate))
                    st.button(
                        ui.risk_cell_label(cell.silent_failure_rate, cell.n),
                        key=key,
                        type="primary" if selected else "secondary",
                        on_click=_select_region, args=(task, dataset),
                        width="stretch",
                        help=f"{task} / {dataset}: VQA accuracy {cell.vqa_acc:.2f}, "
                             f"mean object recall {cell.obj_recall:.2f}, n={cell.n}",
                    )
        ui.inject_css("".join(styles))
        st.html(
            f'<div class="sar-legend"><span>0%</span><div class="bar"></div>'
            f"<span>{100 * max_rate:.1f}%</span></div>"
        )
        small = risk[risk["n"] < 20]
        note = (
            "Cells with fewer than 20 records need closer inspection of the underlying examples"
            f" ({len(small)} of {len(risk)} cells here). "
            if len(small) else ""
        )
        st.caption(
            note + "The Location row folds direct-location and marked-region probes together. "
            "Region rates use every record in the region, including any whose crop image is not "
            "available."
        )


# ------------------------------------------------------------------ right: record audit

with right:
    with st.container(border=True):
        st.html(ui.card_title(
            "Image-region audit",
            "One record at a time: the question a scientist reads, the answer the model gave, and "
            "the point action the workflow would have executed on the same pixels.",
        ))

        queue = scored[scored["quadrant"] == st.session_state.quadrant]
        if st.session_state.region is not None:
            task, dataset = st.session_state.region
            queue = queue[(queue["task"].astype(str) == task) & (queue["dataset"].astype(str) == dataset)]
        # The image filter is only meaningful when some record actually has an
        # image. An uploaded run with no crops would otherwise show an empty
        # queue and hide the audit it came here for.
        images_available = bool(scored["has_image"].any())
        filter_images = st.session_state.images_only and images_available
        if filter_images:
            queue = queue[queue["has_image"]]
        queue = queue.sort_values(["dataset", "task", "question_id"]).reset_index(drop=True)

        # A pinned worked example jumps the cursor to that record, if the current
        # filters still contain it.
        pending = st.session_state.pending_qid
        if pending is not None:
            match = queue.index[queue["question_id"] == pending]
            st.session_state.cursor = int(match[0]) if len(match) else 0

        scope = [QUADRANT_LABEL[st.session_state.quadrant]]
        if st.session_state.region is not None:
            scope.append(" / ".join(st.session_state.region))
        if filter_images:
            scope.append("with crop image")
        elif not images_available:
            scope.append("no crop images in this source")
        st.caption(f"Queue: {' · '.join(scope)} — {len(queue)} record(s)")

        if queue.empty:
            st.info(
                "No records match this combination of behavioural region, workflow region, and "
                "image filter. Clear the risk-map selection or pick another ledger cell."
            )
        else:
            cursor = min(st.session_state.cursor, len(queue) - 1)
            st.session_state.cursor = cursor
            record = queue.iloc[cursor]
            current_record = record
            queue_empty = False

            nav = st.columns([1, 1, 2.4])
            if nav[0].button("‹ prev", width="stretch", disabled=cursor == 0):
                st.session_state.cursor = cursor - 1
                st.rerun()
            if nav[1].button("next ›", width="stretch", disabled=cursor >= len(queue) - 1):
                st.session_state.cursor = cursor + 1
                st.rerun()
            nav[2].html(
                f'<div class="sar-rm-label" style="justify-content:flex-end;min-height:38px">'
                f"{cursor + 1} / {len(queue)}</div>"
            )

            image_path = (image_dir / record["image_file"]
                          if record["image_file"] and image_dir else None)
            current_image_path = image_path
            tile, available = crop_tile(
                image_path, record["gt_centroids"], record["pred_points"],
                palette_name=palette,
            )

            img_col, txt_col = st.columns([0.44, 0.56], gap="medium")
            with img_col:
                st.image(tile, width="stretch")
                st.html(ui.tile_badge_html(available))
                st.html(ui.key_legend_html())
            with txt_col:
                st.html(ui.verdict_html(record["quadrant"], QUADRANT_BLURB[record["quadrant"]]))
                gate = bool(record["action_reliable"])
                st.html(
                    '<div class="sar-note">Hand-off reading: '
                    + (
                        "the answer is correct, so answer-only monitoring would clear this region, "
                        "yet the point action the workflow would run does not pass the gate."
                        if record["quadrant"] == "silent_failure"
                        else "the answer is correct and the point action passes the gate."
                        if record["quadrant"] == "trustworthy"
                        else "the point action passes the gate although the answer is wrong, so the "
                             "answer channel is not tracking action quality here."
                        if record["quadrant"] == "lucky"
                        else "both channels fail, so answer-only monitoring already flags this region."
                    )
                    + "</div>"
                )

            flag = ui.stray_flag_html(record)
            if flag:
                st.html(flag)

            st.html(ui.audit_html(record, gate_reliable=gate, tau=tau, image_available=available))

            # ---------------------------------------------------- sign-off
            log = st.session_state[review_mod.STATE_KEY]
            qid = str(record["question_id"])
            existing = log.get(qid)

            st.html(ui.card_title(
                "Routing decision",
                "The audit ends here. Record what should happen to this point action. Each "
                "decision is saved together with the gate value τ that was in force and the "
                "answer-action state the region was in, so a later reader can tell what it was "
                "accepted against.",
            ))
            route_cols = st.columns(len(review_mod.ROUTES))
            for col, route in zip(route_cols, review_mod.ROUTES):
                with col:
                    st.button(
                        route.label,
                        key=f"route_{route.key}",
                        width="stretch",
                        type=("primary" if existing and existing.route == route.key
                              else "secondary"),
                        on_click=_sign_off,
                        args=(record, build_label, tau, route.key),
                        help=route.blurb,
                    )
            st.text_input(
                "Reason for this decision (optional)",
                key="route_note",
                placeholder="e.g. points sit on a neighbouring organelle, not the mitochondrion",
                help="Saved with the decision and included in the review log and both exports. "
                     "Type it before choosing a routing button.",
            )

            if existing:
                stale = abs(existing.tau - tau) > 1e-9
                st.html(ui.signed_off_html(existing, stale=stale))
                st.button("Clear this decision", key="route_clear",
                          on_click=_clear_sign_off, args=(qid,))

            done, total = review_mod.coverage(log, queue["question_id"])
            st.caption(f"Signed off in this queue: {done} / {total}")

            export = queue.drop(columns=["has_image"]).copy()
            for col in ("gt_centroids", "pred_points", "vqa_choices"):
                export[col] = export[col].map(repr)
            st.download_button(
                "Download this audit queue (CSV)",
                data=export.to_csv(index=False).encode(),
                file_name=f"audit_queue_{condition.replace(' ', '_')}_tau{tau:.2f}.csv",
                mime="text/csv",
                width="stretch",
            )


# ------------------------------------------------------------------ supporting views

st.write("")
sweep_tab, table_tab, log_tab, model_tab = st.tabs(
    ["Threshold sensitivity", "Re-audit across conditions", "Review log", "Live model"]
)

with sweep_tab:
    st.caption(
        f"Both lines below count only the regions where {model} / {condition} answered the "
        "question correctly — that is, everything a supervisor reading answers alone would wave "
        "through. **Green** is the share whose point action also covers enough of the objects. "
        "**Red** is the share where it does not. The x-axis is how strict you make the action "
        "gate: at the far left almost any action counts as good enough, at the far right the "
        "action has to find nearly every object."
    )
    st.line_chart(
        _tau_sweep_frame(scored) if upload_meta else _tau_sweep(build, condition),
        x="τ",
        y=["aligned-pass rate", "silent-failure rate"],
        color=[theme.get(palette).reliable, theme.get(palette).silent],
        height=280,
    )
    st.caption(
        f"Read straight up from the gate you have set (τ = {tau:.2f}) to get the two top numbers "
        "in the ledger above. The red line rising as you move right is the point: demanding more "
        "of the action does not make the answers less trustworthy-looking — it just reveals more "
        "of the regions where a correct answer was never backed by a usable action. Where you put "
        "the gate decides which actions the workflow is allowed to run, so it is a safety setting, "
        "not a chart option."
    )

with table_tab:
    reaudit = "sft_variants"
    reaudit_meta = BUILDS[reaudit]
    st.caption(
        f"The re-audit loop: the same reliability measures recomputed for each of the "
        f"{len(reaudit_meta['conditions'])} supervised conditions in "
        f"**{reaudit_meta['short_label']}**, at τ = {tau:.2f}. A large positive trust gap would "
        "mean answer correctness predicts action reliability; an interval spanning zero means it "
        "does not."
    )
    if build != reaudit:
        st.caption(
            f"This table always reports {reaudit_meta['short_label']}. The current source "
            f"({build_label}) is summarised in the header above."
        )
    table = _conditions_at(reaudit, tuple(reaudit_meta["conditions"]), tau)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Condition": st.column_config.TextColumn("Condition"),
            "n": st.column_config.NumberColumn("n", format="%d"),
            **{
                c: st.column_config.NumberColumn(c, format="%.3f")
                for c in table.columns if c not in ("Condition", "n")
            },
        },
    )

with log_tab:
    log = st.session_state[review_mod.STATE_KEY]
    st.caption(
        "Every routing decision taken in this session, with the gate value and the answer-action "
        "state it was taken under. Decisions live in this session only — export them to keep "
        "them. The JSON export is self-describing: it carries the route vocabulary, the "
        "per-route totals, and a note on every field, so a downstream workflow can consume it "
        "without reference to this app."
    )
    counts = review_mod.counts(log)
    count_cols = st.columns(len(review_mod.ROUTES) + 1)
    for col, route in zip(count_cols, review_mod.ROUTES):
        col.metric(route.label, counts[route.key])
    count_cols[-1].metric("Total", len(log))

    stale_rows = review_mod.stale(log, tau)
    if stale_rows:
        st.warning(
            f"{len(stale_rows)} decision(s) were recorded at a different gate value than the τ = "
            f"{tau:.2f} now in force. They are kept as taken rather than silently re-evaluated."
        )

    if not log:
        st.info(
            "No decisions yet. Open a record in the image-region audit and choose a routing "
            "decision; it will appear here."
        )
    else:
        frame = review_mod.to_frame(log)
        st.dataframe(
            frame[["decided_at", "route_label", "question_id", "dataset", "task", "state",
                   "tau", "obj_recall", "answer_correct", "condition", "source", "note"]],
            hide_index=True, width="stretch",
        )
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            "Download decisions (CSV)", frame.to_csv(index=False).encode(),
            file_name="routing_decisions.csv", mime="text/csv", width="stretch",
        )
        document = review_mod.to_document(log, context={
            "source": build_label,
            "model": model,
            "condition": condition,
            "tau": tau,
            "records_in_source": int(len(scored)),
            "queue_state": st.session_state.quadrant,
        })
        dl2.download_button(
            "Download decisions (JSON)",
            json.dumps(document, indent=2).encode(),
            file_name="routing_decisions.json", mime="application/json", width="stretch",
        )
        dl3.button("Clear all decisions", width="stretch", on_click=_clear_all_sign_offs)


with model_tab:
    endpoint = serve_mod.Endpoint(
        base_url=st.session_state.endpoint_url.rstrip("/"),
        model=st.session_state.endpoint_model,
        api_key=serve_mod.Endpoint.from_env().api_key,
    )
    st.caption(
        "Re-ask the model that is under audit. This closes the loop the rest of the dashboard "
        "only inspects: put the same crop and the same two prompts back to a served checkpoint "
        "and see whether the point action changes. Off unless an endpoint is configured — a "
        "dashboard that quietly answered with a different model than the one being audited "
        "would be worse than one that cannot re-ask at all."
    )
    cfg1, cfg2 = st.columns(2)
    cfg1.text_input("Endpoint base URL", key="endpoint_url",
                    placeholder="http://gpu-host:8000/v1")
    cfg2.text_input("Served model name", key="endpoint_model",
                    placeholder="staged_grpo")

    with st.expander("Serving the checkpoint under audit"):
        st.code(
            "vllm serve Qwen/Qwen3-VL-8B-Instruct \\\n"
            "  --enable-lora \\\n"
            "  --lora-modules staged_grpo=$MODEL_ROOT/checkpoints/"
            "staged_vqa200_grounding_grpo \\\n"
            "  --served-model-name staged_grpo --port 8000",
            language="bash",
        )
        st.caption(
            "Any OpenAI-compatible chat-completions endpoint that accepts image content works. "
            f"The dashboard also reads {serve_mod.ENV_BASE_URL}, {serve_mod.ENV_MODEL} and "
            f"{serve_mod.ENV_KEY} from the environment. Point actions are re-scored by the same "
            "rule as the record set: normalised centroid matching at a 0.1 distance threshold"
            + (", with optimal assignment." if scoring_mod.optimal_assignment_available()
               else ", with the greedy fallback because scipy is not installed.")
        )

    if not endpoint.configured:
        st.info("Enter a base URL and a served model name to enable re-asking.")
    elif queue_empty or current_record is None:
        st.info("Open a record in the image-region audit to re-ask it.")
    else:
        rec = current_record
        st.markdown(
            f"Current record **{rec['crop_id']}** · {rec['task']} / {rec['dataset']} · "
            f"released: A(c)={int(rec['answer_correct'])}, object recall {rec['obj_recall']:.3f}"
        )
        if current_image_path is None:
            st.warning("This record has no crop image in the current source, so it cannot be re-asked.")
        else:
            st.button("Re-ask this record", key="reask", on_click=_reask,
                      args=(rec, current_image_path, tau, endpoint))
            live = st.session_state.live.get(str(rec["question_id"]))
            if live is not None:
                for err in live.errors:
                    st.error(err)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Released record**")
                    st.write({
                        "answer_correct": int(rec["answer_correct"]),
                        "obj_recall": round(float(rec["obj_recall"]), 3),
                        "n_pred": int(rec["n_pred"]),
                        "state": QUADRANT_LABEL[rec["quadrant"]],
                    })
                with col_b:
                    st.markdown("**Live re-ask**")
                    new_state = (
                        None if live.answer_correct is None or live.obj_recall is None
                        else QUADRANT_LABEL[
                            "trustworthy" if live.answer_correct and live.obj_recall >= tau
                            else "silent_failure" if live.answer_correct
                            else "lucky" if live.obj_recall >= tau else "honest"
                        ]
                    )
                    st.write({
                        "answer_letter": live.answer_letter or "—",
                        "answer_correct": live.answer_correct,
                        "obj_recall": (None if live.obj_recall is None
                                       else round(live.obj_recall, 3)),
                        "n_pred": len(live.points),
                        "state": new_state,
                        "latency_s": round(live.latency_s, 2),
                    })
                if live.answer_text:
                    with st.expander("Raw model response"):
                        st.code(live.answer_text)
                st.caption(
                    "A live re-ask is new inference, not part of the audited record set. It is "
                    "shown beside the record rather than replacing it, and it is never written "
                    "into the ledger, the risk map, or the review log."
                )


st.html(
    '<div class="sar-foot">'
    "Green crosses mark labelled objects; red rings mark the points the model would hand to "
    "the workflow. Both are drawn from the stored coordinates — no position is inferred, and "
    "no microscopy image is ever generated."
    "</div>"
)
