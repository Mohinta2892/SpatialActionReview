"""Presentation layer: the stylesheet and the static HTML blocks.

The block structure is carried over from `plan/oversight_inspector.html` so the
live app and the paper's dashboard figure read as one system. Everything visual
is expressed against the CSS custom properties emitted by `css()`, so switching
palette is a matter of swapping variable values rather than maintaining two
stylesheets. That includes the Streamlit widget chrome, which is restyled here
because Streamlit's own base theme is fixed at startup and cannot follow an
in-app toggle.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from . import theme as theme_mod
from .data import QUADRANT_LABEL

MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace'

QUADRANT_VAR = {
    "trustworthy": "--reliable",
    "silent_failure": "--silent",
    "lucky": "--lucky",
    "honest": "--honest",
}


def css(palette_name: str = theme_mod.DEFAULT) -> str:
    p = theme_mod.get(palette_name)
    return f"""
<style>
  :root {{
    --ground:{p.ground}; --panel:{p.panel}; --panel2:{p.panel2}; --hair:{p.hair};
    --ink:{p.ink}; --muted:{p.muted}; --dim:{p.dim};
    --reliable:{p.reliable}; --silent:{p.silent}; --lucky:{p.lucky}; --honest:{p.honest};
    --phosphor:{p.phosphor}; --heat-floor:{p.heat_floor}; --on-accent:{p.on_accent};
    --warn:{p.warn};
    --mono:{MONO};
  }}

  /* ---------- Streamlit chrome, restyled against the palette ---------- */
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    background: var(--ground) !important;
  }}
  /* Streamlit's toolbar keeps its startup colours and would sit unreadably on
     the light palette; the app exposes no functionality through it. */
  [data-testid="stToolbar"] {{ display: none !important; }}
  /* The header stays in the layout but must not overlay the workspace: left
     interactive it swallows clicks on the controls at the top of the page. */
  [data-testid="stHeader"] {{
    border-bottom: none; height: 0; min-height: 0; pointer-events: none;
    background: transparent !important;
  }}
  .block-container {{ padding-top: 1.4rem; max-width: 1340px; }}
  [data-testid="stSidebar"] {{
    background: var(--panel) !important; border-right: 1px solid var(--hair);
  }}
  [data-testid="stSidebar"] * {{ color: var(--ink); }}
  [data-testid="stSidebar"] h4 {{ color: var(--ink); }}
  [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
    color: var(--muted) !important;
  }}
  [data-testid="stCaptionContainer"] p {{ font-size: 13px; line-height: 1.55; }}
  .stMarkdown, .stMarkdown p, .stMarkdown li, label, [data-testid="stWidgetLabel"] p {{
    color: var(--ink);
  }}
  hr, [data-testid="stSidebar"] hr {{ border-color: var(--hair); }}

  /* buttons */
  .stButton button, .stDownloadButton button {{
    background: var(--panel2); color: var(--ink);
    border: 1px solid var(--hair); border-radius: 6px;
  }}
  .stButton button:hover, .stDownloadButton button:hover {{ border-color: var(--phosphor); }}
  .stButton button p, .stDownloadButton button p {{ color: var(--ink); }}
  .stButton button:disabled, .stButton button:disabled p {{ opacity: .45; }}

  /* Inputs. baseweb paints the control, its inner value slot and the dropdown
     list from the startup theme, so each needs the palette applied explicitly or
     it stays dark on the light palette. */
  /* Widget controls. Streamlit 1.60 builds the select and text inputs on
     react-aria (data-rac), not baseweb, so these are matched by test id and ARIA
     role — both stable — rather than by data-baseweb or an emotion class hash.
     Only the control gets a fill: a broader selector also painted the widget
     label, which then read as a highlighted strip. */
  [data-testid="stSelectbox"] [data-rac],
  [data-testid="stSelectbox"] [role="group"],
  [data-testid="stTextInputRootElement"],
  [data-baseweb="select"] > div, [data-baseweb="input"] > div,
  [data-baseweb="base-input"] {{
    background-color: var(--panel2) !important; border-color: var(--hair) !important;
  }}
  [data-testid="stSelectbox"] [role="combobox"],
  [data-testid="stTextInputRootElement"] input,
  [data-testid="stSelectbox"] [data-rac] *, [data-baseweb="select"] > div * {{
    background-color: transparent !important; color: var(--ink) !important;
  }}
  [data-testid="stSelectbox"] [role="combobox"]::placeholder,
  [data-testid="stTextInputRootElement"] input::placeholder {{
    color: var(--dim) !important; opacity: 1;
  }}
  /* Chevron only. The help icon is also an svg inside the widget, and it is
     drawn with stroke on fill:none — filling it turns it into a dark disc. */
  [data-testid="stSelectbox"] [role="group"] svg,
  [data-testid="stTextInputRootElement"] svg {{
    fill: var(--muted) !important; color: var(--muted) !important;
  }}
  /* Labels and their help icons sit outside the control and stay unfilled. */
  [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
  [data-testid="stTooltipIcon"], [data-testid="stTooltipIcon"] * {{
    background-color: transparent !important;
  }}
  [data-testid="stWidgetLabel"] p {{ color: var(--ink) !important; }}
  /* The help icon is stroke-on-none geometry; anything that fills it turns it
     into a solid disc, and its button picks up a tinted hover state. */
  [data-testid="stTooltipIcon"] button {{ background: transparent !important; border: none !important; }}
  [data-testid="stTooltipIcon"] svg,
  [data-testid="stTooltipIcon"] svg circle,
  [data-testid="stTooltipIcon"] svg path,
  [data-testid="stTooltipIcon"] svg line {{
    fill: none !important; stroke: var(--dim) !important;
  }}
  [data-testid="stTooltipIcon"] button:hover svg,
  [data-testid="stTooltipIcon"] button:hover svg circle,
  [data-testid="stTooltipIcon"] button:hover svg path,
  [data-testid="stTooltipIcon"] button:hover svg line {{ stroke: var(--phosphor) !important; }}
  [data-testid="stElementToolbarButtonContainer"] {{ background: var(--panel2) !important; }}
  [data-testid="stSelectbox"] svg, [data-baseweb="select"] svg {{ fill: var(--muted) !important; }}
  [data-testid="stTextInput"] input::placeholder,
  [data-baseweb="input"] input::placeholder {{ color: var(--dim) !important; opacity: 1; }}
  /* The dropdown list is portalled to the document root, outside the sidebar, so
     it has to be targeted by its own test id rather than through an ancestor. */
  [data-testid="stSelectboxVirtualDropdown"], [role="listbox"],
  [data-baseweb="popover"] > div, [data-baseweb="menu"] {{
    background-color: var(--panel) !important;
    border: 1px solid var(--hair) !important;
  }}
  [data-testid="stSelectboxVirtualDropdown"] [role="option"], [role="listbox"] [role="option"],
  [data-baseweb="popover"] li, [data-baseweb="menu"] li {{
    background-color: transparent !important; color: var(--ink) !important;
  }}
  [data-testid="stSelectboxVirtualDropdown"] [role="option"] * {{ color: var(--ink) !important; }}
  [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
  [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
  [data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover {{
    background-color: var(--panel2) !important;
  }}
  [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
  [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] * {{
    color: var(--phosphor) !important;
  }}
  [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color: var(--dim); }}

  /* File uploader, code spans and code blocks all take their fill from the
     startup theme, so each shows as a dark slab once the palette is light. */
  [data-testid="stFileUploaderDropzone"] {{
    background: var(--panel2) !important; border: 1px dashed var(--hair) !important;
  }}
  [data-testid="stFileUploaderDropzone"] * {{ color: var(--muted) !important; }}
  [data-testid="stFileUploaderDropzone"] button {{
    background: var(--panel) !important; color: var(--ink) !important;
    border: 1px solid var(--hair) !important;
  }}
  [data-testid="stFileUploaderDropzone"] button * {{ color: var(--ink) !important; }}
  [data-testid="stFileUploaderFile"] {{ background: var(--panel2) !important; }}
  [data-testid="stFileUploaderFile"] * {{ color: var(--ink) !important; }}
  code, .stMarkdown code {{
    background: color-mix(in srgb, var(--phosphor) 10%, var(--panel2)) !important;
    color: var(--ink) !important; border-radius: 3px; padding: 1px 4px;
  }}
  [data-testid="stCode"], [data-testid="stCode"] pre, pre {{
    background: var(--panel2) !important; border: 1px solid var(--hair);
    border-radius: 6px;
  }}
  [data-testid="stCode"] code, pre code {{
    background: transparent !important; color: var(--ink) !important; padding: 0;
  }}
  /* source selector: plain buttons, so the selected state is ours rather than
     baseweb's — its radio markers are filled from the startup theme and cannot
     follow an in-app palette switch. */
  [class*="st-key-src_"] button {{
    min-height:0; padding:7px 10px; text-align:left; align-items:flex-start;
    background:var(--panel2) !important; border:1px solid var(--hair) !important;
  }}
  [class*="st-key-src_"] button p {{
    font-family:var(--mono) !important; font-size:11px !important; margin:0 !important;
    color:var(--muted) !important;
  }}
  [class*="st-key-src_"] button[kind="primary"] {{
    border-color:var(--phosphor) !important;
    background:color-mix(in srgb,var(--phosphor) 15%,transparent) !important;
  }}
  [class*="st-key-src_"] button[kind="primary"] p {{ color:var(--phosphor) !important; }}
  [data-testid="stExpander"] details {{
    background: var(--panel2) !important; border: 1px solid var(--hair); border-radius: 8px;
  }}
  /* Streamlit fills the header of an *open* expander from its startup theme,
     which is the wrong colour once the palette is switched in-app. */
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] details[open] summary {{
    background: var(--panel2) !important;
  }}
  [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{
    color: var(--ink) !important;
  }}
  [data-testid="stExpander"] summary:hover, [data-testid="stExpander"] summary:hover * {{
    color: var(--phosphor) !important;
  }}

  /* ---------- notices (st.warning / error / info / success) ----------
     Streamlit fixes its alert colours at process start, so on the Day palette a
     dark-theme text colour was left sitting on a light tint and the banner became
     unreadable. Both the fill and the text are therefore owned here. The kind is
     read off the inner `stAlertContent<Kind>` test id, since the container itself
     carries no kind attribute. Fills are mixed into `--panel` so they stay opaque
     over either ground. */
  [data-testid="stAlertContainer"] {{
    background: var(--panel2) !important;
    border: 1px solid var(--hair) !important;
    border-left: 3px solid var(--muted) !important;
    border-radius: 8px;
  }}
  [data-testid="stAlertContainer"],
  [data-testid="stAlertContainer"] * {{ color: var(--ink) !important; }}
  [data-testid="stAlertContainer"] a {{ color: var(--phosphor) !important; }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
    background: color-mix(in srgb, var(--warn) 13%, var(--panel)) !important;
    border-color: color-mix(in srgb, var(--warn) 34%, var(--hair)) !important;
    border-left-color: var(--warn) !important;
  }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
    background: color-mix(in srgb, var(--silent) 13%, var(--panel)) !important;
    border-color: color-mix(in srgb, var(--silent) 34%, var(--hair)) !important;
    border-left-color: var(--silent) !important;
  }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
    background: color-mix(in srgb, var(--reliable) 13%, var(--panel)) !important;
    border-color: color-mix(in srgb, var(--reliable) 34%, var(--hair)) !important;
    border-left-color: var(--reliable) !important;
  }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
    background: color-mix(in srgb, var(--lucky) 13%, var(--panel)) !important;
    border-color: color-mix(in srgb, var(--lucky) 34%, var(--hair)) !important;
    border-left-color: var(--lucky) !important;
  }}

  /* palette toggle: two plain buttons so the selected state is ours, not baseweb's */
  [class*="st-key-pal_"] button {{
    min-height:0; padding:5px 0;
    background:var(--panel2) !important; border:1px solid var(--hair) !important;
  }}
  [class*="st-key-pal_"] button p {{
    font-family:var(--mono) !important; font-size:11px !important; margin:0 !important;
    color:var(--muted) !important;
  }}
  [class*="st-key-pal_"] button[kind="primary"] {{
    background:var(--phosphor) !important; border-color:var(--phosphor) !important;
  }}
  [class*="st-key-pal_"] button[kind="primary"] p {{ color:var(--on-accent) !important; }}

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {{
    border-bottom: 1px solid var(--hair); gap: 22px; background: transparent !important;
  }}
  .stTabs [data-baseweb="tab"], .stTabs button[role="tab"] {{
    background: transparent !important; padding-left: 0; padding-right: 0;
  }}
  .stTabs [data-baseweb="tab"] p, .stTabs button[role="tab"] p {{
    color: var(--muted) !important; font-size: 14px !important;
  }}
  .stTabs [data-baseweb="tab"]:hover p, .stTabs button[role="tab"]:hover p {{
    color: var(--ink) !important;
  }}
  .stTabs [aria-selected="true"] p {{ color: var(--phosphor) !important; font-weight: 600 !important; }}
  /* Streamlit's selected-tab underline keeps its startup accent; ours must win. */
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
    background-color: var(--phosphor) !important;
  }}

  /* cards */
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--panel); border-color: var(--hair) !important; border-radius: 10px;
  }}

  /* ---------- header ---------- */
  .sar-header {{
    border:1px solid var(--hair); background:var(--panel); border-radius:12px;
    padding:18px 20px; margin-bottom:18px;
  }}
  .sar-eyebrow {{
    font-family:var(--mono); font-size:11px; letter-spacing:.22em;
    text-transform:uppercase; color:var(--phosphor); margin:0 0 8px;
  }}
  .sar-header h1 {{
    font-size:26px; font-weight:650; letter-spacing:-.01em; margin:0 0 6px;
    line-height:1.15; color:var(--ink); padding:0;
  }}
  .sar-sub {{ color:var(--muted); max-width:80ch; margin:0; font-size:14px; line-height:1.5; }}
  .sar-readout {{ font-family:var(--mono); color:var(--muted); font-size:12px; margin-top:12px;
                  line-height:1.7; }}
  .sar-readout b {{ color:var(--ink); font-weight:600; }}
  /* routing / sign-off */
  /* !important throughout: Streamlit 1.60's emotion rules for a *secondary*
     button beat a plain selector, so an unmarked background or border silently
     does not apply. */
  [class*="st-key-route_"] button {{
    min-height:0; padding:7px 6px;
    border:1px solid var(--hair) !important; background:var(--panel2) !important;
  }}
  [class*="st-key-route_"] button p {{
    font-family:var(--mono) !important; font-size:10.5px !important; margin:0 !important;
    color:var(--ink) !important;
  }}
  .st-key-route_accept button:hover {{ border-color:var(--reliable) !important; }}
  .st-key-route_human_audit button:hover {{ border-color:var(--silent) !important; }}
  .st-key-route_stricter_gate button:hover {{ border-color:var(--lucky) !important; }}
  .st-key-route_model_revision button:hover {{ border-color:var(--phosphor) !important; }}
  [class*="st-key-route_"] button[kind="primary"] {{
    border-color:var(--phosphor) !important;
    background:color-mix(in srgb,var(--phosphor) 16%,transparent) !important;
  }}
  .sar-flag {{
    margin-top:12px; padding:10px 12px; border-radius:6px; font-size:12.5px; line-height:1.5;
    background:color-mix(in srgb,var(--silent) 14%,var(--panel2));
    border:1px solid color-mix(in srgb,var(--silent) 55%,transparent); color:var(--ink);
  }}
  .sar-flag b {{ color:var(--silent); }}
  .sar-signed {{
    font-family:var(--mono); font-size:11px; border-radius:6px; padding:8px 11px;
    margin-top:10px; border:1px solid var(--hair); background:var(--panel2); color:var(--ink);
  }}
  .sar-signed b {{ color:var(--phosphor); }}
  .sar-signed .stale {{ color:var(--silent); }}

  .sar-stats {{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-top:16px; }}
  @media (max-width:900px) {{ .sar-stats {{ grid-template-columns:repeat(3,1fr); }} }}
  @media (max-width:600px) {{ .sar-stats {{ grid-template-columns:repeat(2,1fr); }} }}
  .sar-stat {{ border:1px solid var(--hair); background:var(--panel2); border-radius:8px; padding:9px 10px; }}
  .sar-stat .k {{
    font-family:var(--mono); font-size:10px; text-transform:uppercase;
    letter-spacing:.12em; color:var(--dim);
  }}
  .sar-stat .v {{ font-family:var(--mono); font-size:18px; color:var(--ink); margin-top:2px; }}

  /* ---------- cards ---------- */
  .sar-card-h {{
    font-size:12px; font-family:var(--mono); letter-spacing:.12em; text-transform:uppercase;
    color:var(--muted); margin:0 0 4px; font-weight:500;
  }}
  .sar-hint {{ color:var(--dim); font-size:12px; margin:0 0 12px; line-height:1.45; }}

  /* ---------- ledger + risk-map cells ----------
     Streamlit tags each widget container with .st-key-<key>, which is what lets
     an individual heat cell keep its own background while staying a real
     button. */
  [class*="st-key-q_"] button {{
    width:100%; min-height:74px; border-radius:7px; border:1px solid transparent;
    text-align:left; align-items:flex-start; padding:10px 12px;
  }}
  .st-key-q_trustworthy button {{ background:color-mix(in srgb,var(--reliable) 15%,var(--panel2)); }}
  .st-key-q_silent_failure button {{ background:color-mix(in srgb,var(--silent) 18%,var(--panel2)); }}
  .st-key-q_lucky button {{ background:color-mix(in srgb,var(--lucky) 15%,var(--panel2)); }}
  .st-key-q_honest button {{ background:color-mix(in srgb,var(--honest) 15%,var(--panel2)); }}
  .st-key-q_trustworthy button p:first-child {{ color:var(--reliable) !important; }}
  .st-key-q_silent_failure button p:first-child {{ color:var(--silent) !important; }}
  .st-key-q_lucky button p:first-child {{ color:var(--lucky) !important; }}
  .st-key-q_honest button p:first-child {{ color:var(--honest) !important; }}
  [class*="st-key-q_"] button p:first-child {{
    font-family:var(--mono) !important; font-size:20px !important; font-weight:600 !important;
    margin:0 !important; line-height:1.1 !important;
  }}
  [class*="st-key-q_"] button p:not(:first-child) {{
    font-size:11px !important; color:var(--muted) !important; margin:2px 0 0 !important;
    line-height:1.3 !important;
  }}
  [class*="st-key-q_"] button:hover, [class*="st-key-rm_"] button:hover {{
    border-color:var(--phosphor) !important;
  }}
  [class*="st-key-q_"] button[kind="primary"], [class*="st-key-rm_"] button[kind="primary"] {{
    border:1px solid var(--phosphor) !important; box-shadow:0 0 0 1px var(--phosphor);
  }}

  [class*="st-key-rm_"] button {{
    width:100%; min-height:48px; border-radius:5px; border:1px solid transparent;
    padding:6px 2px;
  }}
  [class*="st-key-rm_"] button p {{
    font-family:var(--mono) !important; margin:0 !important; line-height:1.25 !important;
  }}
  [class*="st-key-rm_"] button p:first-child {{
    font-size:14px !important; font-weight:600 !important; color:var(--ink) !important;
  }}
  [class*="st-key-rm_"] button p:not(:first-child) {{
    font-size:9.5px !important; color:var(--muted) !important;
  }}

  /* worked-example buttons: compact, left-aligned, monospace descriptor */
  [class*="st-key-ex_"] button {{
    min-height:0; padding:5px 9px; text-align:left; align-items:flex-start;
    background:transparent !important; border:1px solid var(--hair) !important;
  }}
  [class*="st-key-ex_"] button p {{
    font-family:var(--mono) !important; font-size:10.5px !important;
    color:var(--muted) !important; margin:0 !important;
  }}
  [class*="st-key-ex_"] button[kind="primary"] {{
    border-color:var(--phosphor) !important;
    background:color-mix(in srgb,var(--phosphor) 14%,transparent) !important;
  }}
  [class*="st-key-ex_"] button[kind="primary"] p {{ color:var(--phosphor) !important; }}

  .sar-rm-label {{
    font-family:var(--mono); font-size:11px; color:var(--muted);
    display:flex; align-items:center; min-height:48px;
  }}
  .sar-rm-head {{
    font-family:var(--mono); font-size:10.5px; color:var(--muted);
    text-align:center; padding:4px 0;
  }}
  .sar-legend {{
    display:flex; align-items:center; gap:8px; margin-top:10px;
    font-family:var(--mono); font-size:10.5px; color:var(--muted);
  }}
  .sar-legend .bar {{
    height:8px; flex:1; border-radius:4px;
    background:linear-gradient(90deg,var(--heat-floor),var(--silent));
    border:1px solid var(--hair);
  }}

  /* ---------- audit panel ---------- */
  .sar-audit {{
    border:1px solid var(--hair); border-radius:8px; background:var(--panel2);
    overflow:hidden; margin-top:12px;
  }}
  .sar-audit-sec {{ padding:11px 12px; border-bottom:1px solid var(--hair); }}
  .sar-audit-sec:last-child {{ border-bottom:none; }}
  .sar-audit-h {{
    font-family:var(--mono); font-size:10.5px; letter-spacing:.12em;
    text-transform:uppercase; color:var(--muted); margin-bottom:7px;
  }}
  .sar-row {{ display:grid; grid-template-columns:132px 1fr; gap:10px; padding:3px 0; font-size:12.5px; }}
  .sar-k {{ font-family:var(--mono); color:var(--dim); }}
  .sar-v {{ color:var(--ink); overflow-wrap:anywhere; }}
  .sar-v.mono {{ font-family:var(--mono); font-size:11.5px; }}
  .sar-v.ok {{ color:var(--reliable); }}
  .sar-v.warn {{ color:var(--silent); }}
  .sar-v.absent {{ color:var(--dim); }}
  .sar-choices {{ display:grid; gap:3px; }}
  .sar-choice {{ display:grid; grid-template-columns:26px 1fr; gap:6px; font-size:12px; color:var(--ink); }}
  .sar-choice .l {{ font-family:var(--mono); color:var(--muted); }}
  .sar-choice.expected {{ color:var(--reliable); }}
  .sar-choice.expected .l {{ color:var(--reliable); }}
  .sar-choice.picked .l {{ color:var(--silent); }}
  .sar-verdict {{ margin-top:12px; padding:10px 12px; border-radius:6px; font-size:12.5px; line-height:1.45; }}

  /* tile provenance badge, directly under the crop */
  .sar-tilebadge {{
    font-family:var(--mono); font-size:10px; letter-spacing:.08em; text-transform:uppercase;
    border-radius:4px; padding:4px 7px; margin-top:7px; display:block; text-align:center;
  }}
  .sar-tilebadge.real {{
    color:var(--reliable); background:color-mix(in srgb,var(--reliable) 13%,transparent);
    border:1px solid color-mix(in srgb,var(--reliable) 40%,transparent);
  }}
  .sar-tilebadge.placeholder {{
    color:var(--silent); background:color-mix(in srgb,var(--silent) 13%,transparent);
    border:1px solid color-mix(in srgb,var(--silent) 45%,transparent);
  }}

  .sar-keylegend {{
    display:flex; gap:16px; margin-top:9px; font-family:var(--mono);
    font-size:10.5px; color:var(--muted); flex-wrap:wrap;
  }}
  /* Legend marks, matching the overlay in sar/render.py: a haloed cross for a
     ground-truth object, a hollow ring for a predicted point. */
  .sar-mark {{
    display:inline-block; position:relative; width:18px; height:18px;
    vertical-align:-5px; margin-right:7px; flex:0 0 auto;
  }}
  .sar-mark-gt {{
    border-radius:50%;
    background:color-mix(in srgb, var(--reliable) 30%, transparent);
  }}
  .sar-mark-gt::before, .sar-mark-gt::after {{
    content:""; position:absolute; left:50%; top:50%; width:12px; height:2.4px;
    border-radius:1.2px; background:var(--reliable);
  }}
  .sar-mark-gt::before {{ transform:translate(-50%,-50%) rotate(45deg); }}
  .sar-mark-gt::after {{ transform:translate(-50%,-50%) rotate(-45deg); }}
  .sar-mark-pred::before {{
    content:""; position:absolute; left:50%; top:50%; width:11px; height:11px;
    transform:translate(-50%,-50%); border:2.4px solid var(--silent); border-radius:50%;
  }}
  .sar-mark-pred::after {{
    content:""; position:absolute; left:50%; top:50%; width:3.2px; height:1.2px;
    transform:translate(-50%,-50%); background:var(--silent);
  }}
  .sar-note {{
    font-family:var(--mono); font-size:11px; color:var(--phosphor);
    border-left:2px solid var(--phosphor); padding:2px 0 2px 9px; margin-top:10px;
    line-height:1.55;
  }}
  .sar-toolbar-label {{
    font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--dim); text-align:right; padding-top:9px;
  }}
</style>
"""


def inject_css(block: str) -> None:
    """Add a <style> block to the page.

    `st.html` sanitises <style> away, so stylesheets have to go through
    `st.markdown(unsafe_allow_html=True)`. Markup blocks still use `st.html`,
    which does not run the text through the markdown parser and therefore cannot
    italicise a crop id that happens to contain underscores.
    """
    st.markdown(block, unsafe_allow_html=True)


def _fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or x != x else f"{x:.{nd}f}"


def pct(x: float, nd: int = 1) -> str:
    return "—" if x is None or x != x else f"{100 * x:.{nd}f}%"


def header_html(
    *, model: str, condition: str, build_label: str, tau: float, summary,
    signed_off: int = 0,
) -> str:
    s = summary
    readout = (
        f"source <b>{escape(build_label)}</b> · model <b>{escape(model)}</b> · "
        f"condition <b>{escape(condition)}</b> · records <b>{s.n:,}</b> · "
        f"gate <b>Rτ(c)=1[obj_recall(c) ≥ {tau:.2f}]</b><br>"
        f"joint silent-failure rate P(A=1, Rτ=0) <b>{_fmt(s.silent_failure_rate)}</b> · "
        f"conditional failure rate P(Rτ=0 | A=1) <b>{_fmt(s.p_unreliable_given_correct)}</b> · "
        f"trust gap Δ <b>{s.trust_gap:+.3f}</b> [{s.gap_lo:+.3f}, {s.gap_hi:+.3f}] · "
        f"point-biserial <b>{s.point_biserial:+.3f}</b>"
    )
    tiles = [
        ("record set", f"{s.n:,}"),
        ("VQA accuracy", pct(s.vqa_acc)),
        ("object recall", pct(s.obj_recall)),
        ("silent failure", pct(s.silent_failure_rate)),
        ("signed off", f"{signed_off:,}"),
    ]
    stats = "".join(
        f'<div class="sar-stat"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>'
        for k, v in tiles
    )
    return f"""
<div class="sar-header">
  <p class="sar-eyebrow">Autonomous EM oversight · language-to-action hand-off audit</p>
  <h1>Spatial Action Review</h1>
  <p class="sar-sub">A supervisor can inspect the language answer, while a downstream workflow may
  consume the paired point action. This dashboard audits that point action before downstream use
  by keeping the MCQ prompt, model answer, expected option, predicted points, and ground-truth object
  evidence on the same electron-microscopy image region.</p>
  <p class="sar-readout">{readout}</p>
  <div class="sar-stats">{stats}</div>
</div>
"""


def toolbar_label(text: str) -> str:
    return f'<div class="sar-toolbar-label">{escape(text)}</div>'


def upload_prompt_html() -> str:
    """Shown when the upload source is selected but no file has arrived yet."""
    return """
<div class="sar-header">
  <p class="sar-eyebrow">Autonomous EM oversight · audit your own run</p>
  <h1>Spatial Action Review</h1>
  <p class="sar-sub">The dashboard is not tied to the runs in this release. It audits any
  vision-language run that exports the paired answer-action record, because every view is
  computed from that record and nothing else.</p>
  <div class="sar-audit" style="margin-top:16px">
    <div class="sar-audit-sec">
      <div class="sar-audit-h">Required per record</div>
      <div class="sar-row"><span class="sar-k">crop_id</span><span class="sar-v">image-region identity, and the image filename stem if you upload crops</span></div>
      <div class="sar-row"><span class="sar-k">dataset, task</span><span class="sar-v">the two axes of the risk map</span></div>
      <div class="sar-row"><span class="sar-k">answer_correct</span><span class="sar-v">A(c) &isin; {0, 1}</span></div>
      <div class="sar-row"><span class="sar-k">obj_recall</span><span class="sar-v">the fraction of ground-truth objects the points covered; the gate verdict R&tau; is recomputed from it</span></div>
      <div class="sar-row"><span class="sar-k">n_gt, n_pred</span><span class="sar-v">ground-truth objects and emitted points</span></div>
    </div>
    <div class="sar-audit-sec">
      <div class="sar-audit-h">Used when present, reported as absent when not</div>
      <div class="sar-row"><span class="sar-k">language channel</span><span class="sar-v">vqa_question, vqa_choices, expected_letter, expected_answer, pred_answer_snippet</span></div>
      <div class="sar-row"><span class="sar-k">action channel</span><span class="sar-v">gt_centroids, pred_points, point_f1, pim_prec, count_ae</span></div>
      <div class="sar-row"><span class="sar-k">meta</span><span class="sar-v">model, condition, tau</span></div>
    </div>
  </div>
  <p class="sar-readout">Use the sidebar to upload an <b>inspector_data.json</b> and, optionally, a
  <b>zip of crops</b> named <b>&lt;crop_id&gt;.png</b>. Nothing is uploaded anywhere else; the file
  is read in this session only.</p>
</div>
"""


def card_title(title: str, hint: str) -> str:
    return f'<p class="sar-card-h">{escape(title)}</p><p class="sar-hint">{hint}</p>'


def ledger_button_label(quadrant: str, rate: float, count: int) -> str:
    return f"{pct(rate)}\n\n{QUADRANT_LABEL[quadrant]}\n\n{count} image regions"


def risk_cell_label(rate: float, n: int) -> str:
    # One decimal place, matching the precision of the manuscript's risk-map
    # table, so a cell never reads as a different number than the paper's.
    return f"{100 * rate:.1f}%\n\nn={n}"


def risk_cell_style(key: str, rate: float, max_rate: float) -> str:
    """Per-cell heat background, keyed to the widget container class."""
    alpha = 0 if max_rate <= 0 else round(80 * rate / max_rate)
    return (
        f"<style>.st-key-{key} button "
        f"{{ background: color-mix(in srgb, var(--silent) {alpha}%, var(--heat-floor)) !important; }}"
        "</style>"
    )


def tile_badge_html(image_available: bool) -> str:
    if image_available:
        return '<span class="sar-tilebadge real">crop image shown</span>'
    return (
        '<span class="sar-tilebadge placeholder">crop image unavailable · '
        "coordinates on a blank grid</span>"
    )


def audit_html(record, *, gate_reliable: bool, tau: float, image_available: bool) -> str:
    choices = record["vqa_choices"]
    expected_letter = (record["expected_letter"] or "").strip().upper()
    predicted = (record["pred_answer_snippet"] or "").strip()
    picked_letter = predicted[:1].upper() if predicted[:1].isalpha() else ""

    if choices:
        items = []
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            classes = ["sar-choice"]
            if letter == expected_letter:
                classes.append("expected")
            if letter == picked_letter and letter != expected_letter:
                classes.append("picked")
            items.append(
                f'<div class="{" ".join(classes)}"><span class="l">{letter}.</span>'
                f"<span>{escape(str(choice))}</span></div>"
            )
        choices_html = f'<div class="sar-choices">{"".join(items)}</div>'
    else:
        choices_html = '<span class="sar-v absent">MCQ options not in this release</span>'

    def points(seq) -> str:
        if not seq:
            return "no coordinates emitted"
        shown = " ".join(f"({float(p[0]):.3f}, {float(p[1]):.3f})" for p in seq[:3])
        return shown + (f" +{len(seq) - 3} more" if len(seq) > 3 else "")

    def metric(value, fmt: str = "{:.3f}") -> str:
        """A missing metric says so; it is never rendered as a zero."""
        if value is None or value != value:
            return ('<span class="sar-v absent">not in this record set yet — it appears here '
                    'as soon as the export carries it</span>')
        return f'<span class="sar-v mono">{fmt.format(value)}</span>'

    def derived(value) -> str:
        """A value this app computes from the stored coordinates."""
        if value is None or value != value:
            return '<span class="sar-v absent">no coordinates to compute it from</span>'
        return (f'<span class="sar-v mono">{value:.3f}</span> '
                '<span class="sar-k">computed here from the coordinates</span>')

    def exported_or_derived(exported, fallback) -> str:
        """Prefer the value scored in the run; fall back to the derived one, labelled."""
        if exported is not None and exported == exported:
            return (f'<span class="sar-v mono">{exported:.3f}</span> '
                    '<span class="sar-k">as scored in the run</span>')
        return derived(fallback)

    def mask_metric(value) -> str:
        """Point-in-mask needs the label masks, which no record set carries."""
        if value is not None and value == value:
            return (f'<span class="sar-v mono">{value:.3f}</span> '
                    '<span class="sar-k">as scored in the run</span>')
        return ('<span class="sar-v absent">needs the label masks, which are not in this '
                "record set — it appears here as soon as the export carries it</span>")

    answer_ok = bool(record["answer_correct"])
    expected_display = (
        f"{expected_letter}. {record['expected_answer']}" if expected_letter
        else record["expected_answer"]
    )
    err = (record["prediction_error"] or "").strip()
    on_target = int(record.get("points_on_target", 0))
    stray = int(record.get("points_stray", 0))
    precision = float(record.get("point_precision_derived", float("nan")))
    f1_derived = float(record.get("point_f1_derived", float("nan")))
    pixels = ("shown above" if image_available
              else "not available — point coordinates shown on a blank grid")

    return f"""
<div class="sar-audit">
  <div class="sar-audit-sec">
    <div class="sar-audit-h">Record</div>
    <div class="sar-row"><span class="sar-k">image region</span><span class="sar-v mono">{escape(record['crop_id'])}</span></div>
    <div class="sar-row"><span class="sar-k">question id</span><span class="sar-v mono">{escape(record['question_id'])}</span></div>
    <div class="sar-row"><span class="sar-k">dataset</span><span class="sar-v">{escape(str(record['dataset']))} <span class="sar-k">({escape(record['dataset_id'])})</span></span></div>
    <div class="sar-row"><span class="sar-k">task</span><span class="sar-v">{escape(str(record['task']))} <span class="sar-k">({escape(record['probe_id'] or 'probe id not exported')})</span></span></div>
    <div class="sar-row"><span class="sar-k">crop image</span><span class="sar-v {'ok' if image_available else 'warn'}">{escape(pixels)}</span></div>
  </div>
  <div class="sar-audit-sec">
    <div class="sar-audit-h">Language channel — what the supervisor reads</div>
    <div class="sar-row"><span class="sar-k">format</span><span class="sar-v">MCQ prompt, K={len(choices)}</span></div>
    <div class="sar-row"><span class="sar-k">question</span><span class="sar-v">{escape(record['vqa_question'])}</span></div>
    <div class="sar-row"><span class="sar-k">options</span><div class="sar-v">{choices_html}</div></div>
    <div class="sar-row"><span class="sar-k">model answer</span><span class="sar-v mono">{escape(predicted) or '<span class="sar-v absent">empty</span>'}</span></div>
    <div class="sar-row"><span class="sar-k">expected</span><span class="sar-v">{escape(expected_display)}</span></div>
    <div class="sar-row"><span class="sar-k">A(c)</span><span class="sar-v {'ok' if answer_ok else 'warn'}">{'correct' if answer_ok else 'wrong'}</span></div>
  </div>
  <div class="sar-audit-sec">
    <div class="sar-audit-h">Action channel — audited before downstream use</div>
    <div class="sar-row"><span class="sar-k">GT centroids</span><span class="sar-v mono">{escape(points(record['gt_centroids']))}</span></div>
    <div class="sar-row"><span class="sar-k">predicted points</span><span class="sar-v mono">{escape(points(record['pred_points']))}</span></div>
    <div class="sar-row"><span class="sar-k">ground-truth objects</span><span class="sar-v">{record['n_gt']}</span></div>
    <div class="sar-row"><span class="sar-k">points emitted</span><span class="sar-v">{record['n_pred']}</span></div>
    <div class="sar-row"><span class="sar-k">points on target</span><span class="sar-v {'ok' if on_target and not stray else ''}">{on_target} of {record['n_pred']}</span></div>
    <div class="sar-row"><span class="sar-k">points on nothing</span><span class="sar-v {'warn' if stray else 'ok'}">{stray}</span></div>
    <div class="sar-row"><span class="sar-k">object recall</span><span class="sar-v mono">{record['obj_recall']:.3f}</span></div>
    <div class="sar-row"><span class="sar-k">hit rate of points</span>{derived(precision)}</div>
    <div class="sar-row"><span class="sar-k">point F1</span>{exported_or_derived(record['point_f1'], f1_derived)}</div>
    <div class="sar-row"><span class="sar-k">point-in-mask prec</span>{mask_metric(record['pim_prec'])}</div>
    <div class="sar-row"><span class="sar-k">count abs error</span>{metric(record['count_ae'], '{:.0f}')}</div>
    <div class="sar-row"><span class="sar-k">action verdict</span><span class="sar-v {'ok' if gate_reliable else 'warn'}">{'passes' if gate_reliable else 'fails'} the gate at τ={tau:.2f} <span class="sar-k">(Rτ)</span></span></div>
    {f'<div class="sar-row"><span class="sar-k">parse note</span><span class="sar-v warn">{escape(err)}</span></div>' if err else ''}
  </div>
</div>
"""


def verdict_html(quadrant: str, blurb: str) -> str:
    var = QUADRANT_VAR[quadrant]
    return (
        f'<div class="sar-verdict" style="background:color-mix(in srgb,var({var}) 15%,var(--panel2));'
        f'color:var({var})"><b>{escape(QUADRANT_LABEL[quadrant])}.</b> {escape(blurb)}</div>'
    )


def key_legend_html() -> str:
    """Legend swatches drawn as the marks they stand for.

    Two filled dots told the reader nothing they could match against the tile, so
    these are the same two marks: a green cross inside a soft halo, and a hollow
    red ring with a centre tick. Built from CSS rather than inline SVG because
    `st.html` sanitises SVG out, the same way it drops <style>. The proportions
    follow `sar/render.py`, where halo : arm : ring radii are 0.072 : 0.032 :
    0.028 of the tile width.
    """
    return (
        '<div class="sar-keylegend">'
        '<span><i class="sar-mark sar-mark-gt"></i>ground truth</span>'
        '<span><i class="sar-mark sar-mark-pred"></i>predicted point action</span>'
        "</div>"
    )


def signed_off_html(disposition, *, stale: bool) -> str:
    """Confirmation of a recorded routing decision, with the state it was taken under."""
    from .review import ROUTE_BY_KEY
    route = ROUTE_BY_KEY[disposition.route]
    note = f' · note: {escape(disposition.note)}' if disposition.note else ""
    warning = (
        '<br><span class="stale">Recorded under a different gate value than the one now '
        "in force.</span>"
        if stale else ""
    )
    return (
        f'<div class="sar-signed"><b>{escape(route.label)}</b> · '
        f"{escape(QUADRANT_LABEL[disposition.state])} at τ={disposition.tau:.2f} "
        f"{escape(disposition.decided_at)}{note}{warning}</div>"
    )


def stray_flag_html(record) -> str:
    """Warn when a record passes the gate yet part of its action set hits nothing."""
    stray = int(record.get("points_stray", 0))
    emitted = int(record["n_pred"])
    on_target = int(record.get("points_on_target", 0))
    if stray <= 0 or emitted <= 0:
        return ""
    cleared = int(record.get("action_reliable", 0)) == 1
    if cleared:
        headline = "Stray actions on a region that passes the gate"
        consequence = (
            f"This region passes the gate on coverage alone, but {stray} of the "
            f"{emitted} predicted points {'land' if stray > 1 else 'lands'} on no labelled "
            f"object. If a later workflow consumed the whole set, {stray} point"
            f"{'s' if stray > 1 else ''} would target empty image."
        )
    else:
        headline = "Stray actions"
        consequence = (
            f"{stray} of the {emitted} emitted points land on no ground-truth object "
            f"({on_target} on target)."
        )
    return f'<div class="sar-flag"><b>{escape(headline)}.</b> {escape(consequence)}</div>'


def stray_summary_html(summary) -> str:
    """One sentence on how much of the cleared action volume lands on nothing."""
    s = summary
    if not s.n_gate_pass or not s.cleared_points:
        return ""
    share = 100 * s.n_stray_despite_pass / s.n_gate_pass
    pt_share = 100 * s.stray_points_gate_pass / s.cleared_points
    return (
        f'<div class="sar-flag">Of the <b>{s.n_gate_pass}</b> regions whose point action '
        f"passes the gate, <b>{s.n_stray_despite_pass}</b> ({share:.0f}%) also emit at least "
        "one point "
        f"that lands on no ground-truth object. Across all predicted points in those cleared "
        f"regions, <b>{s.stray_points_gate_pass} of {s.cleared_points}</b> "
        f"({pt_share:.0f}%) hit nothing. The gate counts coverage only, so it cannot see this."
        "</div>"
    )


def threshold_chart(frame, *, tau: float, palette_name: str = theme_mod.DEFAULT):
    """Threshold sweep with the gate in force drawn on the chart itself.

    A caption telling the reader to "read up from tau" is a caption doing the
    chart's job, so the gate is a labelled rule on the plot and the two series
    are marked where they cross it.
    """
    import altair as alt

    p = theme_mod.get(palette_name)
    domain = ["aligned-pass rate", "silent-failure rate"]
    long = frame.melt("τ", value_vars=domain, var_name="series", value_name="share")
    colours = [p.reliable, p.silent]

    axis = alt.Axis(labelColor=p.muted, titleColor=p.muted, tickColor=p.hair,
                    domainColor=p.hair, gridColor=p.hair, gridOpacity=0.35,
                    labelFont=MONO, titleFont=MONO, titleFontWeight="normal")

    lines = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("τ:Q", title="action-reliability gate τ",
                scale=alt.Scale(domain=[0.05, 0.95], nice=False), axis=axis),
        y=alt.Y("share:Q", title="share of all image regions",
                scale=alt.Scale(domain=[0, float(long["share"].max()) * 1.15]), axis=axis),
        color=alt.Color("series:N",
                        scale=alt.Scale(domain=domain, range=colours),
                        legend=alt.Legend(title=None, orient="top", direction="horizontal",
                                          labelColor=p.ink, labelFont=MONO, labelFontSize=11,
                                          symbolStrokeWidth=3)),
    )

    gate = alt.Chart(frame.iloc[:1].assign(**{"τ": tau})).mark_rule(
        color=p.phosphor, strokeWidth=1.5, strokeDash=[5, 4],
    ).encode(x=alt.X("τ:Q", scale=alt.Scale(domain=[0.05, 0.95], nice=False)))

    label = alt.Chart(frame.iloc[:1].assign(**{"τ": tau})).mark_text(
        text=f"gate in force  τ = {tau:.2f}", align="left", dx=7, dy=-6,
        baseline="top", color=p.phosphor, font=MONO, fontSize=10.5,
    ).encode(x=alt.X("τ:Q", scale=alt.Scale(domain=[0.05, 0.95], nice=False)),
             y=alt.value(4))

    at_gate = long[(long["τ"] - tau).abs() < 1e-9]
    dots = alt.Chart(at_gate).mark_point(size=95, filled=True, opacity=1).encode(
        x=alt.X("τ:Q", scale=alt.Scale(domain=[0.05, 0.95], nice=False)),
        y="share:Q",
        color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=colours), legend=None),
        tooltip=[alt.Tooltip("series:N", title="series"),
                 alt.Tooltip("share:Q", title="share", format=".3f"),
                 alt.Tooltip("τ:Q", title="gate", format=".2f")],
    )

    # Drawn on the page background rather than a panel fill, so the chart does not
    # sit in a visible white box between two captions.
    return (lines + gate + label + dots).properties(height=300).configure_view(
        stroke=None, fill=p.ground,
    ).configure(background=p.ground)
