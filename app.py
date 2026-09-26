"""Agentic Streamlit chat UI for the investment chatbot."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat import ask  # noqa: E402
from src.graph import NODES, PIPELINE_ORDER  # noqa: E402

# ---------------------------------------------------------------------------
# Page + visual system
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Investment Agent",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --ink: #0f1c2e;
    --muted: #5a6a7e;
    --line: #d8e0ea;
    --panel: #f4f7fb;
    --accent: #0d6e6e;
    --accent-soft: #e4f3f3;
    --sql: #1a5f8a;
    --meet: #7a4a12;
    --both: #4a3d8f;
    --ok: #1b7a4a;
    --err: #a33b3b;
    --warn: #8a6d1a;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
  }

  .block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1100px;
  }

  /* Hero */
  .agent-hero {
    background:
      radial-gradient(1200px 280px at 10% -20%, #d7f0ef 0%, transparent 55%),
      radial-gradient(900px 240px at 90% 0%, #e8eef8 0%, transparent 50%),
      linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 1.35rem 1.5rem 1.2rem;
    margin-bottom: 1rem;
  }
  .agent-hero h1 {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
  }
  .agent-hero p {
    color: var(--muted);
    margin: 0;
    font-size: 0.98rem;
    line-height: 1.45;
    max-width: 52rem;
  }
  .pipeline {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.95rem;
    align-items: center;
  }
  .pipe-chip {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    padding: 0.28rem 0.6rem;
    border-radius: 6px;
    background: var(--panel);
    color: var(--ink);
    border: 1px solid var(--line);
  }
  .pipe-arrow {
    color: #9aa8b8;
    font-size: 0.8rem;
  }

  /* Route badges */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    border: 1px solid transparent;
  }
  .badge-sql { background: #e8f3fa; color: var(--sql); border-color: #c5dceb; }
  .badge-meetings { background: #f8efe3; color: var(--meet); border-color: #ead7bc; }
  .badge-both { background: #eceaf6; color: var(--both); border-color: #d4d0ea; }

  /* Agent timeline */
  .trace-wrap {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.85rem 1rem 0.95rem;
    margin: 0.35rem 0 0.75rem;
  }
  .trace-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.65rem;
  }
  .step {
    display: grid;
    grid-template-columns: 22px 1fr;
    gap: 0.55rem;
    padding: 0.45rem 0;
    border-bottom: 1px solid #e4ebf3;
  }
  .step:last-child { border-bottom: none; padding-bottom: 0; }
  .step-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-top: 0.35rem;
    border: 2px solid #b8c4d4;
    background: #fff;
  }
  .step-dot.running {
    border-color: var(--accent);
    background: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-soft);
  }
  .step-dot.done { border-color: var(--ok); background: var(--ok); }
  .step-dot.error { border-color: var(--err); background: var(--err); }
  .step-dot.skipped { border-color: #c5ced9; background: #e8edf3; }
  .step-dot.pending { border-color: #c5ced9; background: #fff; }
  .step-head {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.45rem;
  }
  .step-name {
    font-weight: 600;
    color: var(--ink);
    font-size: 0.92rem;
  }
  .step-status {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }
  .step-status.running { color: var(--accent); }
  .step-status.done { color: var(--ok); }
  .step-status.error { color: var(--err); }
  .step-detail {
    margin-top: 0.2rem;
    font-size: 0.86rem;
    color: var(--muted);
    line-height: 1.4;
    white-space: pre-wrap;
  }
  .step-detail code, .step-detail pre {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.78rem;
  }

  /* Evidence cards */
  .evidence-grid {
    display: grid;
    gap: 0.65rem;
  }
  .ev-card {
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    background: #fff;
  }
  .ev-meta {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
  }
  .ev-title {
    font-weight: 600;
    color: var(--ink);
    font-size: 0.9rem;
  }
  .ev-body {
    font-size: 0.84rem;
    color: #3d4d61;
    margin-top: 0.3rem;
    line-height: 1.4;
  }
  .score-pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: var(--accent-soft);
    color: var(--accent);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-weight: 500;
  }

  /* Sidebar polish */
  section[data-testid="stSidebar"] {
    background: #f7f9fc;
  }
  section[data-testid="stSidebar"] .stButton > button {
    text-align: left;
    justify-content: flex-start;
    white-space: normal;
    height: auto;
    padding: 0.55rem 0.7rem;
    font-size: 0.84rem;
    border: 1px solid var(--line);
    background: #fff;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
  }

  div[data-testid="stChatMessage"] {
    border-radius: 12px;
  }

  /* Hide default streamlit chrome noise slightly */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROUTE_LABELS = {
    "sql": ("SQL path", "badge-sql", "Structured metrics via text-to-SQL"),
    "meetings": ("Meetings path", "badge-meetings", "Semantic search over meeting notes"),
    "both": ("Hybrid path", "badge-both", "SQL metrics + meeting narrative"),
}

_SQL_PRE_STYLE = (
    "margin:0.4rem 0 0;padding:0.55rem 0.65rem;background:#0f1c2e;"
    "color:#d7e6f5;border-radius:8px;overflow-x:auto;white-space:pre-wrap;"
)


def _esc(text: Any) -> str:
    return (
        str(text if text is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_detail(raw: str) -> str:
    """Convert light markdown in step details to HTML (fences before inline ticks)."""
    fences: list[str] = []

    def _park_fence(match: re.Match[str]) -> str:
        fences.append(match.group(1).strip())
        return f"@@FENCE{len(fences) - 1}@@"

    text = re.sub(r"```(?:sql)?\s*(.*?)```", _park_fence, raw or "", flags=re.DOTALL)
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = text.replace("\n", "<br>")
    for i, code in enumerate(fences):
        text = text.replace(
            f"@@FENCE{i}@@",
            f'<pre style="{_SQL_PRE_STYLE}">{_esc(code)}</pre>',
        )
    return text


def _route_badge(route: str) -> str:
    label, cls, _ = ROUTE_LABELS.get(route, (route, "badge-both", ""))
    return f'<span class="badge {cls}">{label}</span>'


def _render_timeline(
    steps: dict[str, dict[str, str]],
    *,
    title: str | None = "Agent plan",
) -> None:
    """Render agent steps as HTML.

    Important: do not indent the HTML string. Streamlit markdown treats lines
    starting with 4 spaces as a code block, which showed raw <div> tags in the UI.
    Pass title=None when the surrounding expander already provides the label.
    """
    parts = ['<div class="trace-wrap">']
    if title:
        parts.append(f'<div class="trace-title">{_esc(title)}</div>')
    for sid in PIPELINE_ORDER:
        if sid not in steps:
            continue
        meta = steps[sid]
        status = meta.get("status", "pending")
        label = NODES.get(sid, {}).get("title", sid)
        detail = _format_detail(meta.get("detail", "") or "")
        parts.append(
            f'<div class="step">'
            f'<div class="step-dot {status}"></div>'
            f"<div>"
            f'<div class="step-head">'
            f'<span class="step-name">{_esc(label)}</span>'
            f'<span class="step-status {status}">{status}</span>'
            f"</div>"
            f'<div class="step-detail">{detail}</div>'
            f"</div></div>"
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def _render_collapsible_timeline(
    steps: dict[str, dict[str, str]],
    *,
    label: str = "Agent plan & execution",
    expanded: bool = False,
) -> None:
    """Render the agent timeline inside a closable expander."""
    with st.expander(label, expanded=expanded):
        _render_timeline(steps, title=None)


def _render_evidence(debug: dict[str, Any]) -> None:
    if not debug:
        return

    route = debug.get("route") or "—"
    reason = debug.get("route_reason") or ""
    client_id = debug.get("client_id")
    top_k = debug.get("top_k")
    _, _, route_hint = ROUTE_LABELS.get(route, ("", "", ""))

    client_badge = (
        f'<span class="badge badge-sql">client {_esc(client_id)}</span>'
        if client_id
        else ""
    )
    top_k_badge = (
        f'<span class="badge badge-meet">top_k {_esc(top_k)}</span>'
        if top_k is not None and route in {"meetings", "both"}
        else ""
    )
    st.markdown(
        '<div style="margin:0.2rem 0 0.6rem;display:flex;flex-wrap:wrap;'
        'gap:0.5rem;align-items:center;">'
        f"{_route_badge(route)}"
        f"{top_k_badge}"
        f'<span style="color:#5a6a7e;font-size:0.88rem;">{_esc(route_hint)}</span>'
        f"{client_badge}"
        "</div>",
        unsafe_allow_html=True,
    )
    if reason:
        st.caption(f"Assessor: {reason}")

    sql = debug.get("sql") or {}
    meetings = debug.get("meetings") or {}

    tabs = []
    if sql:
        tabs.append("SQL evidence")
    if meetings:
        tabs.append("Meeting evidence")
    tabs.append("Raw debug")

    tlist = st.tabs(tabs)
    ti = 0

    if sql:
        with tlist[ti]:
            if sql.get("error"):
                st.error(sql["error"])
            if sql.get("rationale"):
                st.markdown(f"**Why this query:** {sql['rationale']}")
            if sql.get("query"):
                st.code(sql["query"], language="sql")
            rows = sql.get("sample_rows") or []
            if rows:
                st.caption(f"Showing {len(rows)} of {sql.get('row_count', len(rows))} rows")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            elif not sql.get("error"):
                st.info("Query returned no rows.")
        ti += 1

    if meetings:
        with tlist[ti]:
            if meetings.get("error"):
                st.error(meetings["error"])
            hits = meetings.get("hits") or []
            if hits:
                requested = debug.get("top_k")
                req_note = f" (requested top_k={requested})" if requested else ""
                st.caption(
                    f"{meetings.get('hit_count', len(hits))} notes retrieved"
                    f"{req_note} — ranked by cosine similarity"
                )
                cards = ['<div class="evidence-grid">']
                for h in hits:
                    cards.append(
                        f'<div class="ev-card">'
                        f'<div class="ev-meta">'
                        f'<span class="score-pill">{float(h.get("score") or 0):.3f}</span>'
                        f"&nbsp; {_esc(h.get('meeting_date') or '—')}"
                        f" · {_esc(h.get('sector') or '—')}"
                        f" · {_esc(h.get('region') or '—')}"
                        f" · client {_esc(h.get('client_id') or '—')}"
                        f"</div>"
                        f'<div class="ev-title">{_esc(h.get("company") or "Unknown company")}</div>'
                        f'<div class="ev-body">{_esc(h.get("summary") or "")}</div>'
                        f"</div>"
                    )
                cards.append("</div>")
                st.markdown("".join(cards), unsafe_allow_html=True)
            elif not meetings.get("error"):
                st.info("No meeting notes matched.")
        ti += 1

    with tlist[ti]:
        st.json(debug)


def _empty_steps() -> dict[str, dict[str, str]]:
    return {sid: {"status": "pending", "detail": "…"} for sid in PIPELINE_ORDER}


def _run_agent(question: str, timeline_slot) -> dict[str, Any]:
    """Run ask() while live-updating the agent timeline."""
    steps = _empty_steps()

    def on_step(step_id: str, status: str, detail: str = "") -> None:
        steps[step_id] = {"status": status, "detail": detail}
        with timeline_slot.container():
            _render_timeline(steps, title="Agent working…")

    with timeline_slot.container():
        _render_timeline(steps, title="Agent working…")

    result = ask(question, show_debug=True, on_step=on_step)

    settled = {
        sid: meta for sid, meta in steps.items() if meta.get("status") != "skipped"
    }
    with timeline_slot.container():
        _render_collapsible_timeline(
            settled,
            label="Agent plan & execution",
            expanded=True,
        )

    return {
        "answer": result.answer,
        "debug": result.debug,
        "steps": settled,
        "route": result.route,
        "route_reason": result.route_reason,
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### How the agent works")
    st.markdown(
        """
1. **Assess** — brain decides `sql` / `meetings` / `both` and meeting `top_k`
2. **Act** — conditional edges run text-to-SQL and/or pgvector retrieval
3. **Synthesize** — answers only from retrieved evidence
4. **Improve** — critiques quality; self-heals and retries up to 2×

Entities: **Client · Group · Deal · RM**
"""
    )
    show_trace = st.toggle("Show agent plan & evidence", value=True)
    st.markdown("---")
    st.markdown("**Example questions**")
    st.caption("Click to run. Grouped by route stress / edge case.")

    example_groups: list[tuple[str, list[str]]] = [
        (
            "Basics",
            [
                "What is the total USD invested by client A12345?",
                "Which RM has the highest total investment amount?",
                "What diligence concerns came up for fintech deals in EMEA?",
                "How is client A12345 performing on IRR and what was discussed in recent meetings?",
            ],
        ),
        (
            "SQL — aggregations & ranking",
            [
                "Top 10 clients by total USD invested, excluding realised deals.",
                "Which LoB (COP vs RE vs HF) has the highest average investment size?",
                "For each RM, show deal count and total USD invested; rank by deal count.",
                "Show group-level AUM for the largest 5 groups (is_group = true).",
                "Clients whose CI current IRR is above 15% and total AUM exceeds $50M.",
            ],
        ),
        (
            "SQL — dates, filters, joins",
            [
                "How much did client A12345 invest in 2022 vs 2023?",
                "List unrealised investments for client B12463 with invested_date and deal_name.",
                "Compare CI total IRR vs RE total IRR for client A12345 using the latest as_of_date.",
                "Which clients have CI MOIC above 2x but RE current IRR below 5%?",
                "Find deals where investment_amount_natural is in EUR and convert totals to USD.",
            ],
        ),
        (
            "Meetings — thematic & breadth",
            [
                "Summarize all governance or ESG concerns raised across healthcare meetings.",
                "List every company and meeting date mentioned for client A12345 — be exhaustive.",
                "What action items were assigned after Series B / growth-stage diligence calls?",
                "Any red flags about valuation, churn, or unit economics in SaaS deals in APAC?",
                "Who did the team meet about climate-tech or renewables, and what was the outcome?",
            ],
        ),
        (
            "Hybrid — numbers + narrative",
            [
                "Client A12345: total invested USD, latest CI IRR, and what meetings said about performance.",
                "Which high-AUM clients also had diligence concerns about leverage or covenants?",
                "For the RM with the largest book, what themes show up in their clients' recent meetings?",
                "Compare invested amounts in fintech vs what meeting notes say about fintech risk.",
                "Client B12463 last met date from performance vs what the meeting notes actually covered.",
            ],
        ),
        (
            "Edge cases — routing & traps",
            [
                # Ambiguous / both-leaning
                "Is A12345 a good investment? Justify with data and any qualitative concerns.",
                # Company name must NOT become client_id
                "What was discussed about Stripe or similar payments companies in meetings?",
                # High top_k pressure
                "Give me a complete list of everyone client A12345 met, with dates and companies.",
                # Likely empty / wrong filter → improve should heal
                "Meeting notes for client Z99999 about quantum computing in Antarctica.",
                # Out of scope / refuse gracefully
                "Delete all investments for client A12345 and email the RM.",
                # Ambiguous entity (name vs ID)
                "How is the Smith Family Office doing on IRR and what came up in diligence?",
                # Multi-intent packed question
                "Break down A12345 by LoB USD invested, show CI vs RE IRR, and list meeting action items.",
                # Self-heal bait: over-narrow meetings filter
                "What did we discuss with portfolio companies for client A12345 in Latin America fintech?",
            ],
        ),
    ]

    btn_i = 0
    for group_title, questions in example_groups:
        with st.expander(group_title, expanded=(group_title == "Basics")):
            for q in questions:
                if st.button(q, key=f"ex_{btn_i}", use_container_width=True):
                    st.session_state["prefill"] = q
                btn_i += 1

    st.markdown("---")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="agent-hero">'
    "<h1>Investment Data Agent</h1>"
    "<p>Ask about clients, deals, RMs, performance, or meeting notes. "
    "Watch the agent assess, gather evidence, synthesize, then self-heal if needed.</p>"
    '<div class="pipeline">'
    '<span class="pipe-chip">1 · Assess</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-chip">2 · SQL / Meetings</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-chip">3 · Synthesize</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-chip">4 · Improve</span>'
    "</div></div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and show_trace and msg.get("steps"):
            _render_collapsible_timeline(
                msg["steps"],
                label="Agent plan & execution",
                expanded=False,
            )
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_trace and msg.get("debug"):
            with st.expander("Evidence & debug", expanded=False):
                _render_evidence(msg["debug"])

prefill = st.session_state.pop("prefill", None)
prompt = st.chat_input("Ask about clients, deals, RMs, performance, or meetings…")
if prefill:
    prompt = prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        timeline_slot = st.empty()
        answer_slot = st.empty()
        evidence_slot = st.empty()
        try:
            payload = _run_agent(prompt, timeline_slot)
            answer_slot.markdown(payload["answer"])
            if show_trace:
                with evidence_slot.expander("Evidence & debug", expanded=True):
                    _render_evidence(payload["debug"])
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": payload["answer"],
                    "debug": payload["debug"],
                    "steps": payload["steps"],
                }
            )
        except Exception as e:
            err = f"Error: {e}"
            answer_slot.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
