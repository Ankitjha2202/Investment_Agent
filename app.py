"""Minimal Streamlit chat UI for the investment chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chat import ask  # noqa: E402

st.set_page_config(page_title="Investment Chatbot", page_icon="💬", layout="centered")

st.title("Investment Data Chatbot")
st.caption(
    "Hybrid RAG over investments, performance (text-to-SQL) and meeting notes (pgvector)."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        """
**Entities:** Client · Group · Deal · RM

**Routing**
- Structured numbers → text-to-SQL
- Meeting narrative → embedding retrieval
- Mixed → both, then synthesize

Toggle debug to see route + generated SQL.
"""
    )
    show_debug = st.checkbox("Show debug panel", value=True)
    st.markdown("---")
    st.markdown("**Try asking**")
    examples = [
        "What is the total USD invested by client A12345?",
        "Which RM has the highest total investment amount?",
        "What diligence concerns came up for fintech deals in EMEA?",
        "How is client A12345 performing on IRR and what was discussed in recent meetings?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug") and show_debug:
            with st.expander("Debug"):
                st.json(msg["debug"])

prefill = st.session_state.pop("prefill", None)
prompt = st.chat_input("Ask about clients, deals, RMs, performance, or meetings…")
if prefill:
    prompt = prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = ask(prompt, show_debug=True)
                st.markdown(result.answer)
                if show_debug:
                    with st.expander("Debug"):
                        st.json(result.debug)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "debug": result.debug,
                    }
                )
            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
