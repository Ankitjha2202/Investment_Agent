"""Backward-compatible router API — prefer src.assess.assess_question."""

from __future__ import annotations

from src.assess import AssessDecision as RouteDecision
from src.assess import Route, assess_question

# Old name used by graph/docs historically.
route_question = assess_question

__all__ = ["Route", "RouteDecision", "route_question", "assess_question"]
