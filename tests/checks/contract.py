"""Shared check contract for the phase validator."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Optional

Status = Literal["pass", "fail", "warn"]


@dataclass
class CheckResult:
    """Two-column result: what the phase produced vs what it should have."""

    status: Status
    desired: str
    current: str
    details: Optional[str] = None


@dataclass
class Check:
    """A registered validator check.

    fn receives the parsed validator_config.yaml dict and returns a CheckResult.
    requires_phases is only used by cross-phase checks: the runner skips the
    check unless every listed phase has previously validated clean.
    """

    check_id: str
    title: str
    fn: Callable[[dict], CheckResult]
    requires_phases: list[int] = field(default_factory=list)
