"""Phase validator — current-vs-desired checks per phase.

Usage:
    python -m tests.phase_validator --phase 1
    python -m tests.phase_validator --phase all
    python -m tests.phase_validator --check P1.03
    python -m tests.phase_validator --cross-phase
    python -m tests.phase_validator --phase 3 --verbose

Exit codes: 0 = all pass (warnings allowed), 1 = failures exist.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

import yaml

from tests.checks.contract import Check, CheckResult

PHASE_TITLES = {
    1: "Data Foundry",
    2: "OCR & Degradation",
    3: "Ingestion & Indexing",
    4: "Agentic Pipeline",
    5: "Evaluation",
    6: "Dashboard",
    7: "Deployment",
}

HEAVY = "═" * 51
LIGHT = "─" * 51
STATUS_LABELS = {
    "pass": "✅ PASS",
    "fail": "❌ FAIL",
    "warn": "⚠️  WARN (non-blocking, log for review)",
}

CONFIG_PATH = Path(__file__).parent / "validator_config.yaml"
STATE_PATH = Path(__file__).parent / ".validator_state.json"


def load_config() -> dict:
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def load_phase_checks(phase: int) -> list[Check]:
    module = importlib.import_module(f"tests.checks.phase{phase}")
    return list(module.CHECKS)


def load_cross_checks() -> list[Check]:
    module = importlib.import_module("tests.checks.cross_phase")
    return list(module.CHECKS)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_phase_state(phase: int, passed: bool) -> None:
    state = load_state()
    state[f"phase_{phase}"] = passed
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def run_check(check: Check, config: dict) -> CheckResult:
    try:
        return check.fn(config)
    except Exception as exc:  # a crashing check is a failing check, not a traceback
        return CheckResult(
            status="fail",
            desired="check executes without error",
            current=f"check raised {type(exc).__name__}: {exc}",
            details=traceback.format_exc(limit=3),
        )


def render_check(check: Check, result: CheckResult, verbose: bool) -> None:
    print(f"[{check.check_id}] {check.title}")
    if verbose or result.status != "pass":
        print(f"  DESIRED: {result.desired}")
        print(f"  CURRENT: {result.current}")
        if result.details:
            for line in result.details.rstrip().splitlines():
                print(f"    {line}")
    print(f"  STATUS:  {STATUS_LABELS[result.status]}")
    print()


def cross_checks_for_phase(phase: int) -> list[Check]:
    """Cross-phase checks run with the latest phase they depend on."""
    return [c for c in load_cross_checks() if c.requires_phases and max(c.requires_phases) == phase]


def run_phase(phase: int, config: dict, verbose: bool) -> bool:
    checks = load_phase_checks(phase) + cross_checks_for_phase(phase)
    print(HEAVY)
    print(f" PHASE {phase} — {PHASE_TITLES[phase]} Validation")
    print(HEAVY)
    print()

    if not checks:
        print(f"  No checks implemented yet for phase {phase}.")
        print(f"  STATUS:  {STATUS_LABELS['fail']}")
        print()
        print(LIGHT)
        print(f" PHASE {phase} RESULT: 0/0 passed — no checks registered")
        print(f" ❌ PHASE NOT COMPLETE — implement phase {phase} checks first")
        print(HEAVY)
        save_phase_state(phase, False)
        return False

    passed = failed = warned = 0
    for check in checks:
        result = run_check(check, config)
        render_check(check, result, verbose)
        if result.status == "pass":
            passed += 1
        elif result.status == "fail":
            failed += 1
        else:
            warned += 1

    blocking_total = passed + failed
    summary = f" PHASE {phase} RESULT: {passed}/{blocking_total} passed, {failed} failed"
    if warned:
        summary += f", {warned} warning{'s' if warned > 1 else ''}"
    print(LIGHT)
    print(summary)
    if failed:
        print(f" ❌ PHASE NOT COMPLETE — fix failures before Phase {phase + 1}")
    else:
        print(f" ✅ PHASE {phase} COMPLETE")
    print(HEAVY)
    save_phase_state(phase, failed == 0)
    return failed == 0


def run_all(config: dict, verbose: bool) -> bool:
    for phase in sorted(PHASE_TITLES):
        ok = run_phase(phase, config, verbose)
        print()
        if not ok:
            print(f"Stopping at first failing phase (phase {phase}).")
            return False
    return True


def run_cross_phase(config: dict, verbose: bool) -> bool:
    state = load_state()
    print(HEAVY)
    print(" CROSS-PHASE Validation")
    print(HEAVY)
    print()
    failed = 0
    for check in load_cross_checks():
        unmet = [p for p in check.requires_phases if not state.get(f"phase_{p}", False)]
        if unmet:
            print(f"[{check.check_id}] {check.title}")
            print(f"  STATUS:  ⏭  SKIP (upstream phase(s) {unmet} not validated yet)")
            print()
            continue
        result = run_check(check, config)
        render_check(check, result, verbose)
        if result.status == "fail":
            failed += 1
    print(LIGHT)
    print(f" CROSS-PHASE RESULT: {failed} failure(s)")
    print(HEAVY)
    return failed == 0


def find_check(check_id: str) -> Check | None:
    normalized = check_id.replace("->", "→").upper()
    for phase in PHASE_TITLES:
        for check in load_phase_checks(phase):
            if check.check_id.upper() == normalized:
                return check
    for check in load_cross_checks():
        if check.check_id.upper() == normalized:
            return check
    return None


def run_single(check_id: str, config: dict) -> bool:
    check = find_check(check_id)
    if check is None:
        print(f"No check with ID {check_id!r} is registered.")
        return False
    result = run_check(check, config)
    render_check(check, result, verbose=True)
    return result.status != "fail"


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.phase_validator", description=__doc__)
    parser.add_argument("--phase", help="phase number 1-7, or 'all'")
    parser.add_argument("--check", help="run one check by ID, e.g. P1.03")
    parser.add_argument("--cross-phase", action="store_true", help="run cross-phase checks only")
    parser.add_argument("--verbose", action="store_true", help="show CURRENT vs DESIRED for passes too")
    args = parser.parse_args()

    modes = [bool(args.phase), bool(args.check), args.cross_phase]
    if sum(modes) != 1:
        parser.error("choose exactly one of --phase, --check, --cross-phase")

    config = load_config()
    if args.check:
        ok = run_single(args.check, config)
    elif args.cross_phase:
        ok = run_cross_phase(config, args.verbose)
    elif args.phase == "all":
        ok = run_all(config, args.verbose)
    else:
        try:
            phase = int(args.phase)
        except ValueError:
            parser.error("--phase must be 1-7 or 'all'")
        if phase not in PHASE_TITLES:
            parser.error("--phase must be 1-7 or 'all'")
        ok = run_phase(phase, config, args.verbose)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
