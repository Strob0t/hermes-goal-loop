"""Validation of an untrusted completion claim into concise evidence text for the judge.

A completion claim is a structured object: each criterion has one or more evidence items, and each
check has a ``command``, a ``result`` (``passed``/``failed``/``not-run``) and an optional
``exitCode``. A failed check rejects the whole claim. Validation lives at this tool boundary so a
malformed claim never reaches goal state or the judge prompt. Ported from the method in
``william-ricchiuti/OpenCode-goal-plugin`` (src/completion-claim.js), MIT.
"""
from __future__ import annotations

from typing import Any

MAX_SUMMARY_LENGTH = 500
MAX_CRITERIA = 20
MAX_CHECKS = 20
MAX_CHANGED_FILES = 100
MAX_LIMITATIONS = 20
MAX_CRITERION_LENGTH = 300
MAX_ITEM_LENGTH = 500
MAX_EVIDENCE_PER_CRITERION = 20
MAX_COMPLETION_EVIDENCE_LENGTH = 8000
CHECK_RESULTS = frozenset({"passed", "failed", "not-run"})


class ClaimError(ValueError):
    """A completion claim failed validation; the message is the one actionable reason."""


def _clean_string_list(values: list[Any], field: str) -> list[str]:
    cleaned = [value.strip() if isinstance(value, str) else "" for value in values]
    for value in cleaned:
        if not value or len(value) > MAX_ITEM_LENGTH:
            raise ClaimError(
                f"{field} entries must be non-empty strings of {MAX_ITEM_LENGTH} characters or fewer"
            )
    return cleaned


def serialize_completion_claim(raw: Any = None) -> str:
    """Validate ``raw`` and render the concise evidence text the judge reads.

    Raises :class:`ClaimError` with one actionable reason on any invalid claim.
    """
    claim = raw if isinstance(raw, dict) else {}
    summary = claim.get("summary")
    summary = summary.strip() if isinstance(summary, str) else ""
    if not summary:
        raise ClaimError("summary must be a non-empty string")
    if len(summary) > MAX_SUMMARY_LENGTH:
        raise ClaimError(f"summary must be {MAX_SUMMARY_LENGTH} characters or fewer")

    criteria = claim.get("criteria", []) or []
    checks = claim.get("checks", []) or []
    changed_files = claim.get("changedFiles", []) or []
    known_limitations = claim.get("knownLimitations", []) or []
    for value in (criteria, checks, changed_files, known_limitations):
        if not isinstance(value, list):
            raise ClaimError(
                "criteria, checks, changedFiles, and knownLimitations must be arrays when provided"
            )
    if (len(criteria) > MAX_CRITERIA or len(checks) > MAX_CHECKS
            or len(changed_files) > MAX_CHANGED_FILES or len(known_limitations) > MAX_LIMITATIONS):
        raise ClaimError("completion claim exceeds item limits")

    clean_criteria: list[tuple[str, list[str]]] = []
    for item in criteria:
        evidence_raw = item.get("evidence") if isinstance(item, dict) else None
        if not isinstance(evidence_raw, list) or len(evidence_raw) > MAX_EVIDENCE_PER_CRITERION:
            raise ClaimError(
                f"each criterion requires at most {MAX_EVIDENCE_PER_CRITERION} evidence items"
            )
        criterion = item.get("criterion") if isinstance(item, dict) else None
        criterion = criterion.strip() if isinstance(criterion, str) else ""
        evidence = [v.strip() for v in evidence_raw if isinstance(v, str) and v.strip()]
        if not criterion or not evidence:
            raise ClaimError(
                "each criterion requires a non-empty criterion and at least one evidence item"
            )
        if len(criterion) > MAX_CRITERION_LENGTH or any(len(v) > MAX_ITEM_LENGTH for v in evidence):
            raise ClaimError(
                f"criterion must be {MAX_CRITERION_LENGTH} characters or fewer and evidence items "
                f"{MAX_ITEM_LENGTH} or fewer"
            )
        clean_criteria.append((criterion, evidence))

    clean_checks: list[tuple[str, str, Any, str]] = []
    for item in checks:
        item = item if isinstance(item, dict) else {}
        result = item.get("result")
        result = result.strip() if isinstance(result, str) else ""
        if result not in CHECK_RESULTS:
            raise ClaimError("each check result must be passed, failed, or not-run")
        if result == "failed":
            raise ClaimError("completion cannot include a failed check")
        command = item.get("command")
        command = command.strip() if isinstance(command, str) else ""
        explanation = item.get("explanation")
        explanation = explanation.strip() if isinstance(explanation, str) else ""
        exit_code = item.get("exitCode")
        if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)
                                      or exit_code < 0):
            raise ClaimError("check exitCode must be a non-negative integer")
        if result == "passed" and exit_code is not None and exit_code != 0:
            raise ClaimError("a passed check must have exitCode 0 when provided")
        if result == "not-run" and exit_code is not None:
            raise ClaimError("a not-run check cannot have an exitCode")
        if not command and not explanation:
            raise ClaimError("each check requires a command or explanation")
        if len(command) > MAX_ITEM_LENGTH or len(explanation) > MAX_ITEM_LENGTH:
            raise ClaimError(
                f"check command and explanation must be {MAX_ITEM_LENGTH} characters or fewer"
            )
        clean_checks.append((command, result, exit_code, explanation))

    files = _clean_string_list(changed_files, "changedFiles")
    limitations = _clean_string_list(known_limitations, "knownLimitations")

    lines = [f"Summary: {summary}"]
    for criterion, evidence in clean_criteria:
        lines.append(f"Criterion: {criterion} | Evidence: {'; '.join(evidence)}")
    for command, result, exit_code, explanation in clean_checks:
        subject = command or "manual check"
        details = "; ".join(
            part for part in (("" if exit_code is None else f"exit {exit_code}"), explanation) if part
        )
        lines.append(f"Check: {subject} | {result}{f' | {details}' if details else ''}")
    if files:
        lines.append(f"Changed files: {', '.join(files)}")
    if limitations:
        lines.append(f"Known limitations: {'; '.join(limitations)}")

    evidence_text = "\n".join(lines)
    if len(evidence_text) > MAX_COMPLETION_EVIDENCE_LENGTH:
        raise ClaimError(
            f"completion evidence must be {MAX_COMPLETION_EVIDENCE_LENGTH} characters or fewer"
        )
    return evidence_text
