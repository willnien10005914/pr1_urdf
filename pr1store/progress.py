"""Parse Isaac Lab / RSL-RL logs and estimate remaining time.

The website ETA uses completed learning iterations, matching the Q1 Store
rule: a line inside the log may describe one short segment, while the bar
uses the iteration the runner has actually finished.
"""
from __future__ import annotations

import re

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITERATION = re.compile(
    r"(?:Learning iteration|learning iteration|iteration)\s+(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
REWARD = re.compile(r"Mean reward:\s+(-?[0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
EPISODE = re.compile(r"Mean episode length:\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
STEPS = re.compile(
    r"(?:Steps per second|Computation):\s+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
LOG_ETA = re.compile(r"\bETA:\s*([0-9]+:[0-9:]+|[0-9]+(?:\.[0-9]+)?s)")


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def eta_seconds(iteration: int, total: int, elapsed_s: float) -> float | None:
    """Seconds left, estimated from the pace of finished iterations.

    Returns 0 once the run has reached the requested total, and None until
    at least one iteration has finished or the elapsed time is unusable.
    """
    if total <= 0 or elapsed_s < 0 or iteration < 0:
        return None
    if iteration <= 0:
        return None
    if iteration >= total:
        return 0.0
    return elapsed_s * (total - iteration) / iteration


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    remaining = max(0, int(round(seconds)))
    hours, rest = divmod(remaining, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def parse_metrics(text: str) -> dict:
    """Pull the latest iteration, reward, and throughput out of a log tail."""
    metrics: dict = {}
    for line in strip_ansi(text).splitlines():
        match = ITERATION.search(line)
        if match:
            metrics["iteration"] = int(match.group(1))
            metrics["iteration_total"] = int(match.group(2))
        for key, pattern in (
            ("mean_reward", REWARD),
            ("mean_episode_length", EPISODE),
            ("steps_per_second", STEPS),
        ):
            found = pattern.search(line)
            if found:
                metrics[key] = float(found.group(1))
        eta = LOG_ETA.search(line)
        if eta:
            metrics["log_eta"] = eta.group(1)
    return metrics


def train_percent(iteration: int | None, total: int, *, started: bool) -> float:
    """Map a run onto 0–1, leaving the first slice for Isaac Lab startup."""
    if not started:
        return 0.0
    if iteration is None or iteration <= 0:
        return 0.08
    if total <= 0:
        return 0.08
    done = min(1.0, iteration / total)
    return 0.08 + 0.92 * done
