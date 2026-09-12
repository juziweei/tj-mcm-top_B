"""Full-continuation one-decision counterfactual audit for Q4.

The deterministic simulator is replayed from the same seed, one high-level
choice is replaced, and the unchanged controller finishes the episode.  Total
episode time is therefore a multi-step return, not a one-step proxy target.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cumcm_b.q4_search import (  # noqa: E402
    Q4DecisionChoice,
    Q4DecisionContext,
    SparseDirectionalSearch,
)
from cumcm_b.simulator import (  # noqa: E402
    InterferenceEnvironment,
    Source,
    _source_visible,
    generate_sources,
)


@dataclass(frozen=True)
class OverrideSpec:
    decision_index: int
    kind: str
    choice: Q4DecisionChoice
    oracle_regret_score: float


@dataclass(frozen=True)
class RolloutResult:
    cleared: int
    sources: int
    virtual_time_s: float
    mean_time_s_per_source: float


def _run_controller(
    sources: Sequence[Source], override: OverrideSpec | None = None
) -> tuple[RolloutResult, list[Q4DecisionContext]]:
    decisions: list[Q4DecisionContext] = []

    def hook(context: Q4DecisionContext) -> Q4DecisionChoice:
        decisions.append(context)
        if override is None or context.decision_index != override.decision_index:
            return None
        if context.kind != override.kind:
            raise RuntimeError("counterfactual replay diverged before override")
        return override.choice

    environment = InterferenceEnvironment(sources)
    result = SparseDirectionalSearch(decision_override=hook).run(environment)
    return (
        RolloutResult(
            cleared=result.cleared_count,
            sources=len(sources),
            virtual_time_s=result.virtual_time_s,
            mean_time_s_per_source=result.virtual_time_s / len(sources),
        ),
        decisions,
    )


def _discovery_score(
    context: Q4DecisionContext,
    point: tuple[float, float],
    sources: Sequence[Source],
) -> float:
    unresolved = [
        source
        for source in sources
        if source.channel not in context.detected_channels
        and source.channel not in context.cleared_channels
    ]
    immediate_detections = sum(_source_visible(source, point) for source in unresolved)
    travel_m = math.dist(context.position, point)
    # One extra discovery dominates any possible within-arena distance saving.
    return immediate_detections * 10_000.0 - travel_m


def _shortlist_overrides(
    decisions: Sequence[Q4DecisionContext],
    sources: Sequence[Source],
    *,
    max_decisions: int,
    alternatives_per_decision: int,
) -> list[OverrideSpec]:
    by_channel = {source.channel: source for source in sources}
    ranked: list[tuple[float, Q4DecisionContext, list[Q4DecisionChoice]]] = []
    for context in decisions:
        if context.kind == "discovery":
            assert context.default_point is not None
            scored = sorted(
                (
                    (_discovery_score(context, point, sources), point)
                    for point in context.candidate_points
                    if point != context.default_point
                ),
                reverse=True,
            )
            if not scored:
                continue
            default_score = _discovery_score(context, context.default_point, sources)
            regret = scored[0][0] - default_score
            choices = [point for _, point in scored[:alternatives_per_decision]]
        else:
            assert context.default_channel is not None
            available = [
                channel
                for channel in context.candidate_channels
                if channel != context.default_channel and channel in by_channel
            ]
            if not available or context.default_channel not in by_channel:
                continue
            default_distance = math.dist(
                context.position, by_channel[context.default_channel].position
            )
            ordered = sorted(
                available,
                key=lambda channel: math.dist(
                    context.position, by_channel[channel].position
                ),
            )
            best_distance = math.dist(context.position, by_channel[ordered[0]].position)
            regret = default_distance - best_distance
            choices = ordered[:alternatives_per_decision]
        if regret > 1e-9:
            ranked.append((regret, context, choices))

    ranked.sort(key=lambda item: item[0], reverse=True)
    specifications: list[OverrideSpec] = []
    for regret, context, choices in ranked[:max_decisions]:
        specifications.extend(
            OverrideSpec(
                decision_index=context.decision_index,
                kind=context.kind,
                choice=choice,
                oracle_regret_score=regret,
            )
            for choice in choices
        )
    return specifications


def _is_better(candidate: RolloutResult, incumbent: RolloutResult) -> bool:
    candidate_complete = candidate.cleared == candidate.sources
    incumbent_complete = incumbent.cleared == incumbent.sources
    if candidate_complete != incumbent_complete:
        return candidate_complete
    if candidate.cleared != incumbent.cleared:
        return candidate.cleared > incumbent.cleared
    return candidate.virtual_time_s < incumbent.virtual_time_s


def _evaluate_episode(arguments: tuple[int, int | None, float, int, int]) -> dict:
    seed, source_count, directional_probability, max_decisions, alternatives = arguments
    sources = generate_sources(
        seed,
        count=source_count,
        directional_probability=directional_probability,
    )
    baseline, decisions = _run_controller(sources)
    specifications = _shortlist_overrides(
        decisions,
        sources,
        max_decisions=max_decisions,
        alternatives_per_decision=alternatives,
    )
    best = baseline
    best_spec: OverrideSpec | None = None
    for specification in specifications:
        candidate, _ = _run_controller(sources, specification)
        if _is_better(candidate, best):
            best = candidate
            best_spec = specification
    return {
        "seed": seed,
        "baseline": asdict(baseline),
        "counterfactual_oracle": asdict(best),
        "recorded_decisions": len(decisions),
        "counterfactual_rollouts": len(specifications),
        "best_override": asdict(best_spec) if best_spec is not None else None,
    }


def _p95(values: Sequence[float]) -> float:
    return sorted(values)[int(0.95 * (len(values) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=94_000_000)
    parser.add_argument("--source-count", type=int)
    parser.add_argument("--directional-probability", type=float, default=0.5)
    parser.add_argument("--max-decisions", type=int, default=2)
    parser.add_argument("--alternatives-per-decision", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_decisions < 1 or args.alternatives_per_decision < 1:
        parser.error("episode and counterfactual limits must be positive")

    started = time.perf_counter()
    work = [
        (
            args.seed_start + offset,
            args.source_count,
            args.directional_probability,
            args.max_decisions,
            args.alternatives_per_decision,
        )
        for offset in range(args.episodes)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(_evaluate_episode, work))

    baseline_times = [row["baseline"]["mean_time_s_per_source"] for row in rows]
    oracle_times = [
        row["counterfactual_oracle"]["mean_time_s_per_source"] for row in rows
    ]
    baseline_complete = [
        row["baseline"]["cleared"] == row["baseline"]["sources"] for row in rows
    ]
    oracle_complete = [
        row["counterfactual_oracle"]["cleared"]
        == row["counterfactual_oracle"]["sources"]
        for row in rows
    ]
    baseline_mean = statistics.fmean(baseline_times)
    oracle_mean = statistics.fmean(oracle_times)
    baseline_p95 = _p95(baseline_times)
    oracle_p95 = _p95(oracle_times)
    report = {
        "config": {
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "source_count": args.source_count,
            "directional_probability": args.directional_probability,
            "max_decisions": args.max_decisions,
            "alternatives_per_decision": args.alternatives_per_decision,
            "workers": args.workers,
        },
        "semantics": (
            "best full-episode result after one privilegedly shortlisted "
            "high-level decision replacement"
        ),
        "baseline_complete_rate": statistics.fmean(baseline_complete),
        "oracle_complete_rate": statistics.fmean(oracle_complete),
        "baseline_mean_s_per_source": baseline_mean,
        "oracle_mean_s_per_source": oracle_mean,
        "mean_reduction_fraction": (baseline_mean - oracle_mean) / baseline_mean,
        "baseline_p95_s_per_source": baseline_p95,
        "oracle_p95_s_per_source": oracle_p95,
        "p95_reduction_fraction": (baseline_p95 - oracle_p95) / baseline_p95,
        "episodes_improved": sum(row["best_override"] is not None for row in rows),
        "counterfactual_rollouts": sum(
            row["counterfactual_rollouts"] for row in rows
        ),
        "wall_time_s": time.perf_counter() - started,
        "gate": {
            "oracle_complete_rate_required": 1.0,
            "mean_reduction_required": 0.20,
            "p95_reduction_required": 0.15,
            "passed": (
                all(oracle_complete)
                and (baseline_mean - oracle_mean) / baseline_mean >= 0.20
                and (baseline_p95 - oracle_p95) / baseline_p95 >= 0.15
            ),
        },
        "episodes": rows,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
