"""Audit the evidence required by the Q3/Q4 optimization goal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--official-summary",
        type=Path,
        default=Path("artifacts/official_practice/q4_run_004/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/goal_completion_audit.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    benchmarks = root / "artifacts" / "benchmarks"
    fixed = _load(benchmarks / "q4_final_paired_fixed16_500.json")
    mixed = _load(benchmarks / "q4_final_paired_mixed_500.json")
    geometry_fixed = _load(
        benchmarks / "q4_final_geometry_paired_fixed16_500.json"
    )
    geometry_mixed = _load(
        benchmarks / "q4_final_geometry_paired_mixed_500.json"
    )
    balanced_fixed = _load(benchmarks / "q4_final_balanced_fixed16_500.json")
    balanced_mixed = _load(benchmarks / "q4_final_balanced_mixed_500.json")
    q3 = _load(benchmarks / "q3_joint_planner_20.json")
    q3_official = _load(
        root / "artifacts" / "official_practice" / "q3_run_002" / "summary.json"
    )
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check(
        "q4_fixed_500_paired",
        fixed["episodes"] == 500,
        {"episodes": fixed["episodes"]},
    )
    check(
        "q4_mixed_500_paired",
        mixed["episodes"] == 500,
        {"episodes": mixed["episodes"]},
    )
    balanced_rows = [*balanced_fixed["rows"], *balanced_mixed["rows"]]
    combined_sources = sum(row["sources"] for row in balanced_rows)
    combined_cleared = sum(row["cleared"] for row in balanced_rows)
    combined_source_clear_rate = combined_cleared / combined_sources
    check(
        "balanced_near_100pct_source_clear",
        combined_source_clear_rate >= 0.999,
        {
            "cleared": combined_cleared,
            "sources": combined_sources,
            "combined_source_clear_rate": combined_source_clear_rate,
        },
    )
    mixed_lower_ci = mixed["paired"][
        "mean_relative_savings_95pct_bootstrap_ci"
    ][0]
    check(
        "balanced_mixed_improvement_gate",
        mixed_lower_ci >= 0.09,
        {
            "mean_relative_savings": mixed["paired"]["mean_relative_savings"],
            "lower_95pct_ci": mixed_lower_ci,
        },
    )
    check(
        "conservative_preserves_fixed_clear_rate",
        geometry_fixed["candidate"]["source_clear_rate"]
        >= geometry_fixed["baseline"]["source_clear_rate"],
        {
            "baseline": geometry_fixed["baseline"]["source_clear_rate"],
            "candidate": geometry_fixed["candidate"]["source_clear_rate"],
        },
    )
    check(
        "conservative_preserves_mixed_clear_rate",
        geometry_mixed["candidate"]["source_clear_rate"]
        >= geometry_mixed["baseline"]["source_clear_rate"],
        {
            "baseline": geometry_mixed["baseline"]["source_clear_rate"],
            "candidate": geometry_mixed["candidate"]["source_clear_rate"],
        },
    )
    check(
        "q3_joint_planner_complete",
        q3["complete_case_rate"] == 1.0 and q3["source_clear_rate"] == 1.0,
        {
            "episodes": q3["episodes"],
            "complete_case_rate": q3["complete_case_rate"],
            "source_clear_rate": q3["source_clear_rate"],
            "mean_time_s_per_source": q3["mean_episode_mean_time_s"],
        },
    )
    check(
        "q3_wall_clock_under_100s",
        q3["max_episode_wall_time_s"] < 100.0,
        {
            "episodes": q3["episodes"],
            "complete_case_rate": q3["complete_case_rate"],
            "max_episode_wall_time_s": q3["max_episode_wall_time_s"],
        },
    )
    check(
        "official_q3_practice",
        q3_official.get("mode") == "official_practice"
        and q3_official.get("cleared_count") == len(q3_official.get("cleared_channels", []))
        and q3_official.get("wall_time_s", 100.0) < 100.0,
        {
            "cleared_count": q3_official.get("cleared_count"),
            "mean_time_s_per_source": q3_official.get(
                "mean_virtual_time_per_cleared_source_s"
            ),
            "wall_time_s": q3_official.get("wall_time_s"),
        },
    )
    q4_max_wall = max(
        _load(benchmarks / name)["max_episode_wall_time_s"]
        for name in (
            "q4_final_geometry_fixed16_500.json",
            "q4_final_geometry_mixed_500.json",
            "q4_final_balanced_fixed16_500.json",
            "q4_final_balanced_mixed_500.json",
        )
    )
    check(
        "q4_wall_clock_under_100s",
        q4_max_wall < 100.0,
        {"maximum_across_final_reports_s": q4_max_wall},
    )

    official_path = (
        args.official_summary
        if args.official_summary.is_absolute()
        else root / args.official_summary
    )
    if official_path.is_file():
        official = _load(official_path)
        official_evidence = {
            "path": str(official_path),
            "mode": official.get("mode"),
            "profile": official.get("profile"),
            "cleared_count": official.get("cleared_count"),
            "official_all_cleared": official.get("official_all_cleared"),
            "wall_time_s": official.get("wall_time_s"),
        }
        official_passed = (
            official.get("mode") == "official_practice"
            and official.get("profile") == "balanced"
            and official.get("official_all_cleared") is True
            and official.get("wall_time_s", 100.0) < 100.0
        )
    else:
        official_evidence = {"path": str(official_path), "missing": True}
        official_passed = False
    check("official_q4_balanced_practice", official_passed, official_evidence)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "passed_checks": sum(item["passed"] for item in checks),
        "total_checks": len(checks),
        "checks": checks,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
