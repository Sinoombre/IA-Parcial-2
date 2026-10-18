from pathlib import Path

import pytest

from churnshield.scenario_report import build_scenario_report, write_scenario_report


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SCENARIOS = ROOT / "artifacts" / "public" / "scenarios"
EVALUATION = ROOT / "artifacts" / "public" / "scenario_evaluation" / "evaluation.json"


def test_build_report_aggregates_public_scenarios():
    report = build_scenario_report(
        PUBLIC_SCENARIOS,
        evaluation_path=EVALUATION,
        expected_runs={
            (scenario, seed)
            for scenario in (
                "S02",
                "S03",
                "S04",
                "S05",
                "S06",
                "S07",
                "S08",
                "S09",
                "S10",
            )
            for seed in (101, 102, 103, 104, 105)
        },
    )

    assert report["runs"] == 45
    assert report["scenarios"] == 9
    assert report["evaluation"]["m2_passes"] is True
    assert report["evaluation"]["m3_passes"] is True
    s10 = next(row for row in report["scenario_summaries"] if row["scenario"] == "S10")
    assert s10["runs"] == 5
    assert s10["crm_statuses"]["rejected"] > 0
    assert s10["crm_statuses"]["error"] > 0


def test_write_report_creates_json_csv_and_markdown(tmp_path):
    report = build_scenario_report(PUBLIC_SCENARIOS, evaluation_path=EVALUATION)
    paths = write_scenario_report(report, tmp_path / "report")

    for path in paths.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0
    assert "Resumen ejecutivo" in Path(paths["markdown"]).read_text(encoding="utf-8")


def test_build_report_rejects_missing_expected_run():
    with pytest.raises(ValueError, match="Ejecuciones inesperadas"):
        build_scenario_report(
            PUBLIC_SCENARIOS,
            expected_runs={("S02", 101)},
        )
