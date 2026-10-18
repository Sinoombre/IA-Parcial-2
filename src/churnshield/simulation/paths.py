"""Rutas separadas para resultados operativos y verdad del evaluador."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_EVALUATOR_ROOT = REPO_ROOT / "artifacts" / "private" / "evaluator_truth"


def evaluator_output_path(
    output_dir: str | Path,
    evaluator_output_dir: str | Path | None,
) -> Path:
    public_path = Path(output_dir).expanduser().resolve()
    if evaluator_output_dir is None:
        evaluator_path = PRIVATE_EVALUATOR_ROOT
    else:
        evaluator_path = Path(evaluator_output_dir).expanduser().resolve()

    private_root = PRIVATE_EVALUATOR_ROOT.resolve()
    try:
        evaluator_path.relative_to(private_root)
    except ValueError as exc:
        raise ValueError(
            "evaluator_output_dir debe estar dentro de "
            "artifacts/private/evaluator_truth"
        ) from exc

    if (
        public_path == evaluator_path
        or public_path in evaluator_path.parents
        or evaluator_path in public_path.parents
    ):
        raise ValueError(
            "output_dir y evaluator_output_dir deben ser carpetas separadas "
            "sin relación padre-hijo"
        )

    return evaluator_path
