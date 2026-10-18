"""Rutas separadas para resultados operativos y verdad del evaluador."""

from pathlib import Path


def evaluator_output_path(
    output_dir: str | Path,
    evaluator_output_dir: str | Path | None,
) -> Path:
    public_path = Path(output_dir).expanduser().resolve()
    if evaluator_output_dir is None:
        evaluator_path = public_path.parent / "evaluator_truth"
    else:
        evaluator_path = Path(evaluator_output_dir).expanduser()
    evaluator_path = evaluator_path.resolve()

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
