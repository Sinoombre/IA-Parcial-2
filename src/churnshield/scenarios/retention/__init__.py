"""Registro de escenarios sintéticos de retención."""

from .s02 import SCENARIO as S02
from .s03 import SCENARIO as S03
from .s04 import SCENARIO as S04
from .s05 import SCENARIO as S05
from .s06 import SCENARIO as S06
from .s07 import SCENARIO as S07
from .s08 import SCENARIO as S08
from .s09 import SCENARIO as S09
from .s10 import SCENARIO as S10
from .fixtures import build_fixture

SCENARIO_CONFIGS = {
    scenario.scenario_id: scenario
    for scenario in (S02, S03, S04, S05, S06, S07, S08, S09, S10)
}
SCENARIOS = tuple(SCENARIO_CONFIGS)

__all__ = ["SCENARIOS", "SCENARIO_CONFIGS", "build_fixture"]
