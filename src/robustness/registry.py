"""Generated P4 clean and attack-by-fraction-by-aggregator condition matrix."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pandas as pd

from src.robustness.config import P4Config, RobustnessCondition

ATTACKS = (
    "label_flipping",
    "sign_flipping",
    "model_replacement",
    "targeted_backdoor",
    "byzantine_random",
)


def build_condition_registry(config: P4Config) -> list[RobustnessCondition]:
    conditions = []
    for attack in ATTACKS:
        for fraction in config.threat.malicious_fractions:
            for aggregator in config.aggregation.methods:
                condition_id = f"{attack}_{int(fraction * 100)}pct_{aggregator}_seed{config.threat.attack_seed}"
                conditions.append(
                    RobustnessCondition(
                        condition_id,
                        attack,
                        fraction,
                        aggregator,
                        config.threat.attack_seed,
                    )
                )
    return conditions


def write_condition_registry(config: P4Config, path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        [asdict(condition) for condition in build_condition_registry(config)]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def select_malicious_clients(config: P4Config, fraction: float) -> list[str]:
    import random

    count = config.threat.malicious_count(fraction)
    clients = list(config.client_ids)
    random.Random(config.threat.client_selection_seed + int(fraction * 100)).shuffle(
        clients
    )
    return sorted(clients[:count])
