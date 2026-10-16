import pandas as pd

from src.p2.budget import processed_budget


def test_processed_budget_is_cumulative():
    result = processed_budget(
        pd.DataFrame(
            {
                "round": [1, 2],
                "processed_examples": [10, 20],
                "communication_bytes": [100, 200],
            }
        )
    )
    assert result.cumulative_processed_examples.tolist() == [10, 30]
