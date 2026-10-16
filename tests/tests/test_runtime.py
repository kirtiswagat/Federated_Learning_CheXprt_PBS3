from src.monitoring.runtime import RuntimeTracker


def test_runtime_tracker_records_named_operation():
    tracker = RuntimeTracker()
    with tracker.measure("evaluation_time"):
        pass
    assert tracker.dataframe().iloc[0]["operation"] == "evaluation_time"
