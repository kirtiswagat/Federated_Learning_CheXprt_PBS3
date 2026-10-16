from dataclasses import replace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import ProjectPaths, run_configs
from src.federated.runner import FederatedRunner

pytest.importorskip("cryptography")


def _config(tmp_path, rounds, secure=False):
    config = run_configs(tmp_path)["R01"]
    return replace(
        config,
        paths=ProjectPaths(tmp_path, output_dir=tmp_path / "runs"),
        federated=replace(config.federated, clients=2, rounds=rounds),
        blockchain=replace(config.blockchain, secure=secure),
    )


class _Flaky:
    """Wraps a loader and raises once at batch `fail_at` to simulate a crash."""

    def __init__(self, loader, fail_at):
        self.loader, self.fail_at = loader, fail_at
        self.dataset, self.generator = loader.dataset, loader.generator

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for index, batch in enumerate(self.loader, start=1):
            if self.fail_at == index:
                raise RuntimeError("simulated crash")
            yield batch


def _run(tmp_path, rounds, resume, fail_at=None, checkpoint_every=0, secure=False):
    torch.manual_seed(0)
    loaders = [
        _Flaky(
            DataLoader(
                TensorDataset(torch.randn(8, 4), torch.rand(8, 2).round()),
                batch_size=4,
                shuffle=True,
                generator=torch.Generator(),
            ),
            fail_at if index == 1 else None,
        )
        for index in range(2)
    ]
    run_dir = tmp_path / "runs" / "T"

    def validate(model, round_id, on_batch):
        on_batch(1)
        pd.DataFrame({"x": [round_id]}).to_csv(
            run_dir / "endpoint_valid_predictions.csv", index=False
        )
        per_label = pd.DataFrame({"label": ["a"], "auroc": [0.5]})
        macro = pd.DataFrame([{"macro_auroc": 0.5, "macro_auprc": 0.4}])
        return per_label, macro

    runner = FederatedRunner(
        _config(tmp_path, rounds, secure),
        lambda: torch.nn.Linear(4, 2),
        torch.device("cpu"),
        checkpoint_every=checkpoint_every,
    )
    return runner.run(
        loaders,
        validate,
        torch.nn.BCEWithLogitsLoss(),
        "T",
        resume=resume,
        eval_batches=1,
    )


def test_resume_continues_history_and_ledger(tmp_path):
    _run(tmp_path, rounds=1, resume=False)
    output = _run(tmp_path, rounds=2, resume=True)
    history = pd.read_csv(output / "round_history.csv")
    commitments = pd.read_csv(output / "blockchain_commitment_summary.csv")
    runtime = pd.read_csv(output / "runtime_breakdown.csv")
    assert list(history["round"]) == [1, 2]
    assert len(commitments) == 4
    assert {
        "training_time",
        "signing_time",
        "consensus_time",
        "aggregation_time",
        "evaluation_time",
    } <= set(runtime.operation)


def test_existing_run_requires_resume(tmp_path):
    _run(tmp_path, rounds=1, resume=False)
    with pytest.raises(FileExistsError):
        _run(tmp_path, rounds=1, resume=False)


def test_crash_inside_second_client_resumes_without_losing_finished_work(tmp_path):
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, rounds=2, resume=False, fail_at=2, checkpoint_every=1)
    partial = tmp_path / "runs" / "T" / "partial"
    assert (partial / "round_001_client_0.pt").is_file()
    assert (partial / "mid.pt").is_file()
    output = _run(tmp_path, rounds=2, resume=True, checkpoint_every=1)
    history = pd.read_csv(output / "round_history.csv")
    commitments = pd.read_csv(output / "blockchain_commitment_summary.csv")
    assert list(history["round"]) == [1, 2]
    assert (
        len(commitments) == 4
        and not commitments.duplicated(["round_id", "client_id"]).any()
    )
    assert not partial.exists()


def test_secure_run_resumes_with_persisted_keys_and_valid_ledger(tmp_path):
    import json

    from src.blockchain.secure_ledger import SecureLedger

    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(tmp_path, 2, False, fail_at=2, checkpoint_every=1, secure=True)
    output = _run(tmp_path, 2, True, checkpoint_every=1, secure=True)
    registry = json.loads((output / "ledger_registry.json").read_text())
    blocks = json.loads((output / "ledger.json").read_text())
    ledger = SecureLedger.from_blocks(blocks, registry)
    assert [block.kind for block in ledger.blocks] == [
        "client_update",
        "client_update",
        "round_commitment",
    ] * 2
    assert (output / "identities.json").stat().st_mode & 0o777 == 0o600
