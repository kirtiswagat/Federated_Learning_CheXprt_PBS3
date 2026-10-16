"""Grad-CAM examples from a revised-protocol federated endpoint model (qualitative only)."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS  # noqa: E402
from src.data.chexpert_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.models.resnet_multilabel import build_resnet50  # noqa: E402

CORRECT_LABELS = ("Cardiomegaly", "Pleural Effusion", "Pneumothorax")


def pick_cases(predictions: pd.DataFrame, failures: int = 2):
    cases = []
    for label in CORRECT_LABELS:
        rows = predictions[predictions[f"y_{label}"] == 1]
        best = rows.sort_values(f"p_{label}", ascending=False).iloc[0]
        cases.append(("Label present", label, best))
    errors = []
    for label in LABELS:
        rows = predictions[predictions[f"y_{label}"] == 0]
        top = rows.sort_values(f"p_{label}", ascending=False).iloc[0]
        errors.append((float(top[f"p_{label}"]), label, top))
    for _, label, row in sorted(errors, key=lambda item: -item[0])[:failures]:
        cases.append(("Not recorded (U-Zero)", label, row))
    return cases


def grad_cam(model, tensor, class_index):
    store = {}
    layer = model.layer4
    forward = layer.register_forward_hook(lambda m, i, o: store.update(a=o))
    backward = layer.register_full_backward_hook(
        lambda m, gi, go: store.update(g=go[0])
    )
    model.zero_grad()
    model(tensor)[0, class_index].backward()
    forward.remove()
    backward.remove()
    weights = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * store["a"]).sum(dim=1))[0].detach().cpu().numpy()
    cam -= cam.min()
    return cam / cam.max() if cam.max() > 0 else cam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "Output/p1_critical_revision/runs/R03"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "Output/p6_gradcam/gradcam_examples.png"
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.run_dir / "endpoint_valid_predictions.csv")
    model = build_resnet50(14, False, 0.3)
    state = torch.load(args.run_dir / "endpoint_model.pt", map_location="cpu")
    model.load_state_dict(
        state.get("model_state", state)
        if isinstance(state, dict) and "model_state" in state
        else state
    )
    model.eval()

    cases = pick_cases(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(cases), 3, figsize=(10, 3.4 * len(cases)))
    records = []
    for row_axes, (kind, label, row) in zip(axes, cases):
        with Image.open(args.data_root / row["Path"]) as image:
            image = image.convert("RGB").resize((320, 320), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(
            ((pixels - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy()
        )[None].requires_grad_(True)
        index = LABELS.index(label)
        cam = grad_cam(model, tensor, index)
        cam = (
            np.asarray(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(
                    (320, 320), Image.BILINEAR
                )
            )
            / 255.0
        )
        probability = float(row[f"p_{label}"])
        row_axes[0].imshow(pixels)
        row_axes[1].imshow(cam, cmap="jet")
        row_axes[2].imshow(pixels)
        row_axes[2].imshow(cam, cmap="jet", alpha=0.4)
        row_axes[0].set_ylabel(kind, fontsize=11)
        row_axes[2].set_title(f"{label}  p={probability:.2f}", fontsize=11)
        for axis in row_axes:
            axis.set_xticks([])
            axis.set_yticks([])
        records.append(
            {
                "type": kind,
                "label": label,
                "probability": probability,
                "path": row["Path"],
            }
        )
    for axis, title in zip(axes[0], ("Original", "Grad-CAM", "Overlay")):
        axis.set_title(
            title if axis is not axes[0][2] else axis.get_title(), fontsize=11
        )
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    pd.DataFrame(records).to_csv(args.output.with_suffix(".csv"), index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
