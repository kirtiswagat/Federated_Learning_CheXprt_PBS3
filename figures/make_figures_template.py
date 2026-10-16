"""Standalone figure script. All numbers below are hard-coded; no result files are needed.

Usage:  python3 make_figures.py [output_dir]
Requires: numpy, matplotlib.
Figure 6 (Grad-CAM) needs the trained model and the images, so it is made by P6/run_p6_gradcam.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# ----------------------------------------------------------------------------------
# Hard-coded results (copied from the evidence pack). Intervals are (estimate, low, high).
# ----------------------------------------------------------------------------------
# DATA_BLOCK

CENTRALISED = {
    "auroc": (0.7846, 0.7813, 0.7879),
    "auprc": (0.3832, 0.3791, 0.3881),
    "auroc5": (0.7880, 0.7844, 0.7914),
}
CLIENT_SIZES = [9638, 61435, 30386, 32768, 18789]

# ----------------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------------
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, GREY = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#7f7f7f",
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures_out")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def forest(ax, labels, values, colors, xlabel, xlim=None, ref=None):
    y = np.arange(len(labels))[::-1]
    for yi, (est, lo, hi), colour in zip(y, values, colors):
        ax.plot([lo, hi], [yi, yi], color=colour, lw=2)
        ax.plot(est, yi, "o", color=colour, ms=5)
        ax.text(
            hi + (xlim[1] - xlim[0]) * 0.015 if xlim else hi,
            yi,
            f"{est:.4f}",
            va="center",
            fontsize=7,
        )
    if ref is not None:
        ax.axvline(ref, color=GREY, ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", alpha=0.25)


# ----------------------------------------------------------------------------------
# Figure 1: study pipeline
# ----------------------------------------------------------------------------------
def figure1():
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, colour):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.4,rounding_size=1.2",
                fc=colour,
                ec="#333333",
                lw=0.8,
            )
        )
        lines = title.count("\n") + 1
        ax.text(
            x + w / 2,
            y + h - 2.0,
            title.strip("\n"),
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
        )
        ax.text(
            x + w / 2,
            y + h - 2.0 - 3.9 * lines - 0.8,
            body,
            ha="center",
            va="top",
            fontsize=7.2,
            linespacing=1.15,
        )

    def down(x, y1, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x, y1),
                (x, y2),
                arrowstyle="simple,head_width=7,head_length=6,tail_width=3",
                mutation_scale=1,
                fc="#555555",
                ec="#555555",
            )
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=9,
                lw=1,
                color="#333333",
            )
        )

    c1, c2, c3, c4 = "#e6f0fa", "#e8f5ea", "#fdf1dc", "#f4e6f0"
    box(
        1,
        80,
        29,
        17,
        "Data",
        "CheXpert-v1.0-small\nfrontal images, 14 labels\nU-Zero labels",
        c1,
    )
    box(
        36,
        80,
        29,
        17,
        "Patient-level split",
        "153,016 training and\n38,011 held-out\nfrontal images",
        c1,
    )
    box(
        71,
        80,
        28,
        17,
        "Partitions",
        "5 simulated clients\nIID, Dirichlet 0.3,\nDirichlet 0.1",
        c1,
    )
    arrow(30.6, 88.5, 35.4, 88.5)
    arrow(65.6, 88.5, 70.4, 88.5)
    down(50, 78.5, 75.0)
    box(
        1, 52, 22, 21, "Centralised\nreference", "ResNet-50\nweighted BCE\n5 epochs", c2
    )
    box(
        27,
        52,
        22,
        21,
        "Federated\ntraining",
        "FedAvg, FedProx\n(mu = 0.001)\n1 local epoch / round",
        c2,
    )
    box(
        53,
        52,
        22,
        21,
        "CBRA-FL",
        "7 settings: weights\nfrom sample size,\nlabel rarity, drift",
        c3,
    )
    box(
        79,
        52,
        20,
        21,
        "Server\naggregation",
        "weighted average\n(robust rules in\nattack study)",
        c3,
    )
    arrow(49.6, 62, 52.4, 62)
    arrow(75.6, 62, 78.4, 62)
    down(50, 50.5, 47.0)
    box(
        1,
        24,
        23,
        22,
        "Signed audit ledger\n(simulated)",
        "update digests\n3 simulated validators\nmajority 2",
        c4,
    )
    box(
        27,
        24,
        23,
        22,
        "Held-out\nevaluation",
        "macro and per-label\nAUROC, AUPRC\noperating metrics",
        c1,
    )
    box(
        53,
        24,
        23,
        22,
        "Statistics",
        "patient-cluster bootstrap\npaired tests\none training seed",
        c1,
    )
    box(
        79,
        24,
        20,
        22,
        "Cost",
        "measured runtime\n(one device)\ncommunication\nvolume estimate",
        c1,
    )
    down(50, 22.5, 19.0)
    box(
        1,
        1,
        31,
        17,
        "Attack study (one round)",
        "4 aggregation rules\n5 attacks\n20% and 40% malicious",
        c3,
    )
    box(36, 1, 29, 17, "Grad-CAM", "illustrative\nqualitative examples", c4)
    box(69, 1, 30, 17, "Outputs", "result tables\nsensitivity tables\nfigures", c1)
    save(fig, "figure1_pipeline")


# ----------------------------------------------------------------------------------
# Figure 2: macro AUROC and AUPRC with intervals
# ----------------------------------------------------------------------------------
def figure2():
    order = [
        "Centralised-5ep",
        "FedAvg-IID, round 5",
        "FedAvg-IID, round 10",
        "FedAvg-Dir0.3",
        "FedAvg-Dir0.1",
        "FedProx-IID",
        "FedProx-Dir0.1",
    ]
    colours = [GREY, BLUE, BLUE, BLUE, BLUE, ORANGE, ORANGE]
    table = dict(MACRO)
    table["Centralised-5ep"] = CENTRALISED
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    forest(
        axes[0],
        order,
        [table[k]["auroc"] for k in order],
        colours,
        "Macro AUROC (14 labels)",
        xlim=(0.775, 0.815),
    )
    forest(
        axes[1],
        order,
        [table[k]["auprc"] for k in order],
        colours,
        "Macro AUPRC (14 labels)",
        xlim=(0.375, 0.415),
    )
    axes[0].set_title("a  AUROC", loc="left")
    axes[1].set_title("b  AUPRC", loc="left")
    fig.text(
        0.5,
        -0.02,
        "Points are estimates; bars are 95% patient-cluster bootstrap intervals "
        "(38,011 held-out frontal images, one training seed).",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout()
    save(fig, "figure2_macro_auroc_auprc")


# ----------------------------------------------------------------------------------
# Figure 3: per-label AUROC with intervals
# ----------------------------------------------------------------------------------
def figure3():
    runs = ["FedAvg-IID, round 5", "FedAvg-Dir0.3", "FedAvg-Dir0.1", "FedProx-Dir0.1"]
    colours = [BLUE, SKY, GREEN, ORANGE]
    labels = list(PER_LABEL_AUROC)
    order = sorted(labels, key=lambda l: PER_LABEL_AUROC[l]["FedAvg-IID, round 5"][0])
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    y = np.arange(len(order))
    offsets = np.linspace(-0.27, 0.27, len(runs))
    for run, colour, off in zip(runs, colours, offsets):
        est = np.array([PER_LABEL_AUROC[l][run][0] for l in order])
        lo = np.array([PER_LABEL_AUROC[l][run][1] for l in order])
        hi = np.array([PER_LABEL_AUROC[l][run][2] for l in order])
        ax.errorbar(
            est,
            y + off,
            xerr=[est - lo, hi - est],
            fmt="o",
            ms=3.2,
            lw=1,
            color=colour,
            label=run,
            capsize=0,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("AUROC (estimate and 95% interval)")
    ax.set_xlim(0.66, 0.9)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Per-label AUROC on the held-out set", loc="left")
    fig.tight_layout()
    save(fig, "figure3_per_label_auroc")


# ----------------------------------------------------------------------------------
# Figure 4: CBRA-FL settings and client weights
# ----------------------------------------------------------------------------------
def figure4():
    names = list(CBRA)
    ref_names = ["FedAvg-Dir0.1", "FedProx-Dir0.1"]
    labels = ref_names + names
    values = [MACRO[n]["auroc"] for n in ref_names] + [CBRA[n]["auroc"] for n in names]
    colours = [BLUE, ORANGE] + [GREEN] * len(names)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    forest(
        axes[0],
        labels,
        values,
        colours,
        "Macro AUROC (Dirichlet 0.1, 5 rounds)",
        xlim=(0.783, 0.800),
    )
    axes[0].set_xticks([0.785, 0.790, 0.795, 0.800])
    axes[0].set_xticklabels(["0.785", "0.790", "0.795", "0.800"])
    axes[0].set_title("a  Macro AUROC", loc="left")
    total = sum(CLIENT_SIZES)
    weights = {"FedAvg": [n / total for n in CLIENT_SIZES]}
    weights.update(CBRA_WEIGHTS)
    names_w = list(weights)[::-1]
    left = np.zeros(len(names_w))
    cmap = [BLUE, SKY, GREEN, ORANGE, RED]
    for k in range(5):
        vals = np.array([weights[n][k] for n in names_w])
        axes[1].barh(
            names_w,
            vals,
            left=left,
            color=cmap[k],
            label=f"client {k}",
            edgecolor="white",
            lw=0.4,
        )
        left += vals
    axes[1].set_xlabel("Aggregation weight")
    axes[1].set_title("b  Client weights", loc="left")
    axes[1].legend(
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=6.5,
    )
    fig.tight_layout()
    save(fig, "figure4_cbra")


# ----------------------------------------------------------------------------------
# Figure 5: attack study
# ----------------------------------------------------------------------------------
def figure5():
    aggs = ["fedavg", "norm_clipped_fedavg", "coordinate_median", "trimmed_mean"]
    agg_names = ["FedAvg", "Norm-\nclipped", "Coord.\nmedian", "Trimmed\nmean"]
    attacks = [
        "byzantine_random",
        "sign_flipping",
        "model_replacement",
        "label_flipping",
        "targeted_backdoor",
    ]
    attack_names = [
        "Byzantine random",
        "Sign flipping",
        "Model replacement",
        "Label flipping",
        "Targeted backdoor",
    ]
    rows, row_names = [], []
    for a, an in zip(attacks, attack_names):
        for frac in (0.2, 0.4):
            rows.append([ATTACK_DROP[(a, frac)][g] for g in aggs])
            row_names.append(f"{an}, {int(frac * 100)}%")
    mat = np.array(rows)
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    norm = TwoSlopeNorm(vmin=-0.03, vcenter=0.0, vmax=0.23)
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(agg_names, fontsize=7)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if abs(mat[i, j]) > 0.12 else "black",
            )
    ax.spines[:].set_visible(False)
    ax.set_title("a  Macro AUROC drop (positive = worse)", loc="left", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(4)
    for k, (frac, colour) in enumerate([(0.2, SKY), (0.4, RED)]):
        sub = [r for r in ASR if r[0] == frac]
        sub = sorted(sub, key=lambda r: aggs.index(r[1]))
        vals = [r[3] for r in sub]
        ax2.bar(
            x + (k - 0.5) * 0.35,
            vals,
            0.33,
            color=colour,
            label=f"{int(frac * 100)}% malicious",
        )
    clean = [CLEAN[g]["asr"] for g in aggs]
    ax2.scatter(
        x, clean, marker="_", s=420, color="black", zorder=3, label="clean model"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(agg_names, fontsize=7)
    ax2.set_ylabel("Backdoor attack success rate")
    ax2.legend(frameon=False, fontsize=7)
    ax2.set_title("b  Backdoor success", loc="left", fontsize=8)
    fig.text(
        0.5,
        -0.01,
        "One round, one local epoch, one seed. Absolute AUROC is not comparable with the "
        "five-round models.",
        ha="center",
        fontsize=7,
    )
    save(fig, "figure5_attack_study")


# ----------------------------------------------------------------------------------
# Figure 6 (runtime) is numbered 7 in the text order: runtime and ledger share
# ----------------------------------------------------------------------------------
def figure_runtime():
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    names = [r[0] for r in RUNTIME][::-1]
    train = np.array([r[2] for r in RUNTIME])[::-1]
    evalh = np.array([r[3] for r in RUNTIME])[::-1]
    share = [r[4] for r in RUNTIME][::-1]
    ax.barh(names, train, color=BLUE, label="Local training")
    ax.barh(names, evalh, left=train, color=ORANGE, label="Evaluation")
    for i, (t, e, s) in enumerate(zip(train, evalh, share)):
        ax.text(t + e + 0.1, i, f"ledger {s:.4f}% of time", va="center", fontsize=7)
    ax.set_xlabel("Hours (one Apple MPS device, clients simulated sequentially)")
    ax.set_xlim(0, 11.5)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "figure_runtime")


# ----------------------------------------------------------------------------------
# Supplementary figure: uncertain labels
# ----------------------------------------------------------------------------------
def figure_uncertain():
    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    rows = sorted([r for r in UNCERTAIN if r[0] != "No Finding"], key=lambda r: r[2])
    axes[0].barh([r[0] for r in rows], [r[2] for r in rows], color=PURPLE)
    axes[0].set_xlabel("Uncertain, % of images with the label mentioned")
    axes[0].set_title("a  Uncertain labels", loc="left")
    runs = list(POLICY)[::-1]
    y = np.arange(len(runs))
    for key, colour, off in [
        ("u_ignore", GREEN, 0.25),
        ("u_zero", BLUE, 0.0),
        ("u_one", RED, -0.25),
    ]:
        est = np.array([POLICY[r][key][0] for r in runs])
        lo = np.array([POLICY[r][key][1] for r in runs])
        hi = np.array([POLICY[r][key][2] for r in runs])
        axes[1].errorbar(
            est,
            y + off,
            xerr=[est - lo, hi - est],
            fmt="o",
            ms=3,
            lw=1,
            color=colour,
            label={
                "u_zero": "U-Zero (primary)",
                "u_ignore": "U-Ignore",
                "u_one": "U-One",
            }[key],
        )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(runs, fontsize=6.5)
    axes[1].set_xlabel("Macro AUROC under three test-label conventions")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.14),
        ncol=3,
        fontsize=6.5,
    )
    axes[1].set_title("b  Re-scoring of saved predictions", loc="left")
    fig.tight_layout()
    save(fig, "figureS1_uncertain_labels")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure_runtime()
    figure_uncertain()