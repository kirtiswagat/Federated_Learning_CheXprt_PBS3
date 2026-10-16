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
MACRO = {'FedAvg-IID, round 5': {'auroc': (0.8012, 0.7978, 0.8044),
                         'auprc': (0.3991, 0.3949, 0.4044),
                         'auroc5': (0.7992, 0.7955, 0.8027)},
 'FedAvg-IID, round 10': {'auroc': (0.7976, 0.7946, 0.8007),
                          'auprc': (0.3986, 0.3943, 0.4037),
                          'auroc5': (0.7978, 0.7941, 0.8012)},
 'FedAvg-Dir0.3': {'auroc': (0.7955, 0.7918, 0.7991),
                   'auprc': (0.398, 0.3939, 0.4032),
                   'auroc5': (0.7981, 0.7945, 0.8015)},
 'FedAvg-Dir0.1': {'auroc': (0.7893, 0.7857, 0.7929),
                   'auprc': (0.3912, 0.3875, 0.3961),
                   'auroc5': (0.7987, 0.7951, 0.8022)},
 'FedProx-IID': {'auroc': (0.7976, 0.7941, 0.8009),
                 'auprc': (0.3955, 0.3912, 0.4006),
                 'auroc5': (0.7965, 0.7928, 0.8)},
 'FedProx-Dir0.1': {'auroc': (0.7877, 0.7842, 0.7913),
                    'auprc': (0.3892, 0.3855, 0.394),
                    'auroc5': (0.7951, 0.7914, 0.7985)}}

PER_LABEL_AUROC = {'No Finding': {'FedAvg-IID, round 5': (0.874, 0.867, 0.881),
                'FedAvg-IID, round 10': (0.872, 0.865, 0.879),
                'FedAvg-Dir0.3': (0.873, 0.865, 0.88),
                'FedAvg-Dir0.1': (0.873, 0.866, 0.88),
                'FedProx-IID': (0.872, 0.865, 0.88),
                'FedProx-Dir0.1': (0.872, 0.864, 0.879)},
 'Enlarged Cardiomediastinum': {'FedAvg-IID, round 5': (0.699, 0.685, 0.713),
                                'FedAvg-IID, round 10': (0.689, 0.675, 0.703),
                                'FedAvg-Dir0.3': (0.696, 0.682, 0.71),
                                'FedAvg-Dir0.1': (0.696, 0.682, 0.71),
                                'FedProx-IID': (0.694, 0.68, 0.708),
                                'FedProx-Dir0.1': (0.695, 0.682, 0.709)},
 'Cardiomegaly': {'FedAvg-IID, round 5': (0.874, 0.867, 0.88),
                  'FedAvg-IID, round 10': (0.869, 0.862, 0.875),
                  'FedAvg-Dir0.3': (0.871, 0.864, 0.878),
                  'FedAvg-Dir0.1': (0.871, 0.864, 0.878),
                  'FedProx-IID': (0.872, 0.864, 0.878),
                  'FedProx-Dir0.1': (0.87, 0.863, 0.876)},
 'Lung Opacity': {'FedAvg-IID, round 5': (0.725, 0.719, 0.732),
                  'FedAvg-IID, round 10': (0.724, 0.718, 0.73),
                  'FedAvg-Dir0.3': (0.726, 0.719, 0.732),
                  'FedAvg-Dir0.1': (0.726, 0.72, 0.732),
                  'FedProx-IID': (0.724, 0.718, 0.73),
                  'FedProx-Dir0.1': (0.725, 0.719, 0.731)},
 'Lung Lesion': {'FedAvg-IID, round 5': (0.795, 0.781, 0.81),
                 'FedAvg-IID, round 10': (0.792, 0.778, 0.806),
                 'FedAvg-Dir0.3': (0.792, 0.777, 0.807),
                 'FedAvg-Dir0.1': (0.768, 0.752, 0.784),
                 'FedProx-IID': (0.791, 0.776, 0.807),
                 'FedProx-Dir0.1': (0.771, 0.756, 0.787)},
 'Edema': {'FedAvg-IID, round 5': (0.835, 0.829, 0.84),
           'FedAvg-IID, round 10': (0.833, 0.828, 0.839),
           'FedAvg-Dir0.3': (0.834, 0.828, 0.84),
           'FedAvg-Dir0.1': (0.835, 0.829, 0.84),
           'FedProx-IID': (0.835, 0.829, 0.841),
           'FedProx-Dir0.1': (0.833, 0.827, 0.839)},
 'Consolidation': {'FedAvg-IID, round 5': (0.725, 0.714, 0.737),
                   'FedAvg-IID, round 10': (0.724, 0.713, 0.735),
                   'FedAvg-Dir0.3': (0.723, 0.712, 0.735),
                   'FedAvg-Dir0.1': (0.727, 0.716, 0.739),
                   'FedProx-IID': (0.72, 0.709, 0.732),
                   'FedProx-Dir0.1': (0.72, 0.708, 0.731)},
 'Pneumonia': {'FedAvg-IID, round 5': (0.762, 0.746, 0.779),
               'FedAvg-IID, round 10': (0.747, 0.729, 0.764),
               'FedAvg-Dir0.3': (0.751, 0.733, 0.768),
               'FedAvg-Dir0.1': (0.744, 0.727, 0.761),
               'FedProx-IID': (0.764, 0.747, 0.78),
               'FedProx-Dir0.1': (0.745, 0.727, 0.761)},
 'Atelectasis': {'FedAvg-IID, round 5': (0.695, 0.687, 0.703),
                 'FedAvg-IID, round 10': (0.694, 0.686, 0.702),
                 'FedAvg-Dir0.3': (0.694, 0.686, 0.702),
                 'FedAvg-Dir0.1': (0.691, 0.683, 0.699),
                 'FedProx-IID': (0.691, 0.683, 0.699),
                 'FedProx-Dir0.1': (0.687, 0.679, 0.695)},
 'Pneumothorax': {'FedAvg-IID, round 5': (0.882, 0.874, 0.89),
                  'FedAvg-IID, round 10': (0.879, 0.87, 0.886),
                  'FedAvg-Dir0.3': (0.877, 0.868, 0.886),
                  'FedAvg-Dir0.1': (0.881, 0.873, 0.889),
                  'FedProx-IID': (0.879, 0.87, 0.887),
                  'FedProx-Dir0.1': (0.877, 0.868, 0.886)},
 'Pleural Effusion': {'FedAvg-IID, round 5': (0.868, 0.863, 0.872),
                      'FedAvg-IID, round 10': (0.869, 0.865, 0.874),
                      'FedAvg-Dir0.3': (0.867, 0.862, 0.872),
                      'FedAvg-Dir0.1': (0.869, 0.865, 0.874),
                      'FedProx-IID': (0.864, 0.86, 0.869),
                      'FedProx-Dir0.1': (0.866, 0.861, 0.871)},
 'Pleural Other': {'FedAvg-IID, round 5': (0.827, 0.806, 0.847),
                   'FedAvg-IID, round 10': (0.824, 0.805, 0.843),
                   'FedAvg-Dir0.3': (0.774, 0.749, 0.801),
                   'FedAvg-Dir0.1': (0.717, 0.688, 0.746),
                   'FedProx-IID': (0.821, 0.8, 0.841),
                   'FedProx-Dir0.1': (0.725, 0.7, 0.753)},
 'Fracture': {'FedAvg-IID, round 5': (0.782, 0.761, 0.801),
              'FedAvg-IID, round 10': (0.775, 0.757, 0.792),
              'FedAvg-Dir0.3': (0.785, 0.759, 0.806),
              'FedAvg-Dir0.1': (0.774, 0.752, 0.792),
              'FedProx-IID': (0.771, 0.747, 0.792),
              'FedProx-Dir0.1': (0.768, 0.745, 0.788)},
 'Support Devices': {'FedAvg-IID, round 5': (0.874, 0.869, 0.879),
                     'FedAvg-IID, round 10': (0.876, 0.871, 0.881),
                     'FedAvg-Dir0.3': (0.875, 0.87, 0.88),
                     'FedAvg-Dir0.1': (0.878, 0.873, 0.882),
                     'FedProx-IID': (0.869, 0.864, 0.874),
                     'FedProx-Dir0.1': (0.875, 0.87, 0.88)}}

CBRA = {'CBRA-v1': {'alpha_n': 1.0,
             'beta_q': 1.0,
             'gamma_d': 0.5,
             'auroc': (0.791, 0.7876, 0.7946),
             'auprc': (0.3922, 0.3885, 0.3971),
             'diff_vs_fedavg': (0.0017, 0.001, 0.0025)},
 'CBRA-v2': {'alpha_n': 1.0,
             'beta_q': 1.5,
             'gamma_d': 0.5,
             'auroc': (0.7907, 0.7872, 0.7943),
             'auprc': (0.3906, 0.3869, 0.3956),
             'diff_vs_fedavg': (0.0014, 0.0006, 0.0023)},
 'sample-lower': {'alpha_n': 0.5,
                  'beta_q': 1.0,
                  'gamma_d': 0.5,
                  'auroc': (0.7916, 0.7882, 0.7951),
                  'auprc': (0.3918, 0.388, 0.3968),
                  'diff_vs_fedavg': (0.0023, 0.0014, 0.0033)},
 'sample-higher': {'alpha_n': 1.5,
                   'beta_q': 1.0,
                   'gamma_d': 0.5,
                   'auroc': (0.7877, 0.7842, 0.7914),
                   'auprc': (0.3891, 0.3854, 0.3939),
                   'diff_vs_fedavg': (-0.0016, -0.0025, -0.0007)},
 'rarity-lower': {'alpha_n': 1.0,
                  'beta_q': 0.5,
                  'gamma_d': 0.5,
                  'auroc': (0.7898, 0.7861, 0.7935),
                  'auprc': (0.3916, 0.3878, 0.3964),
                  'diff_vs_fedavg': (0.0005, -0.0007, 0.0017)},
 'drift-off': {'alpha_n': 1.0,
               'beta_q': 1.0,
               'gamma_d': 0.0,
               'auroc': (0.7904, 0.7867, 0.794),
               'auprc': (0.3899, 0.3861, 0.3949),
               'diff_vs_fedavg': (0.0011, -0.0, 0.0022)},
 'drift-strong': {'alpha_n': 1.0,
                  'beta_q': 1.0,
                  'gamma_d': 1.0,
                  'auroc': (0.7923, 0.7888, 0.7958),
                  'auprc': (0.3934, 0.3894, 0.3984),
                  'diff_vs_fedavg': (0.003, 0.002, 0.0041)}}

CBRA_WEIGHTS = {'CBRA-v1': [0.038, 0.369, 0.188, 0.233, 0.172],
 'CBRA-v2': [0.029, 0.35, 0.179, 0.239, 0.202],
 'sample-lower': [0.072, 0.271, 0.195, 0.233, 0.23],
 'sample-higher': [0.019, 0.473, 0.167, 0.218, 0.123],
 'rarity-lower': [0.05, 0.385, 0.195, 0.224, 0.147],
 'drift-off': [0.038, 0.371, 0.188, 0.231, 0.172],
 'drift-strong': [0.038, 0.369, 0.19, 0.231, 0.172]}

ATTACK_DROP = {('byzantine_random', 0.2): {'fedavg': 0.2253,
                             'norm_clipped_fedavg': 0.0285,
                             'coordinate_median': 0.0042,
                             'trimmed_mean': -0.0003},
 ('byzantine_random', 0.4): {'fedavg': 0.2259,
                             'norm_clipped_fedavg': 0.0401,
                             'coordinate_median': 0.0067,
                             'trimmed_mean': 0.2074},
 ('label_flipping', 0.2): {'fedavg': 0.0013,
                           'norm_clipped_fedavg': 0.017,
                           'coordinate_median': 0.0061,
                           'trimmed_mean': 0.0065},
 ('label_flipping', 0.4): {'fedavg': 0.0061,
                           'norm_clipped_fedavg': 0.0099,
                           'coordinate_median': 0.0483,
                           'trimmed_mean': 0.0436},
 ('model_replacement', 0.2): {'fedavg': 0.028,
                              'norm_clipped_fedavg': -0.0,
                              'coordinate_median': -0.0101,
                              'trimmed_mean': -0.0086},
 ('model_replacement', 0.4): {'fedavg': 0.0075,
                              'norm_clipped_fedavg': -0.0167,
                              'coordinate_median': -0.0087,
                              'trimmed_mean': 0.0208},
 ('sign_flipping', 0.2): {'fedavg': 0.0395,
                          'norm_clipped_fedavg': 0.0593,
                          'coordinate_median': 0.0518,
                          'trimmed_mean': 0.0496},
 ('sign_flipping', 0.4): {'fedavg': 0.0213,
                          'norm_clipped_fedavg': 0.0334,
                          'coordinate_median': 0.1068,
                          'trimmed_mean': 0.1093},
 ('targeted_backdoor', 0.2): {'fedavg': -0.0055,
                              'norm_clipped_fedavg': -0.006,
                              'coordinate_median': -0.0057,
                              'trimmed_mean': -0.0032},
 ('targeted_backdoor', 0.4): {'fedavg': -0.0006,
                              'norm_clipped_fedavg': -0.0,
                              'coordinate_median': -0.0047,
                              'trimmed_mean': -0.002}}

CLEAN = {'fedavg': {'auroc': 0.7245, 'auprc': 0.323, 'asr': 0.24},
 'norm_clipped_fedavg': {'auroc': 0.7127, 'auprc': 0.3109, 'asr': 0.155},
 'coordinate_median': {'auroc': 0.7025, 'auprc': 0.2994, 'asr': 0.0203},
 'trimmed_mean': {'auroc': 0.7079, 'auprc': 0.3035, 'asr': 0.0277}}

ASR = [(0.2, 'fedavg', 0.24, 0.238),
 (0.2, 'norm_clipped_fedavg', 0.155, 0.15),
 (0.2, 'coordinate_median', 0.0203, 0.015),
 (0.2, 'trimmed_mean', 0.0277, 0.024),
 (0.4, 'fedavg', 0.24, 0.3097),
 (0.4, 'norm_clipped_fedavg', 0.155, 0.2303),
 (0.4, 'coordinate_median', 0.0203, 0.0877),
 (0.4, 'trimmed_mean', 0.0277, 0.105)]

RUNTIME = [('FedAvg-IID', 10, 8.58, 0.64, 0.0088, 49.6),
 ('FedAvg-Dir0.3', 5, 4.3, 0.32, 0.0089, 49.4),
 ('FedAvg-Dir0.1', 5, 4.34, 0.32, 0.0091, 49.0),
 ('FedProx-IID', 5, 4.34, 0.31, 0.0087, 49.0),
 ('FedProx-Dir0.1', 5, 4.33, 0.31, 0.0088, 49.1)]

UNCERTAIN = [('No Finding', 0.0, 0.0),
 ('Enlarged Cardiomediastinum', 5.48, 29.91),
 ('Cardiomegaly', 3.32, 17.01),
 ('Lung Opacity', 2.38, 4.41),
 ('Lung Lesion', 0.55, 11.88),
 ('Edema', 6.14, 15.34),
 ('Consolidation', 12.59, 42.5),
 ('Pneumonia', 8.12, 70.29),
 ('Atelectasis', 15.57, 49.86),
 ('Pneumothorax', 1.44, 3.95),
 ('Pleural Effusion', 5.03, 8.62),
 ('Pleural Other', 0.99, 41.46),
 ('Fracture', 0.26, 4.91),
 ('Support Devices', 0.55, 0.91)]

POLICY = {'FedAvg-IID, round 5': {'u_zero': (0.8012, 0.7976, 0.8045),
                         'u_ignore': (0.8088, 0.8055, 0.8121),
                         'u_one': (0.7839, 0.7811, 0.7867)},
 'FedAvg-IID, round 10': {'u_zero': (0.7975, 0.7944, 0.8007),
                          'u_ignore': (0.8053, 0.8022, 0.8085),
                          'u_one': (0.7801, 0.7773, 0.7829)},
 'FedAvg-Dir0.3': {'u_zero': (0.7955, 0.7918, 0.7992),
                   'u_ignore': (0.8028, 0.799, 0.8065),
                   'u_one': (0.7778, 0.7745, 0.7808)},
 'FedAvg-Dir0.1': {'u_zero': (0.7893, 0.7854, 0.7931),
                   'u_ignore': (0.7968, 0.7929, 0.8006),
                   'u_one': (0.7743, 0.771, 0.7774)},
 'FedProx-IID': {'u_zero': (0.7976, 0.7942, 0.801),
                 'u_ignore': (0.8053, 0.8018, 0.8086),
                 'u_one': (0.7809, 0.7778, 0.7838)},
 'FedProx-Dir0.1': {'u_zero': (0.7877, 0.7841, 0.7912),
                    'u_ignore': (0.7951, 0.7914, 0.7986),
                    'u_one': (0.7731, 0.7699, 0.7762)},
 'CBRA-v1': {'u_zero': (0.791, 0.7873, 0.7945),
             'u_ignore': (0.7985, 0.7949, 0.802),
             'u_one': (0.7756, 0.7726, 0.7788)},
 'CBRA-v2': {'u_zero': (0.7907, 0.7871, 0.7944),
             'u_ignore': (0.7982, 0.7946, 0.8017),
             'u_one': (0.7751, 0.7722, 0.7781)},
 'sample-lower': {'u_zero': (0.7916, 0.7883, 0.7953),
                  'u_ignore': (0.799, 0.7957, 0.8026),
                  'u_one': (0.7755, 0.7727, 0.7785)},
 'sample-higher': {'u_zero': (0.7877, 0.784, 0.7912),
                   'u_ignore': (0.7953, 0.7915, 0.7987),
                   'u_one': (0.7731, 0.7699, 0.7761)},
 'rarity-lower': {'u_zero': (0.7898, 0.7858, 0.7935),
                  'u_ignore': (0.7975, 0.7935, 0.8012),
                  'u_one': (0.7759, 0.7727, 0.779)},
 'drift-off': {'u_zero': (0.7904, 0.7868, 0.794),
               'u_ignore': (0.798, 0.7944, 0.8015),
               'u_one': (0.7758, 0.7727, 0.7789)},
 'drift-strong': {'u_zero': (0.7923, 0.7888, 0.7957),
                  'u_ignore': (0.7999, 0.7964, 0.8033),
                  'u_one': (0.7772, 0.7743, 0.78)}}

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