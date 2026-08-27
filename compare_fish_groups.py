from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_ROOT = Path("combined_analysis_output")
OUTPUT_FOLDER = INPUT_ROOT / "fish_group_comparison"

COLORS = {
    1: "#007C91",
    2: "#E76F51",
    4: "#2A9D8F",
    8: "#E9C46A",
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.color": "#CBD5E1",
    "grid.alpha": 0.55,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
})


def style_axis(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55, color="#CBD5E1")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def find_groups():
    groups = {}
    for folder in sorted(INPUT_ROOT.iterdir()):
        if not folder.is_dir() or not folder.name.endswith("_fish"):
            continue
        try:
            fish_count = int(folder.name.removesuffix("_fish"))
        except ValueError:
            continue
        groups[fish_count] = folder
    return groups


def load_group_array(folder, filename):
    path = folder / filename
    if not path.exists():
        return None
    return np.asarray(np.load(path), dtype=float)


def plot_autocorrelation(groups):
    fig, ax = plt.subplots(figsize=(9, 5))
    for fish_count, folder in groups.items():
        values = load_group_array(folder, "velocity_autocorrelation_mean.npy")
        if values is None or values.ndim != 2 or values.shape[0] != 2:
            continue
        valid = np.isfinite(values[0]) & np.isfinite(values[1])
        ax.plot(values[0, valid], values[1, valid], linewidth=2, color=COLORS.get(fish_count, "#264653"),
                label=fr"$N={fish_count}$ fish")
    ax.set_title("Mean velocity temporal autocorrelation")
    ax.set_xlabel(r"Lag time [$\mathrm{s}$]")
    ax.set_ylabel(r"$C_v(\tau) / C_v(0)$")
    style_axis(ax)
    ax.legend()
    fig.savefig(OUTPUT_FOLDER / "autocorrelation_by_fish_count.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_msd(groups):
    fig, ax = plt.subplots(figsize=(8, 5))
    for fish_count, folder in groups.items():
        lag_times = load_group_array(folder, "mean_msd_lag_times.npy")
        msd = load_group_array(folder, "mean_msd.npy")
        if lag_times is None or msd is None:
            continue
        valid = (lag_times > 0) & (msd > 0) & np.isfinite(lag_times) & np.isfinite(msd)
        ax.loglog(lag_times[valid], msd[valid], linewidth=2, color=COLORS.get(fish_count, "#264653"),
                  label=fr"$N={fish_count}$ fish")
    ax.set_title(r"Mean square displacement by fish count")
    ax.set_xlabel(r"Lag time [$\mathrm{s}$]")
    ax.set_ylabel(r"Mean $\mathrm{MSD}$ [$\mathrm{mm^2}$]")
    style_axis(ax)
    ax.legend()
    fig.savefig(OUTPUT_FOLDER / "msd_by_fish_count.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_summary(groups):
    labels = []
    summary_values = {"Mean speed": [], "Mean $z$-position": [], "Mean pairwise distance": [],
                      "Mean nearest-neighbor distance": []}
    for fish_count, folder in groups.items():
        summary = load_group_array(folder, "group_summary_statistics.npy")
        if summary is None or summary.size < 5:
            continue
        labels.append(str(fish_count))
        summary_values["Mean speed"].append(summary[0])
        summary_values["Mean $z$-position"].append(summary[2])
        summary_values["Mean pairwise distance"].append(summary[3])
        summary_values["Mean nearest-neighbor distance"].append(summary[4])

    if not labels:
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    plot_data = [
        ("Mean speed", r"Mean speed [$\mathrm{mm\,s^{-1}}$]"),
        ("Mean $z$-position", r"Mean $z$-position [$\mathrm{mm}$]"),
        ("Mean pairwise distance", r"Mean distance [$\mathrm{mm}$]"),
        ("Mean nearest-neighbor distance", r"Distance [$\mathrm{mm}$]"),
    ]
    for ax, (key, ylabel) in zip(axes.flat, plot_data):
        ax.plot(x, summary_values[key], "o-", color="#007C91", linewidth=2, markersize=6)
        ax.set_title(key)
        ax.set_ylabel(ylabel)
        style_axis(ax)
    for ax in axes[1]:
        ax.set_xlabel("Number of fish, $N$")
        ax.set_xticks(x, labels)
    fig.suptitle("Mean run statistics by fish count", fontsize=15, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "mean_statistics_by_fish_count.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    groups = find_groups()
    if not groups:
        raise FileNotFoundError("No fish-group folders found in combined_analysis_output.")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    plot_autocorrelation(groups)
    plot_msd(groups)
    plot_scalar_summary(groups)
    print(f"Comparison plots saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()