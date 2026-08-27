from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Select one fish-count group at a time.
INPUT_ROOT = Path("8_fish_analysis_output")
OUTPUT_FOLDER = Path("combined_analysis_output")
MSD_FIT_END_TIME = 1.5


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

COLORS = ["#007C91", "#E76F51", "#2A9D8F", "#E9C46A", "#264653"]


def style_axis(ax):
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.55, color="#CBD5E1")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def find_run_folders():
    if not INPUT_ROOT.exists():
        print(f"Skipping missing folder: {INPUT_ROOT}")
        return []

    run_folders = []
    for frame_times_path in sorted(INPUT_ROOT.rglob("frame_times.npy")):
        run_folder = frame_times_path.parent
        if run_folder not in run_folders:
            run_folders.append(run_folder)
    return run_folders


def load_array(run_folder, filename):
    path = run_folder / filename
    if not path.exists():
        return None
    return np.asarray(np.load(path, allow_pickle=True), dtype=float)


def padded_mean(arrays):
    """Mean arrays along time, allowing runs with different durations."""
    arrays = [np.asarray(array, dtype=float).reshape(-1) for array in arrays if array is not None]
    if not arrays:
        return np.array([])
    length = max(array.size for array in arrays)
    padded = np.full((len(arrays), length), np.nan)
    for index, array in enumerate(arrays):
        padded[index, :array.size] = array
    with np.errstate(invalid="ignore"):
        return np.nanmean(padded, axis=0)


def concatenate_finite(arrays):
    values = [np.asarray(array, dtype=float).ravel() for array in arrays if array is not None]
    if not values:
        return np.array([])
    combined = np.concatenate(values)
    return combined[np.isfinite(combined)]


def plot_overlaid_autocorrelations(runs):
    fig, ax = plt.subplots(figsize=(9, 5))
    autocorrelations = []
    for index, run in enumerate(runs):
        lag_times = load_array(run, "velocity_autocorrelation_lag_times.npy")
        values = load_array(run, "velocity_autocorrelation.npy")
        if lag_times is None or values is None:
            continue
        ax.plot(lag_times, values, color=COLORS[index % len(COLORS)], linewidth=1.2, alpha=0.55,
                label=run.name)
        autocorrelations.append(values)
    ax.set_title(r"Velocity temporal autocorrelation across runs")
    ax.set_xlabel(r"Lag time [$\mathrm{s}$]")
    ax.set_ylabel(r"$C_v(\tau) / C_v(0)$")
    style_axis(ax)
    if autocorrelations:
        ax.legend(fontsize=8, ncol=2)
    fig.savefig(OUTPUT_FOLDER / "velocity_autocorrelation_overlaid.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mean_autocorrelation(runs):
    lag_arrays = []
    value_arrays = []
    for run in runs:
        lag_times = load_array(run, "velocity_autocorrelation_lag_times.npy")
        values = load_array(run, "velocity_autocorrelation.npy")
        if lag_times is not None and values is not None:
            lag_arrays.append(lag_times)
            value_arrays.append(values)

    if not value_arrays:
        return

    max_length = max(values.size for values in value_arrays)
    padded_values = np.full((len(value_arrays), max_length), np.nan)
    padded_lags = np.full((len(lag_arrays), max_length), np.nan)
    for index, (lags, values) in enumerate(zip(lag_arrays, value_arrays)):
        length = min(lags.size, values.size)
        padded_lags[index, :length] = lags[:length]
        padded_values[index, :length] = values[:length]

    mean_lags = np.nanmean(padded_lags, axis=0)
    mean_values = np.nanmean(padded_values, axis=0)
    valid = np.isfinite(mean_lags) & np.isfinite(mean_values)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(mean_lags[valid], mean_values[valid], color=COLORS[0], linewidth=2.4)
    ax.set_title(r"Mean velocity temporal autocorrelation")
    ax.set_xlabel(r"Lag time [$\mathrm{s}$]")
    ax.set_ylabel(r"Mean $C_v(\tau) / C_v(0)$")
    style_axis(ax)
    fig.savefig(OUTPUT_FOLDER / "velocity_autocorrelation_mean.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    np.save(OUTPUT_FOLDER / "velocity_autocorrelation_mean.npy", np.vstack((mean_lags, mean_values)))


def plot_mean_msd_and_fit(runs):
    lag_arrays = []
    msd_arrays = []
    for run in runs:
        lag_times = load_array(run, "msd_lag_times.npy")
        values = load_array(run, "mean_square_displacement.npy")
        if lag_times is not None and values is not None:
            lag_arrays.append(lag_times)
            msd_arrays.append(values)

    if not msd_arrays:
        return

    max_length = max(values.size for values in msd_arrays)
    padded_lags = np.full((len(lag_arrays), max_length), np.nan)
    padded_msd = np.full((len(msd_arrays), max_length), np.nan)
    for index, (lags, values) in enumerate(zip(lag_arrays, msd_arrays)):
        length = min(lags.size, values.size)
        padded_lags[index, :length] = lags[:length]
        padded_msd[index, :length] = values[:length]

    mean_lags = np.nanmean(padded_lags, axis=0)
    mean_msd = np.nanmean(padded_msd, axis=0)
    valid = np.isfinite(mean_lags) & np.isfinite(mean_msd)

    fit_valid = valid & (mean_lags > 0) & (mean_lags <= MSD_FIT_END_TIME) & (mean_msd > 0)
    if np.count_nonzero(fit_valid) >= 3:
        coefficients = np.polyfit(np.log10(mean_lags[fit_valid]), np.log10(mean_msd[fit_valid]), 1)
        alpha, intercept = coefficients
        fitted_values = 10 ** (intercept + alpha * np.log10(mean_lags[fit_valid]))
        residual = np.sum((np.log10(mean_msd[fit_valid]) - np.log10(fitted_values)) ** 2)
        total = np.sum((np.log10(mean_msd[fit_valid]) - np.mean(np.log10(mean_msd[fit_valid]))) ** 2)
        r_squared = 1 - residual / total if total > 0 else 1.0
        coefficient_a = 10 ** intercept
    else:
        alpha = coefficient_a = r_squared = np.nan
        fitted_values = np.array([])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(mean_lags[valid], mean_msd[valid], color=COLORS[0], linewidth=2.4, label=r"Mean $\mathrm{MSD}$")
    if fitted_values.size:
        ax.loglog(mean_lags[fit_valid], fitted_values, "--", color=COLORS[1], linewidth=1.8,
                  label=fr"Fit ($A={coefficient_a:.3g}$, $\alpha={alpha:.3f}$, $R^2={r_squared:.3f}$)")
    ax.set_title(fr"Mean square displacement, fit up to {MSD_FIT_END_TIME:g} s")
    ax.set_xlabel(r"Lag time [$\mathrm{s}$]")
    ax.set_ylabel(r"Mean $\mathrm{MSD}$ [$\mathrm{mm^2}$]")
    style_axis(ax)
    ax.legend()
    fig.savefig(OUTPUT_FOLDER / "mean_msd_with_fit.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    np.save(OUTPUT_FOLDER / "mean_msd_lag_times.npy", mean_lags)
    np.save(OUTPUT_FOLDER / "mean_msd.npy", mean_msd)
    sqrt_a = np.sqrt(coefficient_a) if np.isfinite(coefficient_a) and coefficient_a >= 0 else np.nan
    np.save(
        OUTPUT_FOLDER / "mean_msd_fit_coefficients.npy",
        np.array([alpha, np.log10(coefficient_a), coefficient_a, sqrt_a, r_squared]),
    )
    print(
        f"MSD fit (t <= {MSD_FIT_END_TIME:g} s): "
        f"A = {coefficient_a:.6g}, sqrt(A) = {sqrt_a:.6g} mm/s, "
        f"alpha = {alpha:.6g}, R^2 = {r_squared:.6g}"
    )


def print_group_speed_statistics(runs):
    group_speed_per_frame = []
    for run in runs:
        speeds = load_array(run, "speeds_frame_by_frame.npy")
        if speeds is None or speeds.ndim != 2:
            continue
        with np.errstate(invalid="ignore"):
            group_speed_per_frame.append(np.nanmean(speeds, axis=1))

    group_speeds = concatenate_finite(group_speed_per_frame)
    if group_speeds.size == 0:
        print("Group speed statistics: no valid speed observations found.")
        return

    group_mean = np.mean(group_speeds)
    group_std = np.std(group_speeds, ddof=1) if group_speeds.size > 1 else 0.0
    print("\nGroup speed statistics over all valid frames and runs:")
    print(f"Mean group speed = {group_mean:.6g} mm/s")
    print(f"Standard deviation = {group_std:.6g} mm/s")
    np.save(OUTPUT_FOLDER / "group_speed_statistics.npy", np.array([group_mean, group_std]))


def save_group_summary_statistics(runs):
    summary = []
    for filename in (
        "mean_z_per_frame.npy",
        "mean_pairwise_distance_per_frame.npy",
        "mean_nearest_neighbor_distance_per_frame.npy",
    ):
        values = concatenate_finite([load_array(run, filename) for run in runs])
        summary.append(np.mean(values) if values.size else np.nan)

    speed_statistics = np.load(OUTPUT_FOLDER / "group_speed_statistics.npy")
    summary = np.array([speed_statistics[0], speed_statistics[1], *summary])
    np.save(OUTPUT_FOLDER / "group_summary_statistics.npy", summary)


def plot_pooled_histogram(runs, filename, title, xlabel, output_name, color):
    values = concatenate_finite([load_array(run, filename) for run in runs])
    if values.size == 0:
        return
    density, edges = np.histogram(values, bins="auto", density=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stairs(density, edges, fill=True, color=color, alpha=0.72, linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    style_axis(ax)
    fig.savefig(OUTPUT_FOLDER / output_name, dpi=250, bbox_inches="tight")
    plt.close(fig)
    np.save(OUTPUT_FOLDER / f"{Path(output_name).stem}_density.npy", density)
    np.save(OUTPUT_FOLDER / f"{Path(output_name).stem}_edges.npy", edges)


def plot_per_run_time_series(runs, filename, title, ylabel, output_name, color_offset=0):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = False
    for index, run in enumerate(runs):
        if filename == "mean_speed_per_frame.npy":
            speeds = load_array(run, "speeds_frame_by_frame.npy")
            values = np.nanmean(speeds, axis=1) if speeds is not None else None
        else:
            values = load_array(run, filename)
        frame_times = load_array(run, "frame_times.npy")
        if values is None or frame_times is None:
            continue
        values = values.reshape(-1)
        length = min(values.size, frame_times.size)
        ax.plot(frame_times[:length], values[:length], color=COLORS[(index + color_offset) % len(COLORS)],
                linewidth=1.25, alpha=0.65, label=run.name)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel(r"Time [$\mathrm{s}$]")
    ax.set_ylabel(ylabel)
    style_axis(ax)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(OUTPUT_FOLDER / output_name, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    global OUTPUT_FOLDER

    runs = find_run_folders()
    if not runs:
        raise FileNotFoundError("No output folders containing frame_times.npy were found.")

    fish_group = INPUT_ROOT.name.removesuffix("_analysis_output")
    OUTPUT_FOLDER = Path("combined_analysis_output") / fish_group
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(runs)} runs for {fish_group}.")
    plot_overlaid_autocorrelations(runs)
    plot_mean_autocorrelation(runs)
    plot_mean_msd_and_fit(runs)
    print_group_speed_statistics(runs)
    save_group_summary_statistics(runs)
    plot_pooled_histogram(
        runs,
        "step_lengths.npy",
        "Pooled step-length distribution",
        r"Step length [$\mathrm{mm}$]",
        "step_length_distribution_pooled.png",
        COLORS[3],
    )
    plot_pooled_histogram(
        runs,
        "nearest_neighbor_distances.npy",
        "Pooled nearest-neighbor distance distribution",
        r"Nearest-neighbor distance [$\mathrm{mm}$]",
        "nearest_neighbor_distance_distribution_pooled.png",
        COLORS[2],
    )
    plot_per_run_time_series(
        runs,
        "mean_speed_per_frame.npy",
        "Mean speed per run",
        r"Mean speed [$\mathrm{mm\,s^{-1}}$]",
        "mean_speed_per_run.png",
    )
    plot_per_run_time_series(
        runs,
        "mean_z_per_frame.npy",
        "Mean z-position per run",
        r"Mean $z$-position [$\mathrm{mm}$]",
        "mean_z_per_run.png",
        1,
    )
    plot_per_run_time_series(
        runs,
        "mean_pairwise_distance_per_frame.npy",
        "Mean pairwise distance per run",
        r"Mean distance [$\mathrm{mm}$]",
        "mean_pairwise_distance_per_run.png",
        2,
    )
    plot_per_run_time_series(
        runs,
        "mean_nearest_neighbor_distance_per_frame.npy",
        "Mean nearest-neighbor distance per run",
        r"Nearest-neighbor distance [$\mathrm{mm}$]",
        "mean_nearest_neighbor_distance_per_run.png",
        3,
    )
    print(f"Combined plots saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
