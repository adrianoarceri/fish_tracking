import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tracking import build_trajectories
from metrics import (
    compute_velocities, compute_pairwise_distances, compute_z_statistics,
    compute_z_zone_times, compute_msd, compute_nearest_neighbor_distances,
    compute_step_length_distribution, compute_velocity_polarization,
    compute_velocity_autocorrelation, compute_neighbor_velocity_correlation,
)
from plotting import plot_collective_trajectories, plot_single_fish_trajectory, make_trajectory_video

def analyze_fish_trajectories(
    recon_folder,
    n_fish,
    output_folder="output_fish_trajectories",
    grid_step=0.75,
    dt=1.0,
    tracking_mode="velocity",
    make_video=True,
    search_range=20.0,
    memory=5,
    velocity_smoothing=0.5,
    min_component_points=10,
):
    recon_path = Path(recon_folder)
    session_name = recon_path.parent.name
    if session_name.endswith("_ANALISI"):
        session_name = f"{session_name[:-len('_ANALISI')]}_output"
    else:
        session_name = f"{session_name}_output"

    output_folder = Path(output_folder) / session_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # costruzione delle traiettorie dalla cartella recon
    trajectories, frame_files, tracking_diagnostics = build_trajectories(
        recon_folder=recon_folder,
        n_fish=n_fish,
        grid_step=grid_step,
        tracking_mode=tracking_mode,
        save_folder=output_folder,
        dt=dt,
        max_displacement=search_range,
        max_gap_frames=memory,
        velocity_smoothing=velocity_smoothing,
        min_component_points=min_component_points,
        return_diagnostics=True,
    )

    frame_times = np.arange(trajectories.shape[0]) * dt

    velocity_results = compute_velocities(
        trajectories,
        dt=dt,
        frame_times=None
    )

    distance_results = compute_pairwise_distances(trajectories)
    msd_results = compute_msd(trajectories, dt=dt, fit_end_time=2.4)
    nearest_results = compute_nearest_neighbor_distances(trajectories)
    step_results = compute_step_length_distribution(trajectories)
    polarization_results = compute_velocity_polarization(velocity_results["velocities"])
    autocorrelation_results = compute_velocity_autocorrelation(velocity_results["velocities"], dt=dt)
    neighbor_correlation_results = compute_neighbor_velocity_correlation(
        trajectories, velocity_results["velocities"]
    )
    z_results = compute_z_statistics(trajectories)
    z_zone_results = compute_z_zone_times(trajectories, dt=dt)

    # salvataggio degli array numpy
    np.save(output_folder / "trajectories.npy", trajectories)
    
    np.save(output_folder / "speeds_frame_by_frame.npy", velocity_results["speeds"])
    np.save(output_folder / "velocities_frame_by_frame.npy", velocity_results["velocities"])
    np.save(output_folder / "mean_speed_per_fish.npy", velocity_results["mean_speed_per_fish"])
    np.save(output_folder / "total_displacement_speed.npy", velocity_results["total_displacement_speed"])
    np.save(output_folder / "path_average_speed.npy", velocity_results["path_average_speed"])
    np.save(output_folder / "path_lengths.npy", velocity_results["path_lengths"])

    np.save(output_folder / "distance_matrices.npy", distance_results["distance_matrices"])
    np.save(output_folder / "mean_pairwise_distance_per_frame.npy", distance_results["mean_pairwise_distance_per_frame"])
    np.save(output_folder / "mean_distance_between_each_pair.npy", distance_results["mean_distance_between_each_pair"])
    np.save(output_folder / "change_first_to_last_distance_matrix.npy", distance_results["change_first_to_last"])

    np.save(output_folder / "msd_per_fish.npy", msd_results["msd_per_fish"])
    np.save(output_folder / "mean_square_displacement.npy", msd_results["msd_mean"])
    np.save(output_folder / "msd_lag_times.npy", msd_results["lag_times"])
    np.save(output_folder / "msd_loglog_fit_coefficients.npy", np.array([msd_results["fit_exponent"], msd_results["fit_intercept_log10"]]))
    np.save(output_folder / "nearest_neighbor_indices.npy", nearest_results["nearest_neighbor_indices"])
    np.save(output_folder / "nearest_neighbor_distances.npy", nearest_results["nearest_neighbor_distances"])
    np.save(output_folder / "mean_nearest_neighbor_distance_per_frame.npy", nearest_results["mean_nearest_neighbor_distance_per_frame"])
    np.save(output_folder / "nearest_neighbor_distance_histogram_density.npy", nearest_results["histogram_density"])
    np.save(output_folder / "nearest_neighbor_distance_histogram_edges.npy", nearest_results["histogram_edges"])
    np.save(output_folder / "step_lengths.npy", step_results["step_lengths"])
    np.save(output_folder / "step_length_histogram_density.npy", step_results["histogram_density"])
    np.save(output_folder / "step_length_histogram_edges.npy", step_results["histogram_edges"])
    np.save(output_folder / "velocity_polarization_vectors.npy", polarization_results["polarization_vectors"])
    np.save(output_folder / "velocity_polarization_magnitude.npy", polarization_results["polarization_magnitude"])
    np.save(output_folder / "velocity_autocorrelation.npy", autocorrelation_results["autocorrelation"])
    np.save(output_folder / "velocity_autocorrelation_lag_times.npy", autocorrelation_results["lag_times"])
    np.save(output_folder / "velocity_autocorrelation_valid_pair_counts.npy", autocorrelation_results["valid_pair_counts"])
    np.save(output_folder / "neighbor_velocity_correlation_bin_centers.npy", neighbor_correlation_results["bin_centers"])
    np.save(output_folder / "neighbor_velocity_correlation.npy", neighbor_correlation_results["mean_correlation"])
    np.save(output_folder / "neighbor_velocity_correlation_valid_pair_counts.npy", neighbor_correlation_results["valid_pair_counts"])

    np.save(output_folder / "z_positions.npy", z_results["z_positions"])
    np.save(output_folder / "mean_z_per_fish.npy", z_results["mean_z_per_fish"])
    np.save(output_folder / "mean_z_per_frame.npy", z_results["mean_z_per_frame"])
    np.save(output_folder / "first_to_last_z_change.npy", z_results["first_to_last_z_change"])

    np.save(output_folder / "z_zone_time_per_fish.npy", z_zone_results["time_per_zone_per_fish"])
    np.save(output_folder / "z_zone_fraction_per_fish.npy", z_zone_results["fraction_per_zone_per_fish"])
    np.save(output_folder / "z_zone_edges.npy", z_zone_results["z_edges"])

    np.save(output_folder / "frame_times.npy", frame_times)
    np.save(output_folder / "frame_files.npy", np.array(frame_files))

    # esportazione dei grafici 3d spaziali
    collective_path = output_folder / "collective_trajectories.pdf"
    plot_collective_trajectories(trajectories, save_path=collective_path)

    single_paths = []
    for fish_idx in range(n_fish):
        single_path = output_folder / f"fish_{fish_idx:02d}_trajectory.pdf"
        plot_single_fish_trajectory(trajectories, fish_idx=fish_idx, save_path=single_path)
        single_paths.append(single_path)

    # esportazione dei grafici temporali
    fig, ax = plt.subplots(figsize=(9, 5))
    time_speed = frame_times[:-1]
    for fish_idx in range(n_fish):
        ax.plot(time_speed, velocity_results["speeds"][:, fish_idx], label=f"Fish {fish_idx}")
    ax.set_title("Speed per frame")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("speed [mm/s]")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", lw=0.5)
    speed_plot_path = output_folder / "speeds_frame_by_frame.pdf"
    fig.savefig(speed_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    positive = msd_results["lag_times"] > 0
    ax.loglog(msd_results["lag_times"][positive], msd_results["msd_mean"][positive], label="average MSD")
    if msd_results["fit_start_index"] is not None:
        fit_slice = slice(msd_results["fit_start_index"], msd_results["fit_end_index"] + 1)
        fit_values = 10 ** (msd_results["fit_intercept_log10"] + msd_results["fit_exponent"] * np.log10(msd_results["lag_times"][fit_slice]))
        ax.loglog(msd_results["lag_times"][fit_slice], fit_values, "--", label=f"power law, $\\alpha$={msd_results['fit_exponent']:.2f}")
    ax.set_title("Mean Square Displacement (MSD)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("MSD [$\\mathrm{mm}^2$]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()
    fig.savefig(output_folder / "mean_square_displacement.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame_times, nearest_results["mean_nearest_neighbor_distance_per_frame"])
    ax.set_title("Mean distance to the nearest neighbor")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [mm]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    fig.savefig(output_folder / "mean_nearest_neighbor_distance.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if nearest_results["histogram_density"].size:
        ax.stairs(nearest_results["histogram_density"], nearest_results["histogram_edges"])
    ax.set_title("Nearest neighbor distance distribution")
    ax.set_xlabel("distance [mm]")
    ax.set_ylabel("probability density")
    ax.grid(True, which="both", ls="--", lw=0.5)
    fig.savefig(output_folder / "nearest_neighbor_distance_histogram.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if step_results["histogram_density"].size:
        ax.stairs(step_results["histogram_density"], step_results["histogram_edges"])
    ax.set_title("Step length distribution")
    ax.set_xlabel("step [mm]")
    ax.set_ylabel("probability density")
    ax.grid(True, which="both", ls="--", lw=0.5)
    fig.savefig(output_folder / "step_length_histogram.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame_times[:-1], polarization_results["polarization_magnitude"], color="black", linewidth=2, label="$|P(t)|$")
    ax.set_title("Velocity polarization intensity")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("$|P(t)|$")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()
    fig.savefig(output_folder / "velocity_polarization.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(autocorrelation_results["lag_times"], autocorrelation_results["autocorrelation"])
    ax.set_title("Temporal autocorrelation of velocities")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_xlabel("lag time $\\tau$ [s]")
    ax.set_ylabel("$\\frac{C_v(\\tau)}{C_v(0)}$")
    fig.savefig(output_folder / "velocity_autocorrelation.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    semilog_mask = (
        (autocorrelation_results["lag_times"] >= 0)
        & (autocorrelation_results["autocorrelation"] > 0)
        & np.isfinite(autocorrelation_results["autocorrelation"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        autocorrelation_results["lag_times"][semilog_mask],
        autocorrelation_results["autocorrelation"][semilog_mask],
        "o-",
    )
    ax.set_title("Temporal autocorrelation of velocities in semilog scale")
    ax.set_xlabel("lag time $\\tau$ [s]")
    ax.set_ylabel("$\\frac{C_v(\\tau)}{C_v(0)}$")
    ax.grid(True, which="both", ls="--", lw=0.5)
    fig.savefig(output_folder / "velocity_autocorrelation_semilog.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(neighbor_correlation_results["bin_centers"], neighbor_correlation_results["mean_correlation"], "o-")
    ax.set_title("Temporal correlation of velocities between nearest neighbors")
    ax.set_xlabel("distance [mm]")
    ax.set_ylabel("directional correlation")
    ax.grid(True, which="both", ls="--", lw=0.5)
    fig.savefig(output_folder / "neighbor_velocity_correlation.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame_times, distance_results["mean_pairwise_distance_per_frame"], linewidth=2)
    ax.set_title("Mean pairwise distance between fish frame by frame")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean distance [mm]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    dist_plot_path = output_folder / "mean_pairwise_distance_per_frame.pdf"
    fig.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame_times, z_results["mean_z_per_frame"], linewidth=2)
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.set_title("Mean height of the group")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean z [mm]")
    z_plot_path = output_folder / "mean_z_per_frame.pdf"
    fig.savefig(z_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    zone_names = ["low", "low-medium", "medium-high", "high"]
    ax.bar(zone_names, z_zone_results["mean_fraction_per_zone"] * 100)
    ax.set_title("Mean percentage of time in z zones")
    ax.set_ylabel("time [%]")
    ax.grid(True, which="both", ls="--", lw=0.5)
    z_zone_plot_path = output_folder / "z_zone_time_percentages.pdf"
    fig.savefig(z_zone_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # generazione dell'animazione
    video_path = None
    if make_video:
        video_path = output_folder / "fish_trajectories.mp4"
        try:
            make_trajectory_video(trajectories, save_path=str(video_path), fps=15)
        except Exception as e:
            print("Non sono riuscito a salvare in mp4, provo con gif.")
            print("Errore:", e)
            video_path = output_folder / "fish_trajectories.gif"
            make_trajectory_video(trajectories, save_path=str(video_path), fps=15)

    # riepilogo su terminale
    print("\n================ RISULTATI GENERALI ================\n")
    print(f"Cartella analizzata: {recon_folder}")
    print(f"Numero frame: {trajectories.shape[0]}")
    print(f"Numero pesci: {trajectories.shape[1]}")
    print(f"Tempo totale: {(trajectories.shape[0] - 1) * dt:.3f} s")
    
    print("\n--- Velocità media lungo il percorso per pesce ---")
    individual_mean_speeds = velocity_results["path_average_speed"]
    for fish_idx, value in enumerate(individual_mean_speeds):
        print(f"Fish {fish_idx}: {value:.3f} mm/s")
    individual_speed_mean = np.nanmean(individual_mean_speeds)
    individual_speed_std = np.nanstd(individual_mean_speeds, ddof=1)
    collective_frame_speeds = np.nanmean(velocity_results["speeds"], axis=1)
    collective_speed_variance = np.nanvar(collective_frame_speeds, ddof=1)
    print(f"Velocità media del gruppo: {individual_speed_mean:.3f} mm/s")
    print(f"Deviazione standard individuale: {individual_speed_std:.3f} mm/s")
    print(f"Varianza della velocità collettiva: {collective_speed_variance:.3f} (mm/s)^2")

    print("\n--- Fit lineare del MSD in scala log-log ---")
    if msd_results["fit_start_index"] is not None:
        fit_coefficient = 10 ** msd_results["fit_intercept_log10"]
        fit_speed_scale = np.sqrt(fit_coefficient)
        print(
            "MSD = "
            f"{fit_coefficient:.6f} * t^{msd_results['fit_exponent']:.6f}"
        )
        print(f"Esponente alpha: {msd_results['fit_exponent']:.6f}")
        print(f"Coefficiente A = 10^intercetta: {fit_coefficient:.6f}")
        print(f"Radice quadrata di A: {fit_speed_scale:.6f} mm/s")
        print(f"R²: {msd_results['fit_r_squared']:.6f}")
        print(
            "Intervallo fit: "
            f"{msd_results['fit_start_time']:.3f}-"
            f"{msd_results['fit_end_time']:.3f} s"
        )
    else:
        print("Fit non disponibile: dati validi insufficienti.")

    print("\n--- Distanza media tra pesci ---")
    print(f"Distanza media totale: {distance_results['total_mean_pairwise_distance']:.3f} mm")

    print("\n--- Altezza z media globale ---")
    print(f"{z_results['total_mean_z']:.3f} mm")

    print(f"\nTutti i file sono stati salvati in: {output_folder}")

    return {
        "trajectories": trajectories,
        "tracking_diagnostics": tracking_diagnostics,
        "frame_files": frame_files,
        "frame_times": frame_times,
        "velocity_results": velocity_results,
        "distance_results": distance_results,
        "msd_results": msd_results,
        "nearest_results": nearest_results,
        "step_results": step_results,
        "polarization_results": polarization_results,
        "autocorrelation_results": autocorrelation_results,
        "neighbor_correlation_results": neighbor_correlation_results,
        "z_results": z_results,
        "z_zone_results": z_zone_results,
        "output_folder": output_folder,
        "video_path": video_path,
    }
