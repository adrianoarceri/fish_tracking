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
    tracking_mode="nearest",
    make_video=True
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # costruzione delle traiettorie dalla cartella recon
    trajectories, frame_files = build_trajectories(
        recon_folder=recon_folder,
        n_fish=n_fish,
        grid_step=grid_step,
        tracking_mode=tracking_mode,
        save_folder=output_folder
    )

    frame_times = np.arange(trajectories.shape[0]) * dt

    velocity_results = compute_velocities(
        trajectories,
        dt=dt,
        frame_times=None
    )

    distance_results = compute_pairwise_distances(trajectories)
    msd_results = compute_msd(trajectories, dt=dt, fit_end_time=1.0)
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
    collective_path = output_folder / "collective_trajectories.png"
    plot_collective_trajectories(trajectories, save_path=collective_path)

    single_paths = []
    for fish_idx in range(n_fish):
        single_path = output_folder / f"fish_{fish_idx:02d}_trajectory.png"
        plot_single_fish_trajectory(trajectories, fish_idx=fish_idx, save_path=single_path)
        single_paths.append(single_path)

    # esportazione dei grafici temporali
    fig, ax = plt.subplots(figsize=(9, 5))
    time_speed = frame_times[:-1]
    for fish_idx in range(n_fish):
        ax.plot(time_speed, velocity_results["speeds"][:, fish_idx], label=f"Fish {fish_idx}")
    ax.set_title("velocità frame per frame")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("velocità [mm/s]")
    ax.legend(fontsize=8)
    speed_plot_path = output_folder / "speeds_frame_by_frame.png"
    fig.savefig(speed_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    positive = msd_results["lag_times"] > 0
    ax.loglog(msd_results["lag_times"][positive], msd_results["msd_mean"][positive], label="MSD medio")
    if msd_results["fit_start_index"] is not None:
        fit_slice = slice(msd_results["fit_start_index"], msd_results["fit_end_index"] + 1)
        fit_values = 10 ** (msd_results["fit_intercept_log10"] + msd_results["fit_exponent"] * np.log10(msd_results["lag_times"][fit_slice]))
        ax.loglog(msd_results["lag_times"][fit_slice], fit_values, "--", label=f"fit, alpha={msd_results['fit_exponent']:.2f}")
    ax.set_title("MSD rispetto alla posizione iniziale")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("MSD [mm²]")
    ax.legend()
    fig.savefig(output_folder / "mean_square_displacement.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame_times, nearest_results["mean_nearest_neighbor_distance_per_frame"])
    ax.set_title("Distanza media dal primo vicino")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("distanza [mm]")
    fig.savefig(output_folder / "mean_nearest_neighbor_distance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if nearest_results["histogram_density"].size:
        ax.stairs(nearest_results["histogram_density"], nearest_results["histogram_edges"])
    ax.set_title("Distribuzione della distanza dal primo vicino")
    ax.set_xlabel("distanza [mm]")
    ax.set_ylabel("densità di probabilità")
    fig.savefig(output_folder / "nearest_neighbor_distance_histogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if step_results["histogram_density"].size:
        ax.stairs(step_results["histogram_density"], step_results["histogram_edges"])
    ax.set_title("Distribuzione dei passi spaziali")
    ax.set_xlabel("passo [mm]")
    ax.set_ylabel("densità di probabilità")
    fig.savefig(output_folder / "step_length_histogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame_times[:-1], polarization_results["polarization_magnitude"], color="black", linewidth=2, label="|P(t)|")
    ax.set_title("Modulo della polarizzazione delle velocità")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("|P(t)|")
    ax.legend()
    fig.savefig(output_folder / "velocity_polarization.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(autocorrelation_results["lag_times"], autocorrelation_results["autocorrelation"])
    ax.set_title("Autocorrelazione temporale delle velocità")
    ax.set_xlabel("ritardo [s]")
    ax.set_ylabel("Cv(tau) / Cv(0)")
    fig.savefig(output_folder / "velocity_autocorrelation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(neighbor_correlation_results["bin_centers"], neighbor_correlation_results["mean_correlation"], "o-")
    ax.set_title("Correlazione delle velocità tra primi vicini")
    ax.set_xlabel("distanza [mm]")
    ax.set_ylabel("correlazione direzionale")
    fig.savefig(output_folder / "neighbor_velocity_correlation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame_times, distance_results["mean_pairwise_distance_per_frame"], linewidth=2)
    ax.set_title("distanza media tra pesci frame per frame")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("distanza media [mm]")
    dist_plot_path = output_folder / "mean_pairwise_distance_per_frame.png"
    fig.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame_times, z_results["mean_z_per_frame"], linewidth=2)
    ax.set_title("altezza media del gruppo")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("z media [mm]")
    z_plot_path = output_folder / "mean_z_per_frame.png"
    fig.savefig(z_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    zone_names = ["basso", "basso-medio", "alto-medio", "alto"]
    ax.bar(zone_names, z_zone_results["mean_fraction_per_zone"] * 100)
    ax.set_title("percentuale media di tempo nelle fasce z")
    ax.set_ylabel("tempo [%]")
    z_zone_plot_path = output_folder / "z_zone_time_percentages.png"
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
    for fish_idx, value in enumerate(velocity_results["path_average_speed"]):
        print(f"Fish {fish_idx}: {value:.3f} mm/s")

    print("\n--- Distanza media tra pesci ---")
    print(f"Distanza media totale: {distance_results['total_mean_pairwise_distance']:.3f} mm")

    print("\n--- Altezza z media globale ---")
    print(f"{z_results['total_mean_z']:.3f} mm")

    print(f"\nTutti i file sono stati salvati in: {output_folder}")

    return {
        "trajectories": trajectories,
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