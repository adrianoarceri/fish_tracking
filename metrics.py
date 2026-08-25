import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_velocities(trajectories, dt=1.0, frame_times=None):
  #Delta s= s(t+1)-s(t)
    displacements=np.diff(trajectories, axis=0)

    if frame_times is None:
        delta_times=np.full(trajectories.shape[0] - 1, dt)
    else:
        delta_times=np.diff(frame_times)

    velocities=displacements/delta_times[:, None, None]

#calcolo il modulo della velocità
    speeds=np.linalg.norm(velocities, axis=2)

#velocità media del pesce nel tempo
    mean_speed_per_fish=np.mean(speeds, axis=0)

    if frame_times is None:
        total_time=(trajectories.shape[0] - 1)*dt
    else:
        total_time=frame_times[-1]-frame_times[0]


#calcolo velocità primo-ultimo frame un po' inutile ma vabbe mettiamocelo

    first_to_last_displacement=trajectories[-1]-trajectories[0]
    first_to_last_distance=np.linalg.norm(first_to_last_displacement, axis=1)

#velocità media first-last frame
    total_displacement_speed = first_to_last_distance / total_time


#calcolo lunghezza reale del percorso
    path_lengths = np.sum(np.linalg.norm(displacements, axis=2), axis=0)

#velocità media considerando tutto il percorso
    path_average_speed = path_lengths / total_time

    return {
        "velocities": velocities,
        "speeds": speeds,
        "mean_speed_per_fish": mean_speed_per_fish,
        "total_displacement_speed": total_displacement_speed,
        "path_average_speed": path_average_speed,
        "path_lengths": path_lengths,
        "delta_times": delta_times,
        "total_time": total_time,
    }

def compute_pairwise_distances(trajectories):
    num_frames, n_fish, _ = trajectories.shape

    distance_matrices=np.zeros((num_frames, n_fish, n_fish))
    mean_pairwise_distance_per_frame=np.zeros(num_frames)

    for t in range(num_frames):
      #calcola la distanza tra tutti i pesci ad un dato frame t
        condensed =pdist(trajectories[t])
        matrix=squareform(condensed)

        distance_matrices[t]=matrix

        #Calcola la distanza media tra tutte le coppie di pesci in quel frame.
        mean_pairwise_distance_per_frame[t]=np.mean(condensed)

    #Calcola la distanza media totale tra pesci, mediando su tutti i frame.
    total_mean_pairwise_distance = np.mean(mean_pairwise_distance_per_frame)

    #Calcola la distanza media tra ogni coppia di pesci lungo tutto il tempo
    mean_distance_between_each_pair = np.mean(distance_matrices, axis=0)

#prende la matrice delle distanze al primo e all’ultimo frame.
    first_frame_distance_matrix = distance_matrices[0]
    last_frame_distance_matrix = distance_matrices[-1]

#cambio distanza primo-ultimo frame
    change_first_to_last = last_frame_distance_matrix - first_frame_distance_matrix

    return {
        "distance_matrices": distance_matrices,
        "mean_pairwise_distance_per_frame": mean_pairwise_distance_per_frame,
        "total_mean_pairwise_distance": total_mean_pairwise_distance,
        "mean_distance_between_each_pair": mean_distance_between_each_pair,
        "first_frame_distance_matrix": first_frame_distance_matrix,
        "last_frame_distance_matrix": last_frame_distance_matrix,
        "change_first_to_last": change_first_to_last,
    }

def compute_z_statistics(trajectories):
    z_positions=trajectories[:, :, 2]

#altezza media per ogni pesce
    mean_z_per_fish =np.mean(z_positions, axis=0)

#altezza media per ogni frame
    mean_z_per_frame= np.mean(z_positions, axis=1)

#valore medio dell'altezza per tutti i pesci e tutti i frame
    total_mean_z= np.mean(z_positions)

#variazione primo-ultimo frame per z
    first_to_last_z_change=z_positions[-1]-z_positions[0]

    return {
        "z_positions": z_positions,
        "mean_z_per_fish": mean_z_per_fish,
        "mean_z_per_frame": mean_z_per_frame,
        "total_mean_z": total_mean_z,
        "first_to_last_z_change": first_to_last_z_change,
    }


def compute_z_zone_times(trajectories, dt=1.0, z_limits=None):

    z=trajectories[:, :, 2]

    num_frames, n_fish=z.shape

    if z_limits is None:
        z_min=np.min(z)
        z_max= np.max(z)
    else:
        z_min, z_max= z_limits


#nota: per fare 4 zone servono 5 bordi
    edges=np.linspace(z_min, z_max, 5)
    frame_durations=np.full(num_frames, dt)

#Crea una matrice in cui salvare il tempo passato da ogni pesce in ogni zona.
#es: time_per_zone_per_fish(2,0)= è il tempo passato dal pesce 2 nella zona più bassa
    time_per_zone_per_fish=np.zeros((n_fish, 4))

#considero la traiettoria verticale di ogni pesce
    for fish_idx in range(n_fish):
        fish_z=z[:, fish_idx]

#assegno per ogni valore di z una fascia
        zone_indices =np.digitize(fish_z, edges[1:-1], right=False)

        for zone in range(4):
            #Conta quanti frame il pesce ha passato in ogni zona
            mask =zone_indices==zone
            time_per_zone_per_fish[fish_idx, zone]=np.sum(frame_durations[mask])

    total_time= np.sum(frame_durations)

    fraction_per_zone_per_fish=time_per_zone_per_fish/total_time

    mean_time_per_zone =np.mean(time_per_zone_per_fish, axis=0)

    mean_fraction_per_zone= np.mean(fraction_per_zone_per_fish, axis=0)

    return {
        "z_edges": edges,
        "time_per_zone_per_fish": time_per_zone_per_fish,
        "fraction_per_zone_per_fish": fraction_per_zone_per_fish,
        "mean_time_per_zone": mean_time_per_zone,
        "mean_fraction_per_zone": mean_fraction_per_zone,
        "frame_durations": frame_durations,
    }


def _probability_histogram(values, bins=30):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    if np.all(values == values[0]):
        width = max(abs(values[0]) * 0.05, 1e-12)
        edges = np.array([values[0] - width, values[0] + width])
    else:
        edges = np.histogram_bin_edges(values, bins=bins)
    density, edges = np.histogram(values, bins=edges, density=True)
    return density, edges


def compute_msd(trajectories, dt=1.0, min_fit_points=5, min_r2=0.9, fit_end_time=1.5):
    """Compute displacement from the initial position and fit the fixed initial interval."""
    trajectories = np.asarray(trajectories, dtype=float)
    displacement = trajectories - trajectories[0:1]
    msd_per_fish = np.sum(displacement ** 2, axis=2)
    msd_mean = np.mean(msd_per_fish, axis=1)
    lag_times = np.arange(trajectories.shape[0], dtype=float) * dt

    valid = (lag_times > 0) & (lag_times <= fit_end_time) & (msd_mean > 0) & np.isfinite(msd_mean)
    valid_indices = np.flatnonzero(valid)
    fit_start = fit_end = None
    slope = intercept = r_squared = np.nan
    min_points = max(3, min_fit_points)
    if valid_indices.size >= min_points:
        # t=0 cannot be used in log-log scale; all positive lags up to 1.5 s are fitted.
        indices = valid_indices
        log_times = np.log10(lag_times[indices])
        log_msd = np.log10(msd_mean[indices])
        coefficients = np.polyfit(log_times, log_msd, 1)
        predicted = np.polyval(coefficients, log_times)
        residual = np.sum((log_msd - predicted) ** 2)
        total = np.sum((log_msd - np.mean(log_msd)) ** 2)
        r_squared = 1.0 - residual / total if total > 0 else 1.0
        slope, intercept = coefficients
        fit_start, fit_end = int(indices[0]), int(indices[-1])

    return {
        "lag_times": lag_times,
        "msd_per_fish": msd_per_fish,
        "msd_mean": msd_mean,
        "fit_start_index": fit_start,
        "fit_end_index": fit_end,
        "fit_start_time": lag_times[fit_start] if fit_start is not None else np.nan,
        "fit_end_time": lag_times[fit_end] if fit_end is not None else np.nan,
        "fit_requested_end_time": fit_end_time,
        "fit_exponent": slope,
        "fit_intercept_log10": intercept,
        "fit_r_squared": r_squared,
    }


def compute_nearest_neighbor_distances(trajectories, histogram_bins=30):
    distance_matrices = compute_pairwise_distances(trajectories)["distance_matrices"]
    nearest_indices = np.argmin(np.where(np.eye(trajectories.shape[1], dtype=bool), np.inf, distance_matrices), axis=2)
    nearest_distances = np.take_along_axis(distance_matrices, nearest_indices[:, :, None], axis=2)[:, :, 0]
    density, edges = _probability_histogram(nearest_distances.ravel(), histogram_bins)
    return {
        "nearest_neighbor_indices": nearest_indices,
        "nearest_neighbor_distances": nearest_distances,
        "mean_nearest_neighbor_distance_per_frame": np.mean(nearest_distances, axis=1),
        "mean_nearest_neighbor_distance": np.mean(nearest_distances),
        "histogram_density": density,
        "histogram_edges": edges,
    }


def compute_step_length_distribution(trajectories, histogram_bins=30):
    step_lengths = np.linalg.norm(np.diff(trajectories, axis=0), axis=2)
    density, edges = _probability_histogram(step_lengths.ravel(), histogram_bins)
    return {
        "step_lengths": step_lengths,
        "pooled_step_lengths": step_lengths.ravel(),
        "histogram_density": density,
        "histogram_edges": edges,
    }


def compute_velocity_polarization(velocities):
    speeds = np.linalg.norm(velocities, axis=2)
    unit_velocities = np.divide(velocities, speeds[:, :, None], out=np.zeros_like(velocities), where=speeds[:, :, None] > 0)
    polarization = np.mean(unit_velocities, axis=1)
    return {"polarization_vectors": polarization, "polarization_magnitude": np.linalg.norm(polarization, axis=1)}


def compute_velocity_autocorrelation(velocities, dt=1.0):
    num_frames = velocities.shape[0]
    values = np.full(num_frames, np.nan)
    valid_pair_counts = np.zeros(num_frames, dtype=int)
    for lag in range(num_frames):
        products = np.sum(velocities[:num_frames - lag] * velocities[lag:], axis=2)
        values[lag] = np.mean(products)
        valid_pair_counts[lag] = products.size
    if np.isfinite(values[0]) and values[0] != 0:
        values /= values[0]
    return {"lag_times": np.arange(num_frames) * dt, "autocorrelation": values, "valid_pair_counts": valid_pair_counts}


def compute_neighbor_velocity_correlation(trajectories, velocities, histogram_bins=20):
    nearest = compute_nearest_neighbor_distances(trajectories)
    unit_velocities = np.divide(velocities, np.linalg.norm(velocities, axis=2)[:, :, None], out=np.zeros_like(velocities), where=np.linalg.norm(velocities, axis=2)[:, :, None] > 0)
    distances = nearest["nearest_neighbor_distances"][:-1]
    correlations = np.sum(unit_velocities * unit_velocities[np.arange(velocities.shape[0])[:, None], nearest["nearest_neighbor_indices"][:-1]], axis=2)
    density_edges = np.histogram_bin_edges(distances.ravel(), bins=histogram_bins)
    bin_indices = np.digitize(distances.ravel(), density_edges) - 1
    mean_correlation = np.full(len(density_edges) - 1, np.nan)
    counts = np.zeros(len(density_edges) - 1, dtype=int)
    flat_correlations = correlations.ravel()
    for bin_index in range(len(mean_correlation)):
        mask = bin_indices == bin_index
        counts[bin_index] = np.count_nonzero(mask)
        if counts[bin_index]:
            mean_correlation[bin_index] = np.mean(flat_correlations[mask])
    return {
        "distances": distances,
        "correlations": correlations,
        "bin_edges": density_edges,
        "bin_centers": (density_edges[:-1] + density_edges[1:]) / 2,
        "mean_correlation": mean_correlation,
        "valid_pair_counts": counts,
    }