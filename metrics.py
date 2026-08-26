import numpy as np


# ============================================================
# GENERAL HELPERS
# ============================================================

def _finite_rows(array):
    """
    Return a boolean mask identifying rows containing only
    finite values.

    For an array of shape (..., 3), this checks whether all
    three coordinates are finite.
    """
    array = np.asarray(array, dtype=float)
    return np.isfinite(array).all(axis=-1)


def _probability_histogram(values, bins=30):
    """
    Compute a probability-density histogram after removing
    NaN and Inf values.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.array([]), np.array([])

    if np.all(values == values[0]):
        width = max(abs(values[0]) * 0.05, 1e-12)
        edges = np.array([
            values[0] - width,
            values[0] + width
        ])
    else:
        edges = np.histogram_bin_edges(
            values,
            bins=bins
        )

    density, edges = np.histogram(
        values,
        bins=edges,
        density=True
    )

    return density, edges


# ============================================================
# VELOCITIES
# ============================================================

def compute_velocities(
    trajectories,
    dt=1.0,
    frame_times=None
):
    """
    Compute frame-to-frame velocities and speeds.

    A velocity is calculated only when both positions are valid.
    If either position is missing, the corresponding velocity
    and speed are NaN.

    Parameters
    ----------
    trajectories : np.ndarray
        Shape (num_frames, n_fish, 3).

    dt : float
        Time between frames.

    frame_times : np.ndarray, optional
        Explicit frame times.

    Returns
    -------
    dict
        Velocity and speed statistics.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    num_frames = trajectories.shape[0]

    if num_frames < 2:
        raise ValueError(
            "At least two frames are required to compute velocities."
        )

    # --------------------------------------------------------
    # Time intervals
    # --------------------------------------------------------

    if frame_times is None:

        delta_times = np.full(
            num_frames - 1,
            dt,
            dtype=float
        )

    else:

        frame_times = np.asarray(
            frame_times,
            dtype=float
        )

        delta_times = np.diff(frame_times)

    if np.any(delta_times <= 0):
        raise ValueError(
            "Frame times must be strictly increasing."
        )

    # --------------------------------------------------------
    # Displacements
    # --------------------------------------------------------

    displacements = np.diff(
        trajectories,
        axis=0
    )

    valid_previous = _finite_rows(
        trajectories[:-1]
    )

    valid_current = _finite_rows(
        trajectories[1:]
    )

    valid_velocity = (
        valid_previous
        & valid_current
    )

    # --------------------------------------------------------
    # Velocities
    # --------------------------------------------------------

    # Assign interval by interval so missing positions remain NaN and the
    # frame-specific time step is applied without ambiguous broadcasting.
    velocities = np.full_like(
        displacements,
        np.nan,
        dtype=float
    )

    for t in range(num_frames - 1):

        mask = valid_velocity[t]

        if np.any(mask):

            velocities[t, mask] = (
                displacements[t, mask]
                / delta_times[t]
            )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    speeds = np.full(
        velocities.shape[:2],
        np.nan,
        dtype=float
    )

    finite_velocity = _finite_rows(
        velocities
    )

    speeds[finite_velocity] = np.linalg.norm(
        velocities[finite_velocity],
        axis=1
    )

    # --------------------------------------------------------
    # Mean instantaneous speed per fish
    # --------------------------------------------------------

    mean_speed_per_fish = np.full(
        trajectories.shape[1],
        np.nan,
        dtype=float
    )

    for fish_idx in range(trajectories.shape[1]):

        valid = np.isfinite(
            speeds[:, fish_idx]
        )

        if np.any(valid):
            mean_speed_per_fish[fish_idx] = np.mean(
                speeds[valid, fish_idx]
            )

    # --------------------------------------------------------
    # Total observation time
    # --------------------------------------------------------

    if frame_times is None:
        total_time = (
            trajectories.shape[0] - 1
        ) * dt
    else:
        total_time = (
            frame_times[-1]
            - frame_times[0]
        )

    # --------------------------------------------------------
    # First-to-last displacement
    # --------------------------------------------------------

    first_to_last_displacement = np.full(
        (trajectories.shape[1], 3),
        np.nan,
        dtype=float
    )

    valid_first_last = (
        _finite_rows(trajectories[0])
        &
        _finite_rows(trajectories[-1])
    )

    first_to_last_displacement[
        valid_first_last
    ] = (
        trajectories[-1, valid_first_last]
        -
        trajectories[0, valid_first_last]
    )

    first_to_last_distance = np.full(
        trajectories.shape[1],
        np.nan,
        dtype=float
    )

    first_to_last_distance[
        valid_first_last
    ] = np.linalg.norm(
        first_to_last_displacement[
            valid_first_last
        ],
        axis=1
    )

    total_displacement_speed = (
        first_to_last_distance / total_time
    )

    # --------------------------------------------------------
    # Path length
    # --------------------------------------------------------

    path_lengths = np.zeros(
        trajectories.shape[1],
        dtype=float
    )

    for fish_idx in range(
        trajectories.shape[1]
    ):

        valid_steps = valid_velocity[
            :,
            fish_idx
        ]

        if np.any(valid_steps):

            path_lengths[fish_idx] = np.sum(
                np.linalg.norm(
                    displacements[
                        valid_steps,
                        fish_idx
                    ],
                    axis=1
                )
            )

    # --------------------------------------------------------
    # Path-average speed
    # --------------------------------------------------------

    path_average_speed = (
        path_lengths / total_time
    )

    return {
        "velocities": velocities,
        "speeds": speeds,
        "mean_speed_per_fish": mean_speed_per_fish,
        "total_displacement_speed": total_displacement_speed,
        "path_average_speed": path_average_speed,
        "path_lengths": path_lengths,
        "delta_times": delta_times,
        "total_time": total_time,
        "valid_velocity": valid_velocity,
    }


# ============================================================
# PAIRWISE DISTANCES
# ============================================================

def compute_pairwise_distances(trajectories):
    """
    Compute pairwise fish distances while ignoring missing fish.

    For a given frame, a pairwise distance is NaN if either fish
    is not observed.

    Mean distances are calculated only from valid pairs.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    num_frames, n_fish, _ = trajectories.shape

    distance_matrices = np.full(
        (num_frames, n_fish, n_fish),
        np.nan,
        dtype=float
    )

    mean_pairwise_distance_per_frame = np.full(
        num_frames,
        np.nan,
        dtype=float
    )

    for t in range(num_frames):

        valid = _finite_rows(
            trajectories[t]
        )

        valid_indices = np.flatnonzero(valid)

        if len(valid_indices) < 2:
            continue

        positions = trajectories[
            t,
            valid_indices
        ]

        differences = (
            positions[:, None, :]
            -
            positions[None, :, :]
        )

        distances = np.linalg.norm(
            differences,
            axis=2
        )

        distance_matrices[
            t
        ][
            np.ix_(
                valid_indices,
                valid_indices
            )
        ] = distances

        # Exclude diagonal.
        upper_triangle = distances[
            np.triu_indices(
                len(valid_indices),
                k=1
            )
        ]

        if upper_triangle.size > 0:
            mean_pairwise_distance_per_frame[t] = (
                np.mean(upper_triangle)
            )

    # --------------------------------------------------------
    # Overall mean
    # --------------------------------------------------------

    finite_frame_means = (
        mean_pairwise_distance_per_frame[
            np.isfinite(
                mean_pairwise_distance_per_frame
            )
        ]
    )

    if finite_frame_means.size:
        total_mean_pairwise_distance = np.mean(
            finite_frame_means
        )
    else:
        total_mean_pairwise_distance = np.nan

    # --------------------------------------------------------
    # Mean distance for each fish pair
    # --------------------------------------------------------

    mean_distance_between_each_pair = np.full(
        (n_fish, n_fish),
        np.nan,
        dtype=float
    )

    for i in range(n_fish):
        for j in range(n_fish):

            values = distance_matrices[
                :,
                i,
                j
            ]

            valid = np.isfinite(values)

            if np.any(valid):
                mean_distance_between_each_pair[
                    i,
                    j
                ] = np.mean(
                    values[valid]
                )

    # --------------------------------------------------------
    # First-to-last distance change
    # --------------------------------------------------------

    first_frame_distance_matrix = (
        distance_matrices[0]
    )

    last_frame_distance_matrix = (
        distance_matrices[-1]
    )

    change_first_to_last = (
        last_frame_distance_matrix
        -
        first_frame_distance_matrix
    )

    return {
        "distance_matrices": distance_matrices,
        "mean_pairwise_distance_per_frame":
            mean_pairwise_distance_per_frame,
        "total_mean_pairwise_distance":
            total_mean_pairwise_distance,
        "mean_distance_between_each_pair":
            mean_distance_between_each_pair,
        "first_frame_distance_matrix":
            first_frame_distance_matrix,
        "last_frame_distance_matrix":
            last_frame_distance_matrix,
        "change_first_to_last":
            change_first_to_last,
    }


# ============================================================
# Z STATISTICS
# ============================================================

def compute_z_statistics(trajectories):
    """
    Compute vertical-position statistics while ignoring missing
    observations.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    z_positions = trajectories[:, :, 2]

    n_fish = trajectories.shape[1]

    # --------------------------------------------------------
    # Mean z per fish
    # --------------------------------------------------------

    mean_z_per_fish = np.full(
        n_fish,
        np.nan,
        dtype=float
    )

    for fish_idx in range(n_fish):

        values = z_positions[
            :,
            fish_idx
        ]

        valid = np.isfinite(values)

        if np.any(valid):
            mean_z_per_fish[fish_idx] = (
                np.mean(values[valid])
            )

    # --------------------------------------------------------
    # Mean z per frame
    # --------------------------------------------------------

    mean_z_per_frame = np.full(
        z_positions.shape[0],
        np.nan,
        dtype=float
    )

    for t in range(
        z_positions.shape[0]
    ):

        values = z_positions[t]

        valid = np.isfinite(values)

        if np.any(valid):
            mean_z_per_frame[t] = (
                np.mean(values[valid])
            )

    # --------------------------------------------------------
    # Global mean z
    # --------------------------------------------------------

    valid_z = z_positions[
        np.isfinite(z_positions)
    ]

    if valid_z.size:
        total_mean_z = np.mean(valid_z)
    else:
        total_mean_z = np.nan

    # --------------------------------------------------------
    # First-to-last z change
    # --------------------------------------------------------

    first_to_last_z_change = np.full(
        n_fish,
        np.nan,
        dtype=float
    )

    valid = (
        np.isfinite(z_positions[0])
        &
        np.isfinite(z_positions[-1])
    )

    first_to_last_z_change[valid] = (
        z_positions[-1, valid]
        -
        z_positions[0, valid]
    )

    return {
        "z_positions": z_positions,
        "mean_z_per_fish": mean_z_per_fish,
        "mean_z_per_frame": mean_z_per_frame,
        "total_mean_z": total_mean_z,
        "first_to_last_z_change":
            first_to_last_z_change,
    }


# ============================================================
# ZONE RESIDENCE TIMES
# ============================================================

def compute_z_zone_times(
    trajectories,
    dt=1.0,
    z_limits=None
):
    """
    Calculate the time spent by each fish in four vertical zones.

    Missing observations do not contribute to the residence time.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    z = trajectories[:, :, 2]

    num_frames, n_fish = z.shape

    # --------------------------------------------------------
    # Determine z limits
    # --------------------------------------------------------

    if z_limits is None:

        finite_z = z[
            np.isfinite(z)
        ]

        if finite_z.size == 0:
            z_min = 0.0
            z_max = 1.0
        else:
            z_min = np.min(finite_z)
            z_max = np.max(finite_z)

    else:

        z_min, z_max = z_limits

    if z_max <= z_min:
        z_max = z_min + 1.0

    # Four zones require five boundaries.
    edges = np.linspace(
        z_min,
        z_max,
        5
    )

    # --------------------------------------------------------
    # Time per zone
    # --------------------------------------------------------

    time_per_zone_per_fish = np.zeros(
        (n_fish, 4),
        dtype=float
    )

    for fish_idx in range(n_fish):

        fish_z = z[
            :,
            fish_idx
        ]

        valid = np.isfinite(
            fish_z
        )

        if not np.any(valid):
            continue

        zone_indices = np.digitize(
            fish_z[valid],
            edges[1:-1],
            right=False
        )

        for zone in range(4):

            count = np.count_nonzero(
                zone_indices == zone
            )

            time_per_zone_per_fish[
                fish_idx,
                zone
            ] = count * dt

    # --------------------------------------------------------
    # Residence fractions
    # --------------------------------------------------------

    observation_time_per_fish = np.sum(
        time_per_zone_per_fish,
        axis=1
    )

    fraction_per_zone_per_fish = np.full(
        (n_fish, 4),
        np.nan,
        dtype=float
    )

    for fish_idx in range(n_fish):

        total_time = (
            observation_time_per_fish[fish_idx]
        )

        if total_time > 0:

            fraction_per_zone_per_fish[
                fish_idx
            ] = (
                time_per_zone_per_fish[
                    fish_idx
                ]
                /
                total_time
            )

    mean_time_per_zone = np.nanmean(
        time_per_zone_per_fish,
        axis=0
    )

    mean_fraction_per_zone = np.nanmean(
        fraction_per_zone_per_fish,
        axis=0
    )

    frame_durations = np.full(
        num_frames,
        dt,
        dtype=float
    )

    return {
        "z_edges": edges,
        "time_per_zone_per_fish":
            time_per_zone_per_fish,
        "fraction_per_zone_per_fish":
            fraction_per_zone_per_fish,
        "mean_time_per_zone":
            mean_time_per_zone,
        "mean_fraction_per_zone":
            mean_fraction_per_zone,
        "frame_durations":
            frame_durations,
    }


# ============================================================
# MSD
# ============================================================

def compute_msd(
    trajectories,
    dt=1.0,
    min_fit_points=5,
    min_r2=0.9,
    fit_end_time=1.5
):
    """
    Compute mean squared displacement from the initial position.

    For each fish, a displacement is only calculated at times for
    which both the initial position and current position are valid.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    num_frames, n_fish, _ = trajectories.shape

    displacement = np.full_like(
        trajectories,
        np.nan,
        dtype=float
    )

    initial_positions = trajectories[0]

    for fish_idx in range(n_fish):

        initial_valid = np.isfinite(
            initial_positions[fish_idx]
        ).all()

        if not initial_valid:
            continue

        current_valid = _finite_rows(
            trajectories[:, fish_idx]
        )

        displacement[
            current_valid,
            fish_idx
        ] = (
            trajectories[
                current_valid,
                fish_idx
            ]
            -
            initial_positions[
                fish_idx
            ]
        )

    msd_per_fish = np.full(
        (num_frames, n_fish),
        np.nan,
        dtype=float
    )

    valid_displacements = _finite_rows(
        displacement
    )

    msd_per_fish[
        valid_displacements
    ] = np.sum(
        displacement[
            valid_displacements
        ] ** 2,
        axis=1
    )

    msd_mean = np.full(
        num_frames,
        np.nan,
        dtype=float
    )

    for t in range(num_frames):

        values = msd_per_fish[t]

        valid = np.isfinite(values)

        if np.any(valid):
            msd_mean[t] = np.mean(
                values[valid]
            )

    lag_times = (
        np.arange(num_frames, dtype=float)
        * dt
    )

    # --------------------------------------------------------
    # Log-log fit
    # --------------------------------------------------------

    valid = (
        (lag_times > 0)
        &
        (lag_times <= fit_end_time)
        &
        (msd_mean > 0)
        &
        np.isfinite(msd_mean)
    )

    valid_indices = np.flatnonzero(
        valid
    )

    fit_start = None
    fit_end = None

    slope = np.nan
    intercept = np.nan
    r_squared = np.nan

    min_points = max(
        3,
        min_fit_points
    )

    if valid_indices.size >= min_points:

        indices = valid_indices

        log_times = np.log10(
            lag_times[indices]
        )

        log_msd = np.log10(
            msd_mean[indices]
        )

        coefficients = np.polyfit(
            log_times,
            log_msd,
            1
        )

        predicted = np.polyval(
            coefficients,
            log_times
        )

        residual = np.sum(
            (log_msd - predicted) ** 2
        )

        total = np.sum(
            (log_msd - np.mean(log_msd)) ** 2
        )

        r_squared = (
            1.0 - residual / total
            if total > 0
            else 1.0
        )

        slope, intercept = coefficients

        fit_start = int(indices[0])
        fit_end = int(indices[-1])

        # min_r2 is retained for API compatibility.
        # We do not discard the fit here because the original
        # function also returned the fitted coefficients.
        #
        # The R² value is reported to the caller instead.

    return {
        "lag_times": lag_times,
        "msd_per_fish": msd_per_fish,
        "msd_mean": msd_mean,
        "fit_start_index": fit_start,
        "fit_end_index": fit_end,
        "fit_start_time":
            lag_times[fit_start]
            if fit_start is not None
            else np.nan,
        "fit_end_time":
            lag_times[fit_end]
            if fit_end is not None
            else np.nan,
        "fit_requested_end_time": fit_end_time,
        "fit_exponent": slope,
        "fit_intercept_log10": intercept,
        "fit_r_squared": r_squared,
    }


# ============================================================
# NEAREST-NEIGHBOUR DISTANCES
# ============================================================

def compute_nearest_neighbor_distances(
    trajectories,
    histogram_bins=30
):
    """
    Calculate the nearest-neighbour distance for each fish.

    A fish is considered only when its position is valid.
    The nearest neighbour must also be a valid observation.

    Missing fish therefore do not contaminate the result.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    num_frames, n_fish, _ = trajectories.shape

    nearest_indices = np.full(
        (num_frames, n_fish),
        -1,
        dtype=int
    )

    nearest_distances = np.full(
        (num_frames, n_fish),
        np.nan,
        dtype=float
    )

    for t in range(num_frames):

        valid = _finite_rows(
            trajectories[t]
        )

        valid_indices = np.flatnonzero(valid)

        if len(valid_indices) < 2:
            continue

        positions = trajectories[
            t,
            valid_indices
        ]

        differences = (
            positions[:, None, :]
            -
            positions[None, :, :]
        )

        distances = np.linalg.norm(
            differences,
            axis=2
        )

        # Ignore self-distance.
        np.fill_diagonal(
            distances,
            np.inf
        )

        nearest_local = np.argmin(
            distances,
            axis=1
        )

        for local_idx, fish_idx in enumerate(
            valid_indices
        ):

            neighbour_local = nearest_local[
                local_idx
            ]

            nearest_indices[
                t,
                fish_idx
            ] = valid_indices[
                neighbour_local
            ]

            nearest_distances[
                t,
                fish_idx
            ] = distances[
                local_idx,
                neighbour_local
            ]

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    density, edges = _probability_histogram(
        nearest_distances.ravel(),
        histogram_bins
    )

    mean_nearest_neighbor_distance_per_frame = np.full(
        num_frames,
        np.nan,
        dtype=float
    )

    for t in range(num_frames):

        values = nearest_distances[t]

        valid = np.isfinite(values)

        if np.any(valid):
            mean_nearest_neighbor_distance_per_frame[t] = (
                np.mean(values[valid])
            )

    valid_all = np.isfinite(
        nearest_distances
    )

    if np.any(valid_all):
        mean_nearest_neighbor_distance = np.mean(
            nearest_distances[valid_all]
        )
    else:
        mean_nearest_neighbor_distance = np.nan

    return {
        "nearest_neighbor_indices":
            nearest_indices,
        "nearest_neighbor_distances":
            nearest_distances,
        "mean_nearest_neighbor_distance_per_frame":
            mean_nearest_neighbor_distance_per_frame,
        "mean_nearest_neighbor_distance":
            mean_nearest_neighbor_distance,
        "histogram_density":
            density,
        "histogram_edges":
            edges,
    }


# ============================================================
# STEP LENGTH DISTRIBUTION
# ============================================================

def compute_step_length_distribution(
    trajectories,
    histogram_bins=30
):
    """
    Calculate frame-to-frame displacement lengths.

    Steps are only included when both consecutive positions
    are valid.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    displacements = np.diff(
        trajectories,
        axis=0
    )

    valid = (
        _finite_rows(trajectories[:-1])
        &
        _finite_rows(trajectories[1:])
    )

    step_lengths = np.full(
        displacements.shape[:2],
        np.nan,
        dtype=float
    )

    step_lengths[valid] = np.linalg.norm(
        displacements[valid],
        axis=1
    )

    pooled_step_lengths = step_lengths[
        np.isfinite(step_lengths)
    ]

    density, edges = _probability_histogram(
        pooled_step_lengths,
        histogram_bins
    )

    return {
        "step_lengths": step_lengths,
        "pooled_step_lengths":
            pooled_step_lengths,
        "histogram_density": density,
        "histogram_edges": edges,
    }


# ============================================================
# VELOCITY POLARIZATION
# ============================================================

def compute_velocity_polarization(
    velocities
):
    """
    Compute the polarization vector and its magnitude.

    Missing velocities are ignored.
    """

    velocities = np.asarray(
        velocities,
        dtype=float
    )

    speeds = np.linalg.norm(
        velocities,
        axis=2
    )

    unit_velocities = np.full_like(
        velocities,
        np.nan,
        dtype=float
    )

    valid = (
        np.isfinite(velocities).all(axis=2)
        &
        (speeds > 0)
    )

    unit_velocities[valid] = (
        velocities[valid]
        /
        speeds[valid, None]
    )

    polarization_vectors = np.full(
        velocities.shape[:2],
        np.nan,
        dtype=float
    )

    # Correct shape is (time, 3).
    polarization_vectors = np.full(
        (velocities.shape[0], 3),
        np.nan,
        dtype=float
    )

    for t in range(
        velocities.shape[0]
    ):

        valid_t = np.isfinite(
            unit_velocities[t]
        ).all(axis=1)

        if np.any(valid_t):

            polarization_vectors[t] = np.mean(
                unit_velocities[
                    t,
                    valid_t
                ],
                axis=0
            )

    polarization_magnitude = np.full(
        velocities.shape[0],
        np.nan,
        dtype=float
    )

    valid_p = np.isfinite(
        polarization_vectors
    ).all(axis=1)

    polarization_magnitude[valid_p] = np.linalg.norm(
        polarization_vectors[valid_p],
        axis=1
    )

    return {
        "polarization_vectors":
            polarization_vectors,
        "polarization_magnitude":
            polarization_magnitude,
    }


# ============================================================
# VELOCITY AUTOCORRELATION
# ============================================================

def compute_velocity_autocorrelation(
    velocities,
    dt=1.0
):
    """
    Compute temporal velocity autocorrelation.

    At each lag, only pairs for which both velocities are
    finite are included.
    """

    velocities = np.asarray(
        velocities,
        dtype=float
    )

    num_frames = velocities.shape[0]

    values = np.full(
        num_frames,
        np.nan,
        dtype=float
    )

    valid_pair_counts = np.zeros(
        num_frames,
        dtype=int
    )

    for lag in range(num_frames):

        v1 = velocities[
            :num_frames - lag
        ]

        v2 = velocities[
            lag:
        ]

        valid = (
            _finite_rows(v1)
            &
            _finite_rows(v2)
        )

        if not np.any(valid):
            continue

        products = np.sum(
            v1[valid]
            *
            v2[valid],
            axis=1
        )

        values[lag] = np.mean(
            products
        )

        valid_pair_counts[lag] = (
            products.size
        )

    # --------------------------------------------------------
    # Normalize by C(0)
    # --------------------------------------------------------

    if (
        np.isfinite(values[0])
        and values[0] != 0
    ):
        values /= values[0]

    return {
        "lag_times":
            np.arange(num_frames) * dt,
        "autocorrelation":
            values,
        "valid_pair_counts":
            valid_pair_counts,
    }


# ============================================================
# NEIGHBOUR VELOCITY CORRELATION
# ============================================================

def compute_neighbor_velocity_correlation(
    trajectories,
    velocities,
    histogram_bins=20
):
    """
    Compute directional velocity correlation between each fish
    and its nearest neighbour.

    Only pairs for which both velocities and both positions are
    valid are included.
    """

    trajectories = np.asarray(
        trajectories,
        dtype=float
    )

    velocities = np.asarray(
        velocities,
        dtype=float
    )

    nearest = compute_nearest_neighbor_distances(
        trajectories
    )

    nearest_indices = nearest[
        "nearest_neighbor_indices"
    ]

    nearest_distances = nearest[
        "nearest_neighbor_distances"
    ]

    num_velocity_frames = velocities.shape[0]

    # nearest-neighbour information corresponds to position
    # frames, while velocities correspond to intervals between
    # frames. Use the starting frame of each velocity interval.
    distances = nearest_distances[
        :-1
    ]

    correlations = np.full(
        distances.shape,
        np.nan,
        dtype=float
    )

    for t in range(
        num_velocity_frames
    ):

        for fish_idx in range(
            trajectories.shape[1]
        ):

            neighbour_idx = nearest_indices[
                t,
                fish_idx
            ]

            if neighbour_idx < 0:
                continue

            v1 = velocities[
                t,
                fish_idx
            ]

            v2 = velocities[
                t,
                neighbour_idx
            ]

            if not (
                np.isfinite(v1).all()
                and
                np.isfinite(v2).all()
            ):
                continue

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                continue

            correlations[
                t,
                fish_idx
            ] = np.dot(
                v1,
                v2
            ) / (
                norm1 * norm2
            )

    # --------------------------------------------------------
    # Distance bins
    # --------------------------------------------------------

    valid_distances = distances[
        np.isfinite(distances)
        &
        np.isfinite(correlations)
    ]

    if valid_distances.size == 0:

        density_edges = np.linspace(
            0,
            1,
            histogram_bins + 1
        )

    else:

        density_edges = np.histogram_bin_edges(
            valid_distances,
            bins=histogram_bins
        )

        # Avoid degenerate binning if all distances are equal.
        if len(density_edges) < 2:
            value = valid_distances[0]
            width = max(
                abs(value) * 0.05,
                1e-12
            )

            density_edges = np.array([
                value - width,
                value + width
            ])

    bin_indices = np.digitize(
        distances.ravel(),
        density_edges
    ) - 1

    mean_correlation = np.full(
        len(density_edges) - 1,
        np.nan,
        dtype=float
    )

    counts = np.zeros(
        len(density_edges) - 1,
        dtype=int
    )

    flat_correlations = correlations.ravel()
    flat_distances = distances.ravel()

    for bin_index in range(
        len(mean_correlation)
    ):

        mask = (
            (bin_indices == bin_index)
            &
            np.isfinite(flat_correlations)
            &
            np.isfinite(flat_distances)
        )

        counts[bin_index] = np.count_nonzero(
            mask
        )

        if counts[bin_index] > 0:

            mean_correlation[
                bin_index
            ] = np.mean(
                flat_correlations[mask]
            )

    return {
        "distances": distances,
        "correlations": correlations,
        "bin_edges": density_edges,
        "bin_centers": (
            density_edges[:-1]
            +
            density_edges[1:]
        ) / 2,
        "mean_correlation":
            mean_correlation,
        "valid_pair_counts":
            counts,
    }
