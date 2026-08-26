"""Detection loading and identity tracking for reconstructed fish point clouds."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ============================================================
# DATA LOADING
# ============================================================

def load_npz(path):
    """Load every component stored in one reconstruction ``.npz`` file."""
    with np.load(path, allow_pickle=False) as data:
        return [np.asarray(data[key]) for key in data.files]


def get_coms(data, min_component_points=1):
    """Return valid component centres of mass and their voxel counts.

    A reconstruction component is expected to have shape ``(3, n_points)``.
    Components below ``min_component_points`` are deliberately discarded before
    tracking: they are usually isolated reconstruction noise, not a measured
    fish. The resulting number of candidates can vary from frame to frame.
    """
    if min_component_points < 1:
        raise ValueError("min_component_points must be at least 1.")

    centres = []
    point_counts = []

    for component in data:
        component = np.asarray(component, dtype=float)

        if component.ndim != 2 or component.shape[0] != 3:
            raise ValueError(
                "Each NPZ component must have shape (3, n_points); "
                f"received {component.shape}."
            )

        finite_points = np.isfinite(component).all(axis=0)
        point_count = int(np.count_nonzero(finite_points))

        if point_count < min_component_points:
            continue

        centres.append(component[:, finite_points].mean(axis=1))
        point_counts.append(point_count)

    if not centres:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=int)

    return np.asarray(centres, dtype=float), np.asarray(point_counts, dtype=int)


def _normalise_detection_frames(all_positions):
    """Convert legacy arrays or variable-length sequences to detection frames."""
    if isinstance(all_positions, np.ndarray):
        positions = np.asarray(all_positions, dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(
                "A position array must have shape (num_frames, n_detections, 3)."
            )
        raw_frames = [positions[t] for t in range(positions.shape[0])]
    else:
        raw_frames = list(all_positions)
        if not raw_frames:
            raise ValueError("At least one detection frame is required.")

    frames = []
    for frame in raw_frames:
        frame = np.asarray(frame, dtype=float)
        if frame.size == 0:
            frame = np.empty((0, 3), dtype=float)
        if frame.ndim != 2 or frame.shape[1] != 3:
            raise ValueError(
                "Each detection frame must have shape (n_detections, 3)."
            )
        frames.append(frame[np.isfinite(frame).all(axis=1)])

    return frames


def _normalise_detection_scores(candidate_scores, detection_frames):
    """Validate optional component-size scores used to seed track identities."""
    if candidate_scores is None:
        return [np.ones(len(frame), dtype=float) for frame in detection_frames]

    scores = list(candidate_scores)
    if len(scores) != len(detection_frames):
        raise ValueError("candidate_scores must contain one array per frame.")

    normalised = []
    for frame, frame_scores in zip(detection_frames, scores):
        frame_scores = np.asarray(frame_scores, dtype=float)
        if frame_scores.shape != (len(frame),):
            raise ValueError(
                "Each candidate-score array must match its detection-frame length."
            )
        normalised.append(frame_scores)

    return normalised


# ============================================================
# TRACKING
# ============================================================

def track_with_velocity_prediction(
    all_positions,
    n_fish=None,
    dt=1.0,
    max_displacement=20.0,
    velocity_smoothing=0.5,
    max_gap_frames=15,
    candidate_scores=None,
    return_diagnostics=False,
):
    """Track known fish identities with prediction, gating, and assignment.

    ``all_positions`` may contain a different number of detections in each
    frame. The exported trajectory has the fixed shape
    ``(num_frames, n_fish, 3)`` and contains ``NaN`` for a fish not observed in
    a frame. Predicted positions are retained only as *internal tracker state*;
    they are never written into the scientific trajectory.

    The first frame with at least ``n_fish`` candidates establishes arbitrary
    but stable identities, selecting the largest components when
    ``candidate_scores`` are supplied. Extra candidates remain available to
    the Hungarian assignment but are ignored when unmatched.

    ``max_displacement`` is a one-frame gate in physical units. After a gap,
    the gate grows linearly with the number of elapsed frames, while tracking
    remains active for at most ``max_gap_frames`` missing frames. This avoids
    both immediate loss of kinematic memory and indefinite, low-confidence
    re-identification.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if max_displacement <= 0:
        raise ValueError("max_displacement must be > 0.")
    if not 0.0 <= velocity_smoothing <= 1.0:
        raise ValueError("velocity_smoothing must be between 0 and 1.")
    if max_gap_frames is not None and max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative or None.")

    detection_frames = _normalise_detection_frames(all_positions)
    score_frames = _normalise_detection_scores(candidate_scores, detection_frames)

    if n_fish is None:
        if isinstance(all_positions, np.ndarray):
            n_fish = all_positions.shape[1]
        else:
            raise ValueError("n_fish is required for variable-length detections.")
    if n_fish < 1:
        raise ValueError("n_fish must be at least 1.")

    num_frames = len(detection_frames)
    trajectories = np.full((num_frames, n_fish, 3), np.nan, dtype=float)
    assignment_distances = np.full((num_frames, n_fish), np.nan, dtype=float)
    gap_lengths = np.full((num_frames, n_fish), -1, dtype=int)
    accepted_assignments = np.zeros(num_frames, dtype=int)
    rejected_by_gate = np.zeros(num_frames, dtype=int)
    ignored_detections = np.zeros(num_frames, dtype=int)

    # A tracker cannot create a biological identity if the reconstruction never
    # supplies enough candidates. The largest components seed the identities.
    seed_frame = next(
        (index for index, frame in enumerate(detection_frames) if len(frame) >= n_fish),
        None,
    )
    if seed_frame is None:
        max_candidates = max(len(frame) for frame in detection_frames)
        raise ValueError(
            f"Cannot initialise {n_fish} tracks: the largest frame contains only "
            f"{max_candidates} candidate components."
        )

    seed_order = np.argsort(score_frames[seed_frame], kind="stable")[::-1]
    seed_indices = seed_order[:n_fish]
    seed_positions = detection_frames[seed_frame][seed_indices]

    trajectories[seed_frame] = seed_positions
    assignment_distances[seed_frame] = 0.0
    accepted_assignments[seed_frame] = n_fish
    ignored_detections[seed_frame] = len(detection_frames[seed_frame]) - n_fish

    # This state is separate from trajectories. It deliberately survives an
    # exported NaN so a fish can be matched after a multi-frame occlusion.
    state_positions = seed_positions.copy()
    last_observed_positions = seed_positions.copy()
    last_observed_frames = np.full(n_fish, seed_frame, dtype=int)
    velocities = np.zeros((n_fish, 3), dtype=float)
    missing_counts = np.zeros(n_fish, dtype=int)
    active = np.ones(n_fish, dtype=bool)

    # Frames before the seed have no justified identity assignment.
    for t in range(seed_frame):
        ignored_detections[t] = len(detection_frames[t])

    for t in range(seed_frame + 1, num_frames):
        detections = detection_frames[t]
        track_indices = np.flatnonzero(active)

        # Advance the state even if the detection is missed. This is the
        # essential difference from deriving predictions from trajectories.
        if track_indices.size:
            state_positions[track_indices] += velocities[track_indices] * dt

        assigned_tracks = set()
        assigned_detection_indices = set()

        if track_indices.size and len(detections):
            predicted_positions = state_positions[track_indices]
            cost_matrix = cdist(predicted_positions, detections)

            elapsed_frames = missing_counts[track_indices] + 1
            gate_per_track = max_displacement * elapsed_frames
            valid_cost = cost_matrix <= gate_per_track[:, None]
            large_cost = max_displacement * (num_frames + 1) * 1e6
            assignment_cost = np.where(valid_cost, cost_matrix, large_cost)

            rows, cols = linear_sum_assignment(assignment_cost)
            for row, col in zip(rows, cols):
                fish_index = track_indices[row]
                distance = cost_matrix[row, col]

                if distance > gate_per_track[row]:
                    rejected_by_gate[t] += 1
                    continue

                position = detections[col]
                elapsed_since_observation = t - last_observed_frames[fish_index]
                measured_velocity = (
                    position - last_observed_positions[fish_index]
                ) / (elapsed_since_observation * dt)

                velocities[fish_index] = (
                    velocity_smoothing * measured_velocity
                    + (1.0 - velocity_smoothing) * velocities[fish_index]
                )
                state_positions[fish_index] = position
                last_observed_positions[fish_index] = position
                last_observed_frames[fish_index] = t
                missing_counts[fish_index] = 0

                trajectories[t, fish_index] = position
                assignment_distances[t, fish_index] = distance
                assigned_tracks.add(fish_index)
                assigned_detection_indices.add(col)

        accepted_assignments[t] = len(assigned_tracks)
        ignored_detections[t] = len(detections) - len(assigned_detection_indices)

        for fish_index in track_indices:
            if fish_index in assigned_tracks:
                continue

            missing_counts[fish_index] += 1
            if max_gap_frames is not None and missing_counts[fish_index] > max_gap_frames:
                active[fish_index] = False

        gap_lengths[t] = missing_counts

    diagnostics = {
        "initialization_frame": seed_frame,
        "initial_component_indices": seed_indices,
        "candidate_counts": np.asarray([len(frame) for frame in detection_frames]),
        "accepted_assignments_per_frame": accepted_assignments,
        "gated_assignments_per_frame": rejected_by_gate,
        "ignored_detections_per_frame": ignored_detections,
        "assignment_distances": assignment_distances,
        "gap_lengths": gap_lengths,
        "final_active_tracks": active.copy(),
        "max_gap_frames": max_gap_frames,
    }

    if return_diagnostics:
        return trajectories, diagnostics
    return trajectories


# ============================================================
# TRAJECTORY CONSTRUCTION
# ============================================================

def build_trajectories(
    recon_folder=None,
    n_fish=8,
    grid_step=0.75,
    tracking_mode="velocity",
    save_folder=None,
    frame_files=None,
    dt=1.0,
    max_displacement=20.0,
    velocity_smoothing=0.5,
    min_component_points=10,
    max_gap_frames=15,
    return_diagnostics=False,
):
    """Build fixed-identity trajectories from variable component detections.

    ``min_component_points=10`` is a conservative initial noise filter for the
    supplied reconstruction. It is intentionally an exposed parameter: the
    diagnostics from the first tracking run must be used to validate or tune
    it for an experiment, not treated as a biological constant.
    """
    if n_fish < 1:
        raise ValueError("n_fish must be at least 1.")
    if grid_step <= 0:
        raise ValueError("grid_step must be > 0.")

    if frame_files is None:
        frame_files = sorted(glob.glob(os.path.join(recon_folder, "frame-*.npz")))
    if not frame_files:
        raise ValueError("Nessun file frame-*.npz trovato.")

    candidate_positions = []
    candidate_point_counts = []
    raw_component_counts = []

    for frame_path in frame_files:
        components = load_npz(frame_path)
        raw_component_counts.append(len(components))
        centres, point_counts = get_coms(
            components,
            min_component_points=min_component_points,
        )
        candidate_positions.append(centres * grid_step)
        candidate_point_counts.append(point_counts)

    if tracking_mode not in {"velocity", "constant_velocity"}:
        raise ValueError(
            "Variable detection counts require tracking_mode='velocity' or "
            "'constant_velocity'."
        )

    trajectories, tracker_diagnostics = track_with_velocity_prediction(
        candidate_positions,
        n_fish=n_fish,
        dt=dt,
        max_displacement=max_displacement,
        velocity_smoothing=velocity_smoothing,
        max_gap_frames=max_gap_frames,
        candidate_scores=candidate_point_counts,
        return_diagnostics=True,
    )

    filtered_component_counts = np.asarray(
        [len(frame) for frame in candidate_positions], dtype=int
    )
    raw_component_counts = np.asarray(raw_component_counts, dtype=int)
    diagnostics = {
        **tracker_diagnostics,
        "raw_component_counts": raw_component_counts,
        "filtered_component_counts": filtered_component_counts,
        "dropped_component_counts": raw_component_counts - filtered_component_counts,
        "min_component_points": min_component_points,
    }

    _print_tracking_diagnostics(trajectories, diagnostics, n_fish)

    if save_folder is not None:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        np.save(save_folder / "all_trajectories.npy", trajectories)
        for fish_index in range(n_fish):
            np.save(
                save_folder / f"fish_{fish_index:02d}_trajectory.npy",
                trajectories[:, fish_index, :],
            )
        np.save(save_folder / "frame_files.npy", np.asarray(frame_files))
        _save_tracking_diagnostics(save_folder, diagnostics)

    if return_diagnostics:
        return trajectories, frame_files, diagnostics
    return trajectories, frame_files


def _print_tracking_diagnostics(trajectories, diagnostics, n_fish):
    """Print compact diagnostics needed before behavioral metrics are trusted."""
    observed = np.isfinite(trajectories).all(axis=2)
    total_positions = trajectories.shape[0] * n_fish
    assignment_distances = diagnostics["assignment_distances"]
    finite_distances = assignment_distances[np.isfinite(assignment_distances)]

    print("\n================ TRACKING DIAGNOSTICS ================")
    print(f"Frame analizzati: {trajectories.shape[0]}")
    print(f"Pesci richiesti: {n_fish}")
    print(
        "Componenti grezzi/frame (min-mediana-max): "
        f"{diagnostics['raw_component_counts'].min()}-"
        f"{np.median(diagnostics['raw_component_counts']):.0f}-"
        f"{diagnostics['raw_component_counts'].max()}"
    )
    print(
        "Candidati dopo filtro/frame (min-mediana-max): "
        f"{diagnostics['filtered_component_counts'].min()}-"
        f"{np.median(diagnostics['filtered_component_counts']):.0f}-"
        f"{diagnostics['filtered_component_counts'].max()} "
        f"(min punti={diagnostics['min_component_points']})"
    )
    print(
        f"Posizioni valide: {np.count_nonzero(observed)}/{total_positions} "
        f"({100 * np.count_nonzero(observed) / total_positions:.2f}%)"
    )
    print(f"Assegnazioni rifiutate dal gate: {diagnostics['gated_assignments_per_frame'].sum()}")
    print(f"Candidati ignorati: {diagnostics['ignored_detections_per_frame'].sum()}")
    if finite_distances.size:
        print(
            "Distanza assegnazione (mediana / 95° percentile): "
            f"{np.median(finite_distances):.3f} / "
            f"{np.percentile(finite_distances, 95):.3f} mm"
        )
    for fish_index in range(n_fish):
        print(
            f"Fish {fish_index}: {np.count_nonzero(observed[:, fish_index])}/"
            f"{trajectories.shape[0]} frame osservati"
        )
    print("=======================================================\n")


def _save_tracking_diagnostics(save_folder, diagnostics):
    """Persist arrays needed to audit assignment quality after a run."""
    arrays = {
        "tracking_raw_component_counts.npy": diagnostics["raw_component_counts"],
        "tracking_filtered_component_counts.npy": diagnostics["filtered_component_counts"],
        "tracking_dropped_component_counts.npy": diagnostics["dropped_component_counts"],
        "tracking_accepted_assignments_per_frame.npy": diagnostics[
            "accepted_assignments_per_frame"
        ],
        "tracking_gated_assignments_per_frame.npy": diagnostics[
            "gated_assignments_per_frame"
        ],
        "tracking_ignored_detections_per_frame.npy": diagnostics[
            "ignored_detections_per_frame"
        ],
        "tracking_assignment_distances.npy": diagnostics["assignment_distances"],
        "tracking_gap_lengths.npy": diagnostics["gap_lengths"],
    }
    for filename, array in arrays.items():
        np.save(save_folder / filename, array)
