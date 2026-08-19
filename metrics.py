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