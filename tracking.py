import numpy as np
import os
import glob
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# caricamento dei singoli file npz per analisi
def load_npz(path):
    data = np.load(path)
    return [data[k] for k in data.files]

# calcolo del centro di massa
def get_coms(data):
    centers=[]
    for fish_points in data:
        x,y,z=fish_points
        centers.append([x.mean(), y.mean(), z.mean()])

    return np.array(centers)

"""
funzione per verificare l'identità dei pesci nel tempo.
Mantiene l'identità dei pesci nel tempo associando i pesci del frame corrente a quelli del frame precedente tramite distanza minima.
Input: (numero di frame, numero di pesci, 3 coordinate spaziali)-> output:(numero di frame, numero di pesci, 3 coordinate spaziali)
NOTA: non so se serve o se per qualche miracolo salva i pesci nello stesso ordine ogni volta, dubito fortemente
"""
def track_by_nearest_neighbor(all_positions):

    num_frames,n_fish,_= all_positions.shape

# dove salvo le traiettorie (inizializzato a zero)
    trajectories=np.zeros_like(all_positions)
    trajectories[0]=all_positions[0] # per il primo frame non so nulla, salvo solo l'identità delle traiettorie

# ora loop dal frame 1 fino all'ultimo e aggiorno le traiettorie salvando la posizione precedente e quella successiva
    for t in range(1, num_frames):
        previous_positions=trajectories[t - 1]
        current_positions= all_positions[t]

        # matrice delle distanze per capire quale pesce era quale alla posizione precedente
        # essenzilamente cost_matrix(i, j)= è la distanza tra l'oggetto i nel frame precedente all'oggetto j nel frame corrente
        cost_matrix=cdist(previous_positions, current_positions)

        # minimizzo la distanza totale in modo da identificare il pesce
        row_ind, col_ind=linear_sum_assignment(cost_matrix)

        # matrice di riorganizzazione
        reordered = np.zeros_like(current_positions)

      # ordino
        for previous_fish_idx, current_fish_idx in zip(row_ind, col_ind):
            reordered[previous_fish_idx] =current_positions[current_fish_idx]

        trajectories[t]=reordered

    return trajectories

"""
Funzione che costruisce le traiettorie.
"""
def build_trajectories(recon_folder=None, n_fish=8, grid_step=0.75, tracking_mode="nearest", save_folder=None, frame_files=None):

    if frame_files is None:
        frame_files=sorted(glob.glob(os.path.join(recon_folder, "frame-*.npz")))

    if len(frame_files)==0:
        raise ValueError("Nessun file frame-*.npz trovato.")

    all_positions=[]

    for frame_path in frame_files:
        data =load_npz(frame_path)[:n_fish]

        if len(data)<n_fish:
            raise ValueError(f"Nel frame {frame_path} ho trovato solo {len(data)} oggetti, ma n_fish={n_fish}")

        coms=get_coms(data)

         # IMPORTANTISSIMO: conversione in mm
        coms=coms*grid_step

        all_positions.append(coms)

    all_positions=np.array(all_positions)

# piccolo check per vedere se nearest funziona decentemente: se scelgo order non riordina nulla (lascia essenzialmente gli array delle posizioni così come sono)
    if tracking_mode=="order":
        trajectories=all_positions.copy()

# se scelgo nearest fa quella cosa implementata prima secondo me è utile verificare come vanno entrambe
    elif tracking_mode=="nearest":
        trajectories=track_by_nearest_neighbor(all_positions)

    else:
        raise ValueError("tracking_mode deve essere 'order' oppure 'nearest'.")

# cose per salvare
    if save_folder is not None:
        save_folder=Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)

        np.save(save_folder / "all_trajectories.npy", trajectories)

        for fish_idx in range(n_fish):
            np.save(save_folder / f"fish_{fish_idx:02d}_trajectory.npy", trajectories[:, fish_idx, :])

        np.save(save_folder / "frame_files.npy", np.array(frame_files))

    return trajectories, frame_files