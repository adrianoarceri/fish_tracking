import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import os

def set_axes_equal_3d(ax):
    """
    Imposta le stesse proporzioni per gli assi 3D in modo che il volume 
    sia rappresentato senza distorsioni spaziali.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def plot_collective_trajectories(trajectories, save_path=None):
    num_frames, n_fish, _ = trajectories.shape
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for fish_idx in range(n_fish):
        fish_traj = trajectories[:, fish_idx, :]
        ax.plot(fish_traj[:, 0], fish_traj[:, 1], fish_traj[:, 2], label=f'Pesce {fish_idx}')
        # Segna l'inizio (cerchio verde) e la fine (X rossa)
        ax.scatter(fish_traj[0, 0], fish_traj[0, 1], fish_traj[0, 2], marker='o', color='green')
        ax.scatter(fish_traj[-1, 0], fish_traj[-1, 1], fish_traj[-1, 2], marker='x', color='red')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Traiettorie collettive')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    set_axes_equal_3d(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def plot_single_fish_trajectory(trajectories, fish_idx, save_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    fish_traj = trajectories[:, fish_idx, :]
    ax.plot(fish_traj[:, 0], fish_traj[:, 1], fish_traj[:, 2], color='blue')
    ax.scatter(fish_traj[0, 0], fish_traj[0, 1], fish_traj[0, 2], marker='o', color='green', label='Inizio')
    ax.scatter(fish_traj[-1, 0], fish_traj[-1, 1], fish_traj[-1, 2], marker='x', color='red', label='Fine')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Traiettoria singola - pesce {fish_idx}')
    ax.legend()
    
    set_axes_equal_3d(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def make_trajectory_video(trajectories, save_path, fps=15):
    num_frames, n_fish, _ = trajectories.shape
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    lines = []
    points = []
    for fish_idx in range(n_fish):
        line, = ax.plot([], [], [], lw=2)
        point, = ax.plot([], [], [], 'o', markersize=6)
        lines.append(line)
        points.append(point)

    # Calcolo dei limiti globali per fissare la telecamera
    x_min, y_min, z_min = np.min(trajectories, axis=(0, 1))
    x_max, y_max, z_max = np.max(trajectories, axis=(0, 1))

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    set_axes_equal_3d(ax)

    def update(frame):
        for fish_idx in range(n_fish):
            fish_traj = trajectories[:frame+1, fish_idx, :]
            
            # Aggiornamento traiettoria
            lines[fish_idx].set_data(fish_traj[:, 0], fish_traj[:, 1])
            lines[fish_idx].set_3d_properties(fish_traj[:, 2])
            
            # Aggiornamento posizione attuale
            points[fish_idx].set_data([fish_traj[-1, 0]], [fish_traj[-1, 1]])
            points[fish_idx].set_3d_properties([fish_traj[-1, 2]])
        return lines + points

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)

    ext = os.path.splitext(save_path)[-1].lower()
    if ext == '.mp4':
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Zebrafish Tracker'), bitrate=1800)
        anim.save(save_path, writer=writer)
    elif ext == '.gif':
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    else:
        raise ValueError("Estensione file non supportata. Usa .mp4 o .gif")

    plt.close(fig)