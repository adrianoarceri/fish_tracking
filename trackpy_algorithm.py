import trackpy as tp
import pandas as pd
import numpy as np

def track_with_trackpy(all_positions, search_range=20.0, memory=5):
    num_frames, n_fish, _ = all_positions.shape
    
    # flatten the 3d array into a pandas dataframe
    records = []
    for t in range(num_frames):
        for i in range(n_fish):
            x, y, z = all_positions[t, i]
            records.append({'frame': t, 'x': x, 'y': y, 'z': z})
            
    df = pd.DataFrame(records)
    
    # execute the trackpy algorithm
    linked_df = tp.link(
        df, 
        search_range, 
        memory=memory, 
        pos_columns=['x', 'y', 'z']
    )
    
    # filter out temporary tracking artifacts
    valid_trajectories = tp.filter_stubs(linked_df, threshold=num_frames * 0.5)
    
    # reconstruct the original output matrix
    particle_counts = valid_trajectories['particle'].value_counts()
    top_particles = particle_counts.nlargest(n_fish).index.tolist()
    
    final_trajectories = np.full((num_frames, n_fish, 3), np.nan)
    for new_idx, particle_id in enumerate(top_particles):
        fish_data = valid_trajectories[valid_trajectories['particle'] == particle_id]
        frames = fish_data['frame'].values
        coords = fish_data[['x', 'y', 'z']].values
        final_trajectories[frames, new_idx, :] = coords
        
    # interpolate coordinate gaps caused by memory occlusions
    for new_idx in range(n_fish):
        for dim in range(3):
            s = pd.Series(final_trajectories[:, new_idx, dim])
            final_trajectories[:, new_idx, dim] = s.interpolate(limit_direction='both').values

    return final_trajectories