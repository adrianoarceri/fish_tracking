from pipeline import analyze_fish_trajectories

input_recon_folder = r"various_heights\H11_ANALISI\10_20_25_ANALISI\recon"
output_destination = r"various_heights\H11_ANALISI\10_20_25_ANALISI\results"
number_of_fish = 8

results = analyze_fish_trajectories(
    recon_folder=input_recon_folder,
    n_fish=number_of_fish,
    output_folder=output_destination,
    grid_step=0.75,
    dt=1/15,
    tracking_mode="velocity",
    make_video=True,  # set to True to generate a video of the tracking results
    search_range=15.0,        # initial one-frame gate in mm; validate after diagnostics
    memory=15,                # maximum occlusion length tracked with internal prediction
    min_component_points=10,  # initial reconstruction-noise filter; validate after diagnostics
)
