from pipeline import analyze_fish_trajectories

# define your specific paths and parameters
input_recon_folder = "8_fish/12_15_33_ANALISI/recon"  # insert your actual folder path here
output_destination = "analysis_output"
number_of_fish = 8  # specify the expected number of fish

# run the tracking and metrics pipeline
results = analyze_fish_trajectories(
    recon_folder=input_recon_folder,
    n_fish=number_of_fish,
    output_folder=output_destination,
    grid_step=0.75,
    dt=1/15,
    tracking_mode="nearest",
    make_video=True
)