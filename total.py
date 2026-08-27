from pathlib import Path

from pipeline import analyze_fish_trajectories

input_root = Path("8_fish")
output_destination = "8_fish_analysis_output"
number_of_fish = 8

input_recon_folders = sorted(input_root.glob("*_ANALISI/recon"))
if not input_recon_folders:
    raise FileNotFoundError(f"Nessuna cartella *_ANALISI/recon trovata in {input_root}")

results = {}
for input_recon_folder in input_recon_folders:
    session_name = input_recon_folder.parent.name
    print(f"\n##### Analisi di {session_name} #####")
    results[session_name] = analyze_fish_trajectories(
        recon_folder=input_recon_folder,
        n_fish=number_of_fish,
        output_folder=output_destination,
        grid_step=0.75,
        dt=1/15,
        tracking_mode="velocity",
        make_video=True,
        search_range=15.0,        # initial one-frame gate in mm; validate after diagnostics
        memory=15,                # maximum occlusion length tracked with internal prediction
        min_component_points=10,  # initial reconstruction-noise filter; validate after diagnostics
    )