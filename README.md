Zebrafish 3D trajectory analysis pipeline

This pipeline processes 3D reconstructed point cloud data of zebrafish to extract physical coordinates, track individual fish across frames using a nearest-neighbor algorithm, and compute behavioral and kinematic metrics. It automatically generates numpy arrays, static plots, and a video animation of the tracking session.

How to use the pipeline

* Activate your python virtual environment in the terminal.
* Open the `main.py` script.
* Modify the `input_recon_folder` variable to point to your specific directory containing the `frame-*.npz` files.
* Set the `number_of_fish` variable to match your experimental setup.
* Run the script from your terminal using the command `python main.py`.

Detailed script breakdown

* `main.py`
This is the entry point of the pipeline. It defines the exact paths for the input and output directories and sets the physical parameters, such as the grid step, the time delta between frames, and the expected number of fish. It triggers the entire analytical process by calling the main wrapper function.
* `pipeline.py`
This script acts as the main execution wrapper. It brings together all the modularized functions. It first calls the tracking module to build the trajectories, then passes those trajectories to the metrics module to compute velocities, distances, and z-zone statistics. Finally, it saves all the numerical results as `.npy` arrays, calls the plotting module to export images and video files, and prints a summary to the console.
* `tracking.py`
This module handles data loading and spatial processing. It extracts the center of mass for each fish from the raw numpy archives and scales the coordinates into millimeters using a 0.75 conversion factor. Crucially, it applies a nearest-neighbor tracking algorithm using the hungarian method to maintain the identity of each fish across consecutive frames and minimize total spatial displacement.
* `metrics.py`
This file contains all the mathematical and statistical calculations. It computes instantaneous frame-by-frame velocity vectors and absolute speeds. It evaluates group cohesion by calculating pairwise distances between all fish. It also analyzes the vertical position of the group, dividing the water column into four vertical z-zones to determine the residence time fraction of each fish in specific layers.
* `plotting.py`
This module generates the visual output. It provides functions to render 3D static plots of both collective and individual fish trajectories, forcing equal aspect ratios to prevent spatial distortion. It also contains the routine to animate the trajectories over time, exporting an `.mp4` video (or falling back to `.gif` if FFmpeg is unavailable).