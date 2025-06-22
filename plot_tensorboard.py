import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import termios
import sys
import tty


def extract_data_from_tensorboard(
    log_dir: str, tags: List[str]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Extract scalar data for each run and tag found in ``log_dir``.

    The directory may contain multiple runs (subdirectories). This function
    keeps the data of each run separate so that plotting does not connect the end
    of one run with the start of the next. Additionally, it reads ``notes.txt``
    in each run folder to append notes to the run name in the legend.
    """

    # Nested dictionary ``data[run][tag]`` where the values are lists of
    # ``(step, value)`` tuples. ``run`` is the relative directory name that
    # contains the TensorBoard event file.
    data: Dict[str, Dict[str, List[tuple]]] = {}
    run_notes: Dict[str, str] = {}

    # Traverse ``log_dir`` recursively and read all TensorBoard event files.
    for run_folder in os.listdir(log_dir):
        run_path = os.path.join(log_dir, run_folder)
        if os.path.isdir(run_path):
            tensorboard_folder = os.path.join(run_path, "tensorboard")
            notes_file = os.path.join(run_path, "notes.txt")

            # Read notes.txt if it exists
            notes = ""
            if os.path.isfile(notes_file):
                with open(notes_file, "r") as f:
                    notes = f.read().strip()
            run_notes[run_folder] = notes

            # Ensure tensorboard_folder exists before processing files
            if os.path.isdir(tensorboard_folder):
                for file in os.listdir(tensorboard_folder):
                    if file.startswith("events.out.tfevents"):
                        event_file_path = os.path.join(tensorboard_folder, file)

                        # Use the run folder name as the run name
                        if run_folder not in data:
                            data[run_folder] = {tag: [] for tag in tags}

                        # Load events from the file
                        ea = event_accumulator.EventAccumulator(event_file_path)
                        ea.Reload()

                        # Extract the requested scalar data
                        for tag in tags:
                            if tag in ea.Tags()["scalars"]:
                                scalar_data = ea.Scalars(tag)
                                for scalar in scalar_data:
                                    data[run_folder][tag].append(
                                        (scalar.step, scalar.value)
                                    )
                    else:
                        print(f"Skipping non-event file: {file}")
            else:
                print(f"TensorBoard folder not found: {tensorboard_folder}")

    # Convert lists into sorted DataFrames for easier plotting
    run_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    for run_name, tag_dict in data.items():
        run_dfs[run_name] = {}
        for tag, values in tag_dict.items():
            df = pd.DataFrame(values, columns=["Step", tag])
            run_dfs[run_name][tag] = df.sort_values("Step")

    return run_dfs, run_notes


def smooth_data(series: pd.Series, factor: float) -> pd.Series:
    """Smooth the data using exponential moving average."""
    return series.ewm(alpha=factor).mean()


def plot_tensorboard_data(
    run_dfs: Dict[str, Dict[str, pd.DataFrame]],
    run_notes: Dict[str, str],
    smoothing_factor: float = 1.0,
    display_notes: bool = True,  # Toggle to display notes in the legend
    save_path: str = None,  # Optional path to save the plot
) -> None:
    """Plot episode length and reward for every run with optional smoothing and saving."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create a colour map so that each run gets a unique colour
    cmap = plt.get_cmap("tab10")
    run_names = sorted(run_dfs.keys())

    # Define line thickness and legend font sizes
    line_thickness = 2
    legend_font_size = 10
    notes_font_size = 2

    for idx, run_name in enumerate(run_names):
        color = cmap(idx % cmap.N)
        run_data = run_dfs[run_name]
        notes = run_notes.get(run_name, "").strip()
        legend_name = f"{run_name}\n({notes})" if notes and display_notes else run_name

        # Episode length plot with legend entry
        if "rollout/ep_len_mean" in run_data:
            df = run_data["rollout/ep_len_mean"]
            if not df.empty:
                smoothed_series = (
                    smooth_data(df["rollout/ep_len_mean"], smoothing_factor)
                    if smoothing_factor < 1.0
                    else df["rollout/ep_len_mean"]
                )
                axes[0].plot(
                    df["Step"],
                    smoothed_series,
                    label=legend_name,
                    color=color,
                    linewidth=line_thickness,
                )

        # Episode reward plot without additional legend entry
        if "rollout/ep_rew_mean" in run_data:
            df = run_data["rollout/ep_rew_mean"]
            if not df.empty:
                smoothed_series = (
                    smooth_data(df["rollout/ep_rew_mean"], smoothing_factor)
                    if smoothing_factor < 1.0
                    else df["rollout/ep_rew_mean"]
                )
                axes[1].plot(
                    df["Step"],
                    smoothed_series,
                    label="_nolegend_",
                    color=color,
                    linewidth=line_thickness,
                )

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Episode Length")
    axes[0].set_title("Episode Length over Time")
    axes[0].grid(True)
    axes[0].legend(
        title="Run", fontsize=legend_font_size, title_fontsize=notes_font_size
    )

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Episode Reward")
    axes[1].set_title("Episode Reward over Time")
    axes[1].grid(True)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


def get_single_character_input(prompt: str) -> str:
    """Get a single character input from the user."""
    print(prompt, end="", flush=True)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print(char)  # Echo the character
    return char.lower()


def filter_runs(
    run_dfs: Dict[str, Dict[str, pd.DataFrame]],
    run_notes: Dict[str, str],  # Pass run_notes to display notes
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Filter runs based on user input."""
    print("Available runs:")
    sorted_run_names = sorted(run_dfs.keys())  # Sort runs by name
    selected_runs = {}

    for idx, run_name in enumerate(sorted_run_names):
        notes = run_notes.get(run_name, "").strip()
        while True:
            choice = get_single_character_input(
                f"Plot run {idx}: {run_name} - Notes: {notes}? (p for yes, x for no): "
            )
            if choice == "p":
                selected_runs[run_name] = run_dfs[run_name]
                break
            elif choice == "x":
                break
            else:
                print("Invalid input. Please press 'p' for yes or 'x' for no.")

    # Print notes for selected runs at the end
    print("\nSelected runs and their notes:")
    for run_name in selected_runs.keys():
        notes = run_notes.get(run_name, "").strip()
        print(f"{run_name} - Notes: {notes}")

    return selected_runs


# Specify the directory containing your TensorBoard logs
log_dir = "/home/ferdinand/masterthesis/mt_start/runssaved"  # Replace with your actual directory

# Tags to extract (episode length and reward)
tags = ["rollout/ep_len_mean", "rollout/ep_rew_mean"]

# ``dataframes`` maps ``run_name`` -> ``{tag: DataFrame}``
dataframes, run_notes = extract_data_from_tensorboard(log_dir, tags)

# # Preview the last few entries of each run to verify loading
# for run_name, run_dfs in dataframes.items():
#     for tag in tags:
#         if tag in run_dfs:
#             print(f"Last rows for {run_name} / {tag}:")
#             print(run_dfs[tag].tail())

filtered_run_dfs = filter_runs(dataframes, run_notes)

plot_tensorboard_data(
    filtered_run_dfs,
    run_notes,
    smoothing_factor=0.4,
    display_notes=False,
    save_path="/home/ferdinand/masterthesis/mt_start/plots/tensorboard_plot.png",
)
