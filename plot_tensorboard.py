import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_data_from_tensorboard(log_dir: str, tags: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Extract scalar data for each run and tag found in ``log_dir``.

    The directory may contain multiple runs (subdirectories).  This function
    keeps the data of each run separate so that plotting does not connect the end
    of one run with the start of the next.
    """

    # Nested dictionary ``data[run][tag]`` where the values are lists of
    # ``(step, value)`` tuples.  ``run`` is the relative directory name that
    # contains the TensorBoard event file.
    data: Dict[str, Dict[str, List[tuple]]] = {}

    # Traverse ``log_dir`` recursively and read all TensorBoard event files.
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file_path = os.path.join(root, file)

                # Use the directory containing the event file as the run name.
                run_name = os.path.relpath(root, log_dir)
                if run_name not in data:
                    data[run_name] = {tag: [] for tag in tags}

                # Load events from the file.
                ea = event_accumulator.EventAccumulator(event_file_path)
                ea.Reload()

                print(f"Available tags in {event_file_path}: {ea.Tags()}")

                # Extract the requested scalar data.
                for tag in tags:
                    if tag in ea.Tags()["scalars"]:
                        scalar_data = ea.Scalars(tag)
                        for scalar in scalar_data:
                            data[run_name][tag].append((scalar.step, scalar.value))

    # Convert lists into sorted DataFrames for easier plotting.
    run_dfs: Dict[str, Dict[str, pd.DataFrame]] = {}
    for run_name, tag_dict in data.items():
        run_dfs[run_name] = {}
        for tag, values in tag_dict.items():
            df = pd.DataFrame(values, columns=["Step", tag])
            run_dfs[run_name][tag] = df.sort_values("Step")

    return run_dfs


# Specify the directory containing your TensorBoard logs
log_dir = "/home/ferdinand/masterthesis/mt_start/runs/tensorboards"  # Replace with your actual directory

# Tags to extract (episode length and reward)
tags = ["rollout/ep_len_mean", "rollout/ep_rew_mean"]

# ``dataframes`` maps ``run_name`` -> ``{tag: DataFrame}``
dataframes = extract_data_from_tensorboard(log_dir, tags)

# Preview the last few entries of each run to verify loading
for run_name, run_dfs in dataframes.items():
    for tag in tags:
        if tag in run_dfs:
            print(f"Last rows for {run_name} / {tag}:")
            print(run_dfs[tag].tail())


# Plot the data
def plot_tensorboard_data(run_dfs: Dict[str, Dict[str, pd.DataFrame]]) -> None:
    """Plot episode length and reward for every run.

    The same ``run`` is plotted with the same colour across both subplots.  Each
    run's label appears only once in the legend of the left subplot.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create a colour map so that each run gets a unique colour
    cmap = plt.get_cmap("tab10")
    run_names = sorted(run_dfs.keys())

    for idx, run_name in enumerate(run_names):
        color = cmap(idx % cmap.N)
        run_data = run_dfs[run_name]

        # Episode length plot with legend entry
        if "rollout/ep_len_mean" in run_data:
            df = run_data["rollout/ep_len_mean"]
            axes[0].plot(
                df["Step"],
                df["rollout/ep_len_mean"],
                label=run_name,
                color=color,
            )

        # Episode reward plot without additional legend entry
        if "rollout/ep_rew_mean" in run_data:
            df = run_data["rollout/ep_rew_mean"]
            axes[1].plot(
                df["Step"],
                df["rollout/ep_rew_mean"],
                label="_nolegend_",
                color=color,
            )

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Episode Length")
    axes[0].set_title("Episode Length over Time")
    axes[0].grid(True)
    axes[0].legend(title="Run")

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Episode Reward")
    axes[1].set_title("Episode Reward over Time")
    axes[1].grid(True)

    fig.tight_layout()
    plt.show()


# Call the function to plot the data
plot_tensorboard_data(dataframes)
