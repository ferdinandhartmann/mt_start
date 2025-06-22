import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_data_from_tensorboard(log_dir, tags):
    """
    Extract data from TensorBoard event files in the log_dir for specified tags,
    including traversing subdirectories.
    """
    # Dictionary to store data for the tags
    data = {tag: [] for tag in tags}

    # Traverse through the subdirectories of the log_dir
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_file_path = os.path.join(root, file)

                # Initialize EventAccumulator for the event file
                ea = event_accumulator.EventAccumulator(event_file_path)
                ea.Reload()  # Load the events from the file

                # Print out available tags for this event file
                print(
                    f"Available tags in {event_file_path}: {ea.Tags()}"
                )  # Notice the use of Tags() instead of tags()

                # Extract the data for the specified tags
                for tag in tags:
                    if (
                        tag in ea.Tags()["scalars"]
                    ):  # Check if the tag exists in the scalar data
                        scalar_data = ea.Scalars(tag)
                        for scalar in scalar_data:
                            data[tag].append(
                                (scalar.step, scalar.value)
                            )  # Corrected access to step and value

    # Convert data to DataFrame for easier plotting
    dfs = {}
    for tag in tags:
        dfs[tag] = pd.DataFrame(data[tag], columns=["Step", tag])

    return dfs


# Specify the directory containing your TensorBoard logs
log_dir = "/home/ferdinand/masterthesis/mt_start/runs/tensorboards"  # Replace with your actual directory

# Tags to extract (episode length and reward)
tags = ["rollout/ep_len_mean", "rollout/ep_rew_mean"]
dataframes = extract_data_from_tensorboard(log_dir, tags)

# Preview data
if "rollout/ep_len_mean" in dataframes:
    print(dataframes["rollout/ep_len_mean"].tail())
else:
    print("No data found for rollout/ep_len_mean")

if "rollout/ep_rew_mean" in dataframes:
    print(dataframes["rollout/ep_rew_mean"].tail())
else:
    print("No data found for rollout/ep_rew_mean")


# Plot the data
def plot_tensorboard_data(dfs):
    """
    Plot episode length and reward from TensorBoard logs.
    """
    plt.figure(figsize=(14, 6))

    # Plot episode length
    if "rollout/ep_len_mean" in dfs:
        plt.subplot(1, 2, 1)
        plt.plot(
            dfs["rollout/ep_len_mean"]["Step"],
            dfs["rollout/ep_len_mean"]["rollout/ep_len_mean"],
            label="Episode Length",
            color="blue",
        )
        plt.xlabel("Step")
        plt.ylabel("Episode Length")
        plt.title("Episode Length over Time")
        plt.grid(True)

    # Plot episode reward
    if "rollout/ep_rew_mean" in dfs:
        plt.subplot(1, 2, 2)
        plt.plot(
            dfs["rollout/ep_rew_mean"]["Step"],
            dfs["rollout/ep_rew_mean"]["rollout/ep_rew_mean"],
            label="Episode Reward",
            color="green",
        )
        plt.xlabel("Step")
        plt.ylabel("Episode Reward")
        plt.title("Episode Reward over Time")
        plt.grid(True)

    plt.tight_layout()
    plt.show()


# Call the function to plot the data
plot_tensorboard_data(dataframes)
