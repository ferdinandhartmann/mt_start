import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np


def parse_urdf_and_plot(urdf_path, trajectory_data_path):
    # Parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    walls = []

    # Loop through the URDF XML to find walls (both visual and collision)
    for link in root.findall(".//link"):
        for visual in link.findall("visual"):
            # Default color is white if color is not found
            rgba = (1.0, 1.0, 1.0, 1.0)

            # Get the color information
            material = visual.find("material")
            if material is not None:
                color = material.find("color")
                if color is not None:
                    print(f"Found color: {color.text}")
                    try:
                        rgba = tuple(map(float, color.text.split()))
                    except AttributeError:
                        rgba = (1.0, 1.0, 1.0, 1.0)  # Default color: white
                        print(f"Warning: Invalid color format for link '{link.get('name')}'")
            else:
                rgba = (1.0, 1.0, 1.0, 1.0)  # Default color: white
                print(f"Warning: No material found for link '{link.get('name')}'")

            # Get the geometry (box size)
            geometry = visual.find("geometry")
            if geometry is not None:
                box = geometry.find("box")
                if box is not None:
                    size = box.find("size")
                    if size is not None:
                        size_values = tuple(map(float, size.text.split()))
                    else:
                        size_values = (1.0, 1.0, 1.0)  # Default size: 1x1x1
                        print(f"Warning: No size found for box in link '{link.get('name')}'")

                    # Get the position (origin)
                    origin = visual.find("origin")
                    if origin is not None:
                        xyz = tuple(map(float, origin.get("xyz", "0 0 0").split()))
                        rpy = tuple(map(float, origin.get("rpy", "0 0 0").split()))

                        # Handle horizontal and vertical walls
                        if size_values[0] > size_values[1]:  # Horizontal wall
                            x1 = xyz[0] - size_values[0] / 2
                            x2 = xyz[0] + size_values[0] / 2
                            y = xyz[1]  # Constant y value for horizontal wall
                            walls.append({"x1": x1, "x2": x2, "y": y, "color": rgba})
                        else:  # Vertical wall
                            x = xyz[0]  # Constant x value for vertical wall
                            y1 = xyz[1] - size_values[1] / 2
                            y2 = xyz[1] + size_values[1] / 2
                            walls.append({"x1": x, "y1": y1, "y2": y2, "color": rgba})

    # ovverride Walls because it doesnt work with the current URDF

    # walls = [
    #     {"x1": -5, "x2": 5, "y": 5, "color": "black"},  # Top wall
    #     {"x1": -5, "x2": 5, "y": -5, "color": "black"},  # Bottom wall
    #     {"x1": -5, "x2": -5, "y1": -4, "y2": 4, "color": "black"},  # Left wall
    #     {"x1": 5, "x2": 5, "y1": -4, "y2": 4, "color": "black"},  # Right wall
    #     # Inner horizontal walls
    #     {"x1": -3, "x2": 3, "y": 3, "color": "black"},
    #     {"x1": -3, "x2": 3, "y": 1, "color": "black"},
    #     {"x1": -3, "x2": 3, "y": -1, "color": "black"},
    #     {"x1": -3, "x2": 3, "y": -3, "color": "black"},
    #     # Inner vertical walls
    #     {"x1": 2, "x2": 2, "y1": -2, "y2": 2, "color": "black"},
    #     {"x1": -2, "x2": -2, "y1": -2, "y2": 2, "color": "black"},
    # ]

    print(walls)

    # Load the trajectory data
    all_trajs = np.load(trajectory_data_path, allow_pickle=True)

    # Plot the walls and trajectories
    fig, ax = plt.subplots()

    # Plot each wall as a line with its color
    for wall in walls:
        if "y" in wall:  # Horizontal wall
            ax.plot(
                [wall["x1"], wall["x2"]],
                [wall["y"], wall["y"]],
                color=wall["color"],
                linewidth=5,
            )
        elif "x1" in wall and "y1" in wall:  # Vertical wall
            ax.plot(
                [wall["x1"], wall["x1"]],
                [wall["y1"], wall["y2"]],
                color=wall["color"],
                linewidth=5,
            )

    # Plot all trajectories
    for ep_traj in all_trajs:
        ep_traj = np.array(ep_traj)
        ax.plot(ep_traj[:, 0], ep_traj[:, 1])

    # Formatting the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("All Trajectories with Maze Walls")
    ax.grid(True)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal", "box")  # Equal scaling for both axes

    # Save and show the plot
    plt.savefig("maze_trajectories_with_walls.png", dpi=300)
    plt.show()


urdf_file = "/home/ferdinand/masterthesis/mt_start/urdf/maze_colored.urdf"  
trajectory_file = "/home/ferdinand/masterthesis/mt_start/runs/sac_run_20250619-140215/trajectories.npy"  
parse_urdf_and_plot(urdf_file, trajectory_file)
