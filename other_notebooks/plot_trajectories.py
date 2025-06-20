import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np


def parse_urdf_and_plot(urdf_path, trajectory_data_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    walls = []

    # Each visual element of the URDF describes one wall.  We read the box size
    # from the ``size`` attribute and the position from the ``origin`` tag.  The
    # original version of this script expected nested ``size`` and ``color``
    # elements which do not exist in the provided URDF.  Because of that it
    # always fell back to default values and the walls were drawn at the wrong
    # position.  The code below correctly handles the URDF structure where the
    # attributes are stored on the elements themselves.
    for visual in root.findall(".//visual"):
        # Default color
        rgba = (1.0, 1.0, 1.0, 1.0)

        material = visual.find("material")
        if material is not None:
            color_el = material.find("color")
            if color_el is not None and color_el.get("rgba"):
                try:
                    rgba = tuple(map(float, color_el.get("rgba").split()))
                except Exception:
                    rgba = (1.0, 1.0, 1.0, 1.0)

        geometry = visual.find("geometry")
        if geometry is None:
            continue
        box = geometry.find("box")
        if box is None or not box.get("size"):
            continue

        size_values = tuple(map(float, box.get("size").split()))

        origin = visual.find("origin")
        if origin is not None:
            xyz = tuple(map(float, origin.get("xyz", "0 0 0").split()))
        else:
            xyz = (0.0, 0.0, 0.0)

        # Determine orientation (horizontal or vertical) from the box
        if size_values[0] >= size_values[1]:
            # Horizontal wall (longer in x-direction)
            x1 = xyz[0] - size_values[0] / 2
            x2 = xyz[0] + size_values[0] / 2
            y = xyz[1]
            thickness = size_values[1]  # Thickness is the shorter dimension
            walls.append(
                {"x1": x1, "x2": x2, "y": y, "color": rgba, "thickness": thickness}
            )
        else:
            # Vertical wall (longer in y-direction)
            x = xyz[0]
            y1 = xyz[1] - size_values[1] / 2
            y2 = xyz[1] + size_values[1] / 2
            thickness = size_values[0]  # Thickness is the shorter dimension
            walls.append(
                {"x1": x, "y1": y1, "y2": y2, "color": rgba, "thickness": thickness}
            )

    # Load the trajectory data
    all_trajs = np.load(trajectory_data_path, allow_pickle=True)

    # Plot the walls and trajectories
    fig, ax = plt.subplots()

    # Plot each wall as a line with its color and thickness
    for wall in walls:
        if "y" in wall:  # Horizontal wall
            ax.plot(
                [wall["x1"], wall["x2"]],
                [wall["y"], wall["y"]],
                color=wall["color"],
                linewidth=wall["thickness"] * 100,
            )
        elif "x1" in wall and "y1" in wall:  # Vertical wall
            ax.plot(
                [wall["x1"], wall["x1"]],
                [wall["y1"], wall["y2"]],
                color=wall["color"],
                linewidth=wall["thickness"] * 100,
            )

    # Plot all trajectories
    for ep_traj in all_trajs:
        ep_traj = np.array(ep_traj)
        ax.plot(ep_traj[:, 0], ep_traj[:, 1])

    # Formatting the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Trajectories of run")
    ax.grid(True)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.5, 6.5)
    ax.set_aspect("equal", "box")  # Equal scaling for both axes

    # Save and show the plot
    plt.savefig(f"{trajectory_data_path.rsplit('/', 1)[0]}/trajectories.png", dpi=300)
    plt.show()


# Example file locations relative to the repository root
urdf_file = "urdf/maze_colored.urdf"
trajectory_file = "runs/sac_run_20250619-140215/trajectories.npy"
parse_urdf_and_plot(urdf_file, trajectory_file)
