import numpy as np
import random

# Define maze dimensions
height = 21  # Must be an odd number
width = 21  # Must be an odd number
# Create an empty maze
maze = np.zeros((height, width), dtype=int)  # 0 is path, 1 is wall

# Directions (Up, Down, Right, Left)
directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

def is_valid_move(x, y):
    # Ensure we're within bounds and the space is unvisited
    return 0 <= x < height and 0 <= y < width and maze[x][y] == 0

def carve(x, y):
    # Mark the current position as a path (1 is path)
    maze[x][y] = 1
    random.shuffle(directions)

    # Try to carve paths in all directions
    for dx, dy in directions:
        nx, ny = x + dx * 2, y + dy * 2
        if is_valid_move(nx, ny):
            maze[x + dx][y + dy] = 1
            carve(nx, ny)

# Start carving from a random position (odd indices)
start_x = random.randint(0, (height - 1) // 2) * 2 + 1
start_y = random.randint(0, (width - 1) // 2) * 2 + 1
carve(start_x, start_y)

# Ensure the start is open (just in case)
maze[start_x][start_y] = 0

# Randomly select the exit (goal) position, but make sure it's an open space
goal_x, goal_y = random.randint(1, height - 2), random.randint(1, width - 2)

# Ensure the goal is not on the start position and is a valid open space
while maze[goal_x][goal_y] == 1 or (goal_x == start_x and goal_y == start_y):
    goal_x, goal_y = random.randint(1, height - 2), random.randint(1, width - 2)

# Mark the goal as an open space (exit)
maze[goal_x][goal_y] = 0  # Exit point

import matplotlib.pyplot as plt

# Plot the maze
plt.figure(figsize=(10, 10))
plt.imshow(maze, cmap='binary')
plt.axis('off')  # Turn off the axes

# Mark the goal with a red square
plt.scatter(goal_y, goal_x, c='red', s=100, label='Goal')

plt.show()