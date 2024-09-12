import numpy as np
import matplotlib.pyplot as plt





# Define the maze
maze = np.array([
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],  
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],  
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],  
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],  
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],  
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
])


# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Use imshow to display the maze, setting the color map to distinguish free space and walls
# Free space (0) will be white and walls (1) will be black
ax.imshow(maze, cmap='gray_r')

# Mark the start point (Red) and goal point (Blue)
start = (0, 1)  # Starting point
goal = (9, 18)  # Goal point

# Add markers for start (red) and goal (blue)
ax.scatter(start[1], start[0], color='red', s=100, label="Start")
ax.scatter(goal[1], goal[0], color='blue', s=100, label="Goal")

# Add gridlines for better visibility of the maze structure
ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

# Hide the major ticks
ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)

# Show the legend
ax.legend(loc="upper left")

# Display the plot
plt.show()