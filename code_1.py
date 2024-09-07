
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
#samir akhalil
# Load the maze image
image_path = '/home/samir-akhalil/Pictures/Screenshots/Maze.png'  # Make sure the path is correct
img = Image.open(image_path).convert('L')  # Convert the image to grayscale

# Convert the grayscale image to a NumPy array
maze_array = np.array(img)

# Apply a threshold to convert black pixels (paths) to 0 and white pixels (walls) to 1
threshold = 128  # Adjust based on the image contrast
maze_binary = (maze_array > threshold).astype(int)  # Now, 0 = black (free path), 1 = white (wall)


# Visualize the binary maze (with 0 as black, 1 as white)
plt.imshow(maze_binary, cmap='gray')
plt.show()

# Save the processed binary maze image
output_path = '/home/samir-akhalil/Desktop/python/maze_binary_output.png'
plt.imsave(output_path, maze_binary, cmap='gray')

print(f"Maze saved as 'maze_binary_output.png' in the specified directory.")


maze = [
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],  # Row 1: Starting point (Red node at (0, 1))
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],  # Row 2
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # Row 3
    [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # Row 4
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],  # Row 5
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],  # Row 6
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # Row 7
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],  # Row 8
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],  # Row 9
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]   # Row 10: Goal point (Blue node at (9, 18)
]



def bfs(maze, start, goal):
    """
    Finds the shortest path from the start to the goal in a maze using Breadth-First Search (BFS).

    Args:
        maze (list[list[int]]): 2D list representing the maze grid. 0 indicates a free space, 1 indicates an obstacle.
        start (tuple[int, int]): The starting coordinates (x, y) in the maze.
        goal (tuple[int, int]): The target coordinates (x, y) in the maze.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the path from start to goal, or None if no path exists.
    """
    
    rows, cols = len(maze), len(maze[0])
    
    # Ensure start and goal are within bounds and not blocked by obstacles
    if not (0 <= start[0] < rows and 0 <= start[1] < cols) or maze[start[0]][start[1]] == 1:
        print(f"Invalid start point: {start}")
        return None
    
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols) or maze[goal[0]][goal[1]] == 1:
        print(f"Invalid goal point: {goal}")
        return None
    
    # Initialize BFS structures
    queue = deque([[start]])  # Queue to explore paths
    visited = set([start])    # Set to track visited nodes

    while queue:
        path = queue.popleft()  # Dequeue the next path
        x, y = path[-1]        # Current position (last cell in path)
        
        # Check if we have reached the goal
        if (x, y) == goal:
            return path
        
        # Explore the four possible directions (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            
            # Ensure the neighbor cell is within bounds, walkable, and not visited
            if (0 <= nx < rows and 0 <= ny < cols and 
                maze[nx][ny] == 0 and (nx, ny) not in visited):
                
                visited.add((nx, ny))           # Mark as visited
                queue.append(path + [(nx, ny)])  # Enqueue the new path
    
    # No path found if we exhaust the queue
    print("No path found")
    return None


def dfs(maze, start, goal):
    """
    Finds a path from the start to the goal in a maze using Depth-First Search (DFS).

    Args:
        maze (list[list[int]]): 2D list representing the maze grid. 0 indicates a free space, 1 indicates an obstacle.
        start (tuple[int, int]): The starting coordinates (x, y) in the maze.
        goal (tuple[int, int]): The target coordinates (x, y) in the maze.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the path from start to goal, or None if no path exists.
    """
    
    rows, cols = len(maze), len(maze[0])
    
    # Initialize DFS structures
    stack = [[start]]  # Stack to explore paths
    visited = set([start])  # Set to track visited nodes

    while stack:
        path = stack.pop()  # Pop the last path from the stack
        x, y = path[-1]    # Get the current position (last cell in path)
        
        # Check if we have reached the goal
        if (x, y) == goal:
            return path
        
        # Explore the four possible directions (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            
            # Ensure the neighbor cell is within bounds, walkable, and not visited
            if (0 <= nx < rows and 0 <= ny < cols and 
                maze[nx][ny] == 0 and (nx, ny) not in visited):
                
                visited.add((nx, ny))           # Mark as visited
                stack.append(path + [(nx, ny)])  # Push the new path onto the stack
    
    # No path found if we exhaust the stack
    print("No path found")
    return None

import heapq

def manhattan_distance(a, b):
    """
    Computes the Manhattan distance between two points.

    Args:
        a (tuple[int, int]): Coordinates of the first point (x1, y1).
        b (tuple[int, int]): Coordinates of the second point (x2, y2).

    Returns:
        int: The Manhattan distance between points a and b.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal):
    """
    Finds the shortest path from start to goal in a maze using the A* algorithm.

    Args:
        maze (list[list[int]]): 2D list representing the maze grid. 0 indicates a free space, 1 indicates an obstacle.
        start (tuple[int, int]): The starting coordinates (x, y) in the maze.
        goal (tuple[int, int]): The target coordinates (x, y) in the maze.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the path from start to goal, or None if no path exists.
    """
    
    rows, cols = len(maze), len(maze[0])
    
    # Priority queue for A* search, initialized with (estimated total cost, cost so far, current node, path)
    pq = [(0 + manhattan_distance(start, goal), 0, start, [])]
    visited = set()
    
    while pq:
        # Get the node with the lowest estimated total cost
        est_total_cost, cost, (x, y), path = heapq.heappop(pq)
        
        # Check if we have reached the goal
        if (x, y) == goal:
            return path + [(x, y)]
        
        # Skip this node if it has already been visited
        if (x, y) in visited:
            continue
        
        # Mark the node as visited and update the path
        visited.add((x, y))
        path = path + [(x, y)]
        
        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            
            # Check if the neighbor is within bounds and is walkable
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                new_cost = cost + 1
                est_total_cost = new_cost + manhattan_distance((nx, ny), goal)
                heapq.heappush(pq, (est_total_cost, new_cost, (nx, ny), path))
    
    # If the priority queue is exhausted and goal is not found
    print("No path found")
    return None



def print_maze_with_path(maze, path):
    # Create a copy of the maze as a list of strings
    maze_copy = np.array(maze, dtype=str)
    
    # Replace the path coordinates with '*'
    for (x, y) in path:
        maze_copy[x][y] = '*'
    
    # Print the maze, replacing 1 with '1', 0 with '0', and showing '*' where the path is
    for row in maze_copy:
        print("".join(row))



def run_algorithm(algorithm_name, algorithm_func, maze, start_point, goal_point):
    print(f"Running {algorithm_name}...")
    start_time = time.time()
    path = algorithm_func(maze, start_point, goal_point)
    execution_time = time.time() - start_time
    
    if path:
        print(f"{algorithm_name} Path Length: {len(path)}")
        print(f"{algorithm_name} Time: {execution_time:.4f} seconds")
        print_maze_with_path(maze, path)
        print()
        print("--------------------------------------------")
        print()
    else:
        print(f"No path found using {algorithm_name}.")
    
    return execution_time

def run_algorithm_comparison(maze, start_point, goal_point):
    # Run BFS
    bfs_time = run_algorithm("BFS", bfs, maze, start_point, goal_point)
    
    # Run DFS
    dfs_time = run_algorithm("DFS", dfs, maze, start_point, goal_point)
    
    # Run A* Search
    a_star_time = run_algorithm("A*", a_star, maze, start_point, goal_point)
    
    # Compare times
    print("\nComparison of Execution Times:")
    print(f"BFS Time: {bfs_time:.5f} seconds")
    print(f"DFS Time: {dfs_time:.5f} seconds")
    print(f"A*  Time : {a_star_time:.5f} seconds")

    # Determine the fastest algorithm
    if bfs_time < dfs_time and bfs_time < a_star_time:
        print("BFS is the fastest algorithm.")
    elif dfs_time < bfs_time and dfs_time < a_star_time:
        print("DFS is the fastest algorithm.")
    else:
        print("A* Search is the fastest algorithm.")


print("samiiiiiiiiiiiiiiiiiiiiiiir")
# Main program flow
start_point = (0, 1)  # Adjust these as per your maze's start point
goal_point = (9, 18)  # Adjust these as per your maze's goal point


# Run the algorithm comparison
run_algorithm_comparison(maze, start_point, goal_point)