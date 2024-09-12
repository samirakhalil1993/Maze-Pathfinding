import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import heapq

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

# Function to plot the maze
def plot_maze(maze, path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(maze, cmap='gray_r')

    if path:
        for (x, y) in path:
            ax.scatter(y, x, color='green', s=50)

    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True)
    ax.tick_params(which='major', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.show()

# Algorithm implementations (BFS, DFS, A*)
def bfs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == goal:
            return path
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(path + [(nx, ny)])
    return None

def dfs(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    stack = [[start]]
    visited = set([start])

    while stack:
        path = stack.pop()
        x, y = path[-1]
        if (x, y) == goal:
            return path
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                stack.append(path + [(nx, ny)])
    return None

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal):
    rows, cols = len(maze), len(maze[0])
    pq = [(0 + manhattan_distance(start, goal), 0, start, [])]
    visited = set()

    while pq:
        est_total_cost, cost, (x, y), path = heapq.heappop(pq)
        if (x, y) == goal:
            return path + [(x, y)]
        
        if (x, y) in visited:
            continue
        
        visited.add((x, y))
        path = path + [(x, y)]
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 0 and (nx, ny) not in visited:
                new_cost = cost + 1
                est_total_cost = new_cost + manhattan_distance((nx, ny), goal)
                heapq.heappush(pq, (est_total_cost, new_cost, (nx, ny), path))
    return None

# Running the algorithm and comparing times
def run_algorithm(algorithm_name, algorithm_func, maze, start_point, goal_point):
    start_time = time.time()
    path = algorithm_func(maze, start_point, goal_point)
    execution_time = time.time() - start_time
    
    if path:
        print(f"{algorithm_name} found a path with length {len(path)} in {execution_time:.4f} seconds.")
        plot_maze(maze, path)
    else:
        print(f"{algorithm_name} could not find a path.")
    
    return execution_time

def select_algorithm(available_algorithms, selected_algorithms):
    while True:
        print("Choose an algorithm to run:")
        for i, algo in enumerate(available_algorithms, 1):
            print(f"{i}. {algo}")
        choice = input("Enter the number of the algorithm you want to run: ")

        if choice.isdigit() and 1 <= int(choice) <= len(available_algorithms):
            selected_algorithm = available_algorithms[int(choice) - 1]
            if selected_algorithm not in selected_algorithms:
                selected_algorithms.append(selected_algorithm)  # Track the selected algorithm
                return selected_algorithm
            else:
                print("You have already chosen this algorithm. Please select another.")
        else:
            print("Invalid choice. Please choose again.")


def main():
    # Get user input for start and goal points
    start_point = (0, 1)  # Adjust these as per your maze's start point
    goal_point = (9, 18)  # Adjust these as per your maze's goal point

    # List of available algorithms
    algorithms = {
        "BFS": bfs,
        "DFS": dfs,
        "A*": a_star
    }

    selected_algorithms = []  # List to store selected algorithms
    algorithm_names = list(algorithms.keys())  # Available algorithm names
    
    times = {}  # Dictionary to store execution times
    
    # Allow the user to select 3 different algorithms
    for _ in range(3):
        selected_algorithm = select_algorithm(algorithm_names, selected_algorithms)  # Pass both arguments
        print(f"Selected Algorithm: {selected_algorithm}")

        # Run the selected algorithm, visualize the path, and show the time it takes
        algo_func = algorithms[selected_algorithm]
        times[selected_algorithm] = run_algorithm(selected_algorithm, algo_func, maze, start_point, goal_point)

    # After all 3 algorithms have run, compare execution times
    print("\nComparison of Execution Times:")
    for algo_name, exec_time in times.items():
        print(f"{algo_name} Time: {exec_time:.5f} seconds")

    # Find the fastest algorithm
    fastest_algorithm = min(times, key=times.get)
    print(f"\nThe fastest algorithm is {fastest_algorithm}.")

    
    


    fastest_algorithm = min(times, key=times.get)
    print(f"\nThe fastest algorithm is {fastest_algorithm}.")

# Run the program
if __name__ == "__main__":
    main()
