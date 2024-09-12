# Maze Pathfinding Algorithms

This project implements three classic pathfinding algorithms (BFS, DFS, and A*) for solving mazes. The project allows users to select algorithms, visualize the paths found by each algorithm, and compare the execution times.

## Algorithms Used:
- **Breadth-First Search (BFS)**: Explores all nodes at the current level before moving to the next. Guarantees the shortest path in an unweighted grid.
- **Depth-First Search (DFS)**: Explores as deeply as possible along a branch before backtracking. May not find the shortest path.
- **A* Search**: Uses a heuristic (Manhattan distance) to guide the search and find the optimal path more efficiently than BFS.

## Features:
- Allows the user to select different algorithms (BFS, DFS, A*) for solving a maze.
- Visualizes the maze with the path found by each algorithm.
- Compares execution times between BFS, DFS, and A*.
- Provides feedback if no path is found.

## How to Use:
1. When running the program, the user is prompted to select three different algorithms to solve the maze.
2. After each selection, the algorithm will be executed, the path will be visualized, and the execution time will be displayed.
3. After running all three algorithms, the program compares the execution times and displays the fastest algorithm.

## Requirements:
To run this project, you need Python 3.x and the following libraries:

- **NumPy** (for array manipulations)
- **Matplotlib** (for visualization)
- **time** (to track execution times)
- **collections.deque** (for BFS queue management)
- **heapq** (for A* priority queue management)

You can install the required libraries with:

```bash
pip install numpy matplotlib


## License

This project is licensed under the [MIT License](LICENSE).





