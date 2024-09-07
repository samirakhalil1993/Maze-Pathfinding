# Maze Pathfinding Algorithms

This project implements three classic pathfinding algorithms (BFS, DFS, and A*) for solving mazes. The project visualizes the maze and highlights the path found by each algorithm, and also compares the execution times of the different approaches.

## Algorithms Used
- **Breadth-First Search (BFS)**: Explores all nodes at the current level before moving to the next. Guarantees the shortest path in an unweighted grid.
- **Depth-First Search (DFS)**: Explores as deeply as possible along a branch before backtracking. May not find the shortest path.
- **A* Search**: Uses a heuristic (Manhattan distance) to guide the search and find the optimal path more efficiently than BFS.

## Features
- Load a maze from an image file and convert it to a binary format.
- Solve the maze using BFS, DFS, and A* algorithms.
- Visualize the maze with the path found by each algorithm.
- Compare execution times between BFS, DFS, and A*.

## Requirements
To run this project, you need Python 3.x and the following libraries:
- `Pillow` (for image processing)
- `NumPy` (for array manipulations)
- `Matplotlib` (for visualization)
- `time` (to track execution times)

You can install the required libraries with:

```pip install Pillow numpy matplotlib```

## License

This project is licensed under the [MIT License](LICENSE).





