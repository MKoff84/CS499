# Enhanced Pirate Intelligent Agent

## CS 499 Computer Science Capstone
**Student:** Matthew Biletnikoff  
**Enhancements:** Software Design and Engineering | Algorithms and Data Structures

## Project Overview

The Pirate Intelligent Agent was originally developed during CS 370:
Current/Emerging Trends in Computer Science. The project uses deep
Q-learning to train an intelligent agent to navigate a maze and reach
a treasure location.

For CS 499, the original project has been progressively enhanced across
multiple computer science categories. The Software Design and Engineering
enhancement refactored the original notebook-centered project into a
modular, maintainable, validated, and testable application.

The Algorithms and Data Structures enhancement extends that architecture
with deterministic pathfinding algorithms. A* search was implemented as
a heuristic-guided approach to maze navigation, while breadth-first search
(BFS) was implemented as an uninformed baseline. The two algorithms allow
the project to demonstrate algorithm selection, data-structure choices,
optimal pathfinding, and design trade-offs alongside the existing
reinforcement-learning solution.

## Software Design and Engineering Enhancement

The original project relied heavily on a Jupyter Notebook and combined
several responsibilities within the training workflow. The enhanced
version separates these responsibilities into independent modules.

The software design enhancement includes:

- Centralized and validated training configuration
- Separate maze environment
- Encapsulated deep Q-learning agent
- Dedicated replay-memory component
- Separate training coordinator
- Structured training metrics
- Reproducible random-number generation
- Input validation and exception handling
- Explicit action enumeration
- Automated unit testing
- Standalone application entry point
- Improved documentation and type hints

The enhancement also corrects an invalid-action state-handling issue
identified during the review of the original maze environment.

## Algorithms and Data Structures Enhancement

The second enhancement expands the project by introducing deterministic
search algorithms that solve the same maze-navigation problem from a
different computational perspective.

### A* Pathfinding

A* search was implemented to find an optimal path between a starting
position and the treasure. The algorithm evaluates candidate states using:

`f(n) = g(n) + h(n)`

where `g(n)` represents the known path cost from the starting position
and `h(n)` represents the estimated remaining cost to the goal. Manhattan
distance is used as the heuristic because movement in the maze is limited
to horizontal and vertical directions.

The A* implementation uses several data structures for specific purposes:

- A min-heap priority queue manages the search frontier.
- A set provides efficient tracking of explored maze positions.
- Dictionaries store accumulated path costs and predecessor relationships.
- Tuples represent immutable row-and-column coordinates.
- A list stores the reconstructed final path.

The implementation also validates start and goal positions and handles
unreachable goals without returning an invalid path.

### Breadth-First Search Baseline

Breadth-first search was implemented as an uninformed search baseline.
BFS uses a FIFO queue to explore neighboring states level by level. In
the unit-cost maze environment, BFS can also find a shortest path, but
it does not use information about the direction of the goal when deciding
which state to explore next.

Comparing BFS with A* demonstrates the trade-off between uninformed and
heuristic-guided search. A* uses the Manhattan-distance heuristic to
prioritize states that appear more promising, while BFS systematically
explores the search space according to distance from the starting state.

In the comparison maze used during testing, both algorithms found an
optimal path with a path length and total cost of 14. A* explored 25
nodes, while BFS explored 26. This result demonstrates the behavior of
the algorithms on the tested maze and is not intended to imply that A*
will always explore fewer nodes than BFS.

## Testing and Validation

The project includes automated testing for both the existing software
components and the Algorithms and Data Structures enhancement.

Testing covers:

- Training configuration validation
- Maze initialization and movement
- Invalid maze values and actions
- Replay-memory capacity and sampling
- Neural-network prediction and input validation
- A* shortest-path behavior
- BFS shortest-path behavior
- Equal-cost comparison between A* and BFS
- Blocked start and goal positions
- Out-of-bounds positions
- Unreachable goals

The complete automated test suite currently contains 20 tests, all of
which pass.

The enhanced deep Q-learning application also continues to execute
successfully after the Algorithms and Data Structures enhancement,
demonstrating that the new functionality did not break the existing
software architecture.

## Project Structure

```text
PirateIntelligentAgent_Milestone3/
├── agent/
│   ├── __init__.py
│   ├── pirate_agent.py
│   └── replay_memory.py
├── algorithms/
│   ├── __init__.py
│   ├── astar.py
│   └── bfs.py
├── environment/
│   ├── __init__.py
│   └── treasure_maze.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── tests/
│   ├── __init__.py
│   ├── test_algorithms.py
│   └── test_core.py
├── config.py
├── main.py
├── test_enhancement.ipynb
└── README.md