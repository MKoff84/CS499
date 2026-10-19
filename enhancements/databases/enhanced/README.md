# Enhanced Pirate Intelligent Agent

## CS 499 Computer Science Capstone
**Student:** Matthew Biletnikoff  
**Enhancements:** Software Design and Engineering | Algorithms and Data Structures | Databases

## Project Overview

The Pirate Intelligent Agent was originally developed during CS 370:
Current/Emerging Trends in Computer Science. The project uses deep
Q-learning to train an intelligent agent to navigate a maze and reach
a treasure location.

For CS 499, the original project has been progressively enhanced across
three computer science categories. The Software Design and Engineering
enhancement refactored the original notebook-centered project into a
modular, maintainable, validated, and testable application.

The Algorithms and Data Structures enhancement extends that architecture
with deterministic pathfinding algorithms. A* search was implemented as
a heuristic-guided approach to maze navigation, while breadth-first search
(BFS) was implemented as an uninformed baseline. The two algorithms allow
the project to demonstrate algorithm selection, data-structure choices,
optimal pathfinding, and design trade-offs alongside the existing
reinforcement-learning solution.

The Databases enhancement adds persistent experiment tracking using
SQLite. Deep Q-learning training metrics can now be stored after an
application run instead of existing only in memory. A relational structure
also stores A* and BFS results beneath a pathfinding-comparison experiment,
allowing results from different computational approaches to be retained
and retrieved for later analysis.

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

In the controlled comparison maze used by the automated algorithm tests,
both algorithms found an optimal path with a path length and total cost
of 14. A* explored 25 nodes, while BFS explored 26. This result
demonstrates the behavior of the algorithms on that test maze and is not
intended to imply that A* will always explore fewer nodes than BFS.

When the algorithms were later integrated with the complete original
8-by-8 CS 370 maze for the database enhancement, both found a shortest
path with a path length and total cost of 24. A* explored 50 nodes and
BFS explored 51. Execution times are also recorded by the application,
but the maze is small enough that sub-millisecond timing differences
should not be interpreted as a general performance advantage for either
algorithm.

## Databases Enhancement

The third enhancement introduces a SQLite database layer for persistent
experiment tracking. Before this enhancement, training and algorithm
results were displayed during execution but were not retained after the
program ended. The database allows experiment results to be stored,
retrieved, and compared across application runs.

SQLite was selected because it provides relational database functionality
while remaining lightweight and portable. Python includes SQLite support,
so the enhanced artifact can demonstrate database design and SQL without
requiring a separate database server.

### Relational Design

The database uses two related tables:

- `experiment_runs` stores information about an experiment, including
  the algorithm or experiment type, maze dimensions, training metrics,
  random seed, and execution duration when applicable.
- `algorithm_results` stores deterministic pathfinding results, including
  algorithm name, whether a path was found, path length, nodes explored,
  total cost, and measured execution time.

Each experiment is identified by a primary key. The `algorithm_results`
table uses a foreign key to associate one or more algorithm results with
an experiment. This creates a one-to-many relationship that allows a
single pathfinding-comparison experiment to contain both A* and BFS
results.

The schema also uses `CHECK` constraints to protect valid ranges,
indexes to support retrieval, foreign-key enforcement, and
`ON DELETE CASCADE` to maintain referential integrity when a parent
experiment is removed.

### Database Access and Data Integrity

Database operations are encapsulated in the `ExperimentDatabase` class
rather than being placed directly throughout the application. SQL values
are supplied using parameterized queries instead of being concatenated
into SQL statements, reducing SQL-injection risk and separating SQL
commands from application data.

Application-level validation rejects invalid values before they reach
the database where appropriate, while database constraints provide an
additional integrity layer. SQLite foreign-key enforcement is enabled
for each connection.

Database connections are managed through a context manager. Successful
operations are committed, failed operations are rolled back, and
connections are explicitly closed. This prevents database resources
from remaining open after an operation completes.

### Experiment Persistence

The application entry point now records actual deep Q-learning results
after training. Stored values include:

- Maze dimensions
- Epochs completed
- Wins and losses
- Win rate
- Average training loss
- Final epsilon
- Random seed
- Measured training duration

A separate `Pathfinding Comparison` experiment stores the results of
actual A* and BFS searches through related `algorithm_results` records.

This design keeps reinforcement-learning metrics and deterministic
pathfinding metrics logically separate while allowing both types of
experiments to be represented within the same database architecture.

## Testing and Validation

The project includes automated testing across all three enhancement
categories.

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
- Database experiment insertion and retrieval
- One-to-many algorithm-result relationships
- Empty algorithm-name validation
- Invalid win-rate validation
- Foreign-key enforcement
- Cascade deletion of related records

The complete automated test suite contains 26 tests, all of which pass.

The database tests use temporary SQLite databases so that each test runs
in isolation and does not modify the application's persistent experiment
database. The complete test suite also verifies that the database
enhancement did not break functionality introduced during the earlier
Software Design and Engineering or Algorithms and Data Structures
enhancements.

## Project Structure

```text
PirateIntelligentAgent_Milestone4/
├── agent/
│   ├── __init__.py
│   ├── pirate_agent.py
│   └── replay_memory.py
├── algorithms/
│   ├── __init__.py
│   ├── astar.py
│   └── bfs.py
├── database/
│   ├── __init__.py
│   └── experiment_database.py
├── environment/
│   ├── __init__.py
│   └── treasure_maze.py
├── training/
│   ├── __init__.py
│   └── trainer.py
├── tests/
│   ├── __init__.py
│   ├── test_algorithms.py
│   ├── test_core.py
│   └── test_database.py
├── config.py
├── main.py
├── requirements.txt
├── test_enhancement.ipynb
└── README.md
```

## Running the Application

From the project root, run:

```powershell
python main.py
```

The application trains the deep Q-learning agent, records the resulting
training metrics, executes the A* and BFS pathfinding comparison, and
stores the experiment results in `pirate_experiments.db`.

## Running the Automated Tests

From the project root, run:

```powershell
python -m unittest discover -s tests -v
```

The current test suite contains 26 automated tests.