"""
A* pathfinding algorithm for the Pirate Intelligent Agent.

This module provides a deterministic search algorithm for finding an
optimal path through the treasure maze. It demonstrates the use of
priority queues, sets, dictionaries, heuristic search, and path
reconstruction.
"""

from dataclasses import dataclass
import heapq
from typing import Optional

import numpy as np


Position = tuple[int, int]


@dataclass
class SearchResult:
    """Stores the results and performance metrics of an A* search."""

    path: list[Position]
    path_length: int
    nodes_explored: int
    total_cost: float
    found: bool


class AStarPathfinder:
    """Find an optimal path through a maze using the A* algorithm."""

    def __init__(self, maze: np.ndarray) -> None:
        """Initialize the pathfinder with a validated maze."""

        self._validate_maze(maze)

        self.maze = np.array(
            maze,
            dtype=float,
            copy=True,
        )

        self.nrows, self.ncols = self.maze.shape

    @staticmethod
    def _validate_maze(maze: np.ndarray) -> None:
        """Validate the maze supplied to the search algorithm."""

        if not isinstance(maze, np.ndarray):
            raise TypeError(
                "maze must be provided as a NumPy array."
            )

        if maze.ndim != 2:
            raise ValueError(
                "maze must be a two-dimensional array."
            )

        if maze.size == 0:
            raise ValueError(
                "maze cannot be empty."
            )

        if not np.all(np.isin(maze, [0, 1])):
            raise ValueError(
                "maze cells must contain only 0 for blocked cells "
                "or 1 for open cells."
            )

    def _validate_position(
        self,
        position: Position,
        name: str,
    ) -> None:
        """Validate a start or goal position."""

        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not all(
                isinstance(value, int)
                for value in position
            )
        ):
            raise TypeError(
                f"{name} must be a tuple containing two integers."
            )

        row, col = position

        if not (
            0 <= row < self.nrows
            and 0 <= col < self.ncols
        ):
            raise ValueError(
                f"{name} must be within the maze boundaries."
            )

        if self.maze[row, col] != 1.0:
            raise ValueError(
                f"{name} must be located on an open maze cell."
            )

    @staticmethod
    def heuristic(
        position: Position,
        goal: Position,
    ) -> int:
        """Calculate Manhattan distance from a position to the goal."""

        return (
            abs(position[0] - goal[0])
            + abs(position[1] - goal[1])
        )

    def neighbors(
        self,
        position: Position,
    ) -> list[Position]:
        """Return all valid neighboring cells."""

        row, col = position

        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        valid_neighbors = []

        for neighbor_row, neighbor_col in candidates:
            if (
                0 <= neighbor_row < self.nrows
                and 0 <= neighbor_col < self.ncols
                and self.maze[
                    neighbor_row,
                    neighbor_col,
                ] == 1.0
            ):
                valid_neighbors.append(
                    (neighbor_row, neighbor_col)
                )

        return valid_neighbors

    @staticmethod
    def _reconstruct_path(
        came_from: dict[Position, Position],
        current: Position,
    ) -> list[Position]:
        """Reconstruct the completed path from goal to start."""

        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        return path

    def search(
        self,
        start: Position,
        goal: Optional[Position] = None,
    ) -> SearchResult:
        """Find an optimal path from the start position to the goal."""

        if goal is None:
            goal = (
                self.nrows - 1,
                self.ncols - 1,
            )

        self._validate_position(start, "start")
        self._validate_position(goal, "goal")

        if start == goal:
            return SearchResult(
                path=[start],
                path_length=0,
                nodes_explored=0,
                total_cost=0.0,
                found=True,
            )

        frontier: list[
            tuple[float, int, Position]
        ] = []

        counter = 0

        heapq.heappush(
            frontier,
            (
                float(self.heuristic(start, goal)),
                counter,
                start,
            ),
        )

        came_from: dict[
            Position,
            Position,
        ] = {}

        cost_so_far: dict[
            Position,
            float,
        ] = {
            start: 0.0
        }

        explored: set[Position] = set()

        while frontier:
            _, _, current = heapq.heappop(frontier)

            if current in explored:
                continue

            explored.add(current)

            if current == goal:
                path = self._reconstruct_path(
                    came_from,
                    current,
                )

                return SearchResult(
                    path=path,
                    path_length=len(path) - 1,
                    nodes_explored=len(explored),
                    total_cost=cost_so_far[current],
                    found=True,
                )

            for neighbor in self.neighbors(current):
                new_cost = (
                    cost_so_far[current]
                    + 1.0
                )

                if (
                    neighbor not in cost_so_far
                    or new_cost
                    < cost_so_far[neighbor]
                ):
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current

                    priority = (
                        new_cost
                        + self.heuristic(
                            neighbor,
                            goal,
                        )
                    )

                    counter += 1

                    heapq.heappush(
                        frontier,
                        (
                            priority,
                            counter,
                            neighbor,
                        ),
                    )

        return SearchResult(
            path=[],
            path_length=0,
            nodes_explored=len(explored),
            total_cost=float("inf"),
            found=False,
        )