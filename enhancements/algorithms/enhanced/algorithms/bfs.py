"""
Breadth-first search implementation for the Pirate Intelligent Agent.

This module provides an uninformed search algorithm that can be used as
a baseline for evaluating the A* pathfinding enhancement.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class BFSResult:
    """Stores the result and performance information from a BFS search."""

    found: bool
    path: list[tuple[int, int]]
    path_length: int
    nodes_explored: int
    total_cost: float


class BFSPathfinder:
    """Finds shortest paths through a binary maze using breadth-first search."""

    def __init__(self, maze: np.ndarray) -> None:
        """Initialize and validate the maze."""

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

        self.maze = np.array(
            maze,
            dtype=int,
            copy=True,
        )

        self.nrows, self.ncols = self.maze.shape

    def _validate_position(
        self,
        position: tuple[int, int],
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

        if self.maze[row, col] != 1:
            raise ValueError(
                f"{name} must be located on an open maze cell."
            )

    def _neighbors(
        self,
        position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Return all valid neighboring cells."""

        row, col = position

        candidates = [
            (row, col - 1),
            (row - 1, col),
            (row, col + 1),
            (row + 1, col),
        ]

        neighbors = []

        for next_row, next_col in candidates:
            if (
                0 <= next_row < self.nrows
                and 0 <= next_col < self.ncols
                and self.maze[next_row, next_col] == 1
            ):
                neighbors.append(
                    (next_row, next_col)
                )

        return neighbors

    @staticmethod
    def _reconstruct_path(
        came_from: dict[
            tuple[int, int],
            tuple[int, int] | None,
        ],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Reconstruct the path from the start cell to the goal."""

        path = []
        current = goal

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()

        return path

    def search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> BFSResult:
        """Find a shortest path from start to goal using BFS."""

        self._validate_position(
            start,
            "start",
        )

        self._validate_position(
            goal,
            "goal",
        )

        if start == goal:
            return BFSResult(
                found=True,
                path=[start],
                path_length=0,
                nodes_explored=1,
                total_cost=0.0,
            )

        frontier = deque([start])

        came_from: dict[
            tuple[int, int],
            tuple[int, int] | None,
        ] = {
            start: None
        }

        nodes_explored = 0

        while frontier:
            current = frontier.popleft()
            nodes_explored += 1

            if current == goal:
                path = self._reconstruct_path(
                    came_from,
                    goal,
                )

                path_length = len(path) - 1

                return BFSResult(
                    found=True,
                    path=path,
                    path_length=path_length,
                    nodes_explored=nodes_explored,
                    total_cost=float(path_length),
                )

            for neighbor in self._neighbors(current):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    frontier.append(neighbor)

        return BFSResult(
            found=False,
            path=[],
            path_length=0,
            nodes_explored=nodes_explored,
            total_cost=float("inf"),
        )