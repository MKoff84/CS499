"""
Maze environment for the enhanced Pirate Intelligent Agent.

This module separates environment behavior from the reinforcement-learning
agent and training process. It manages maze validation, pirate movement,
rewards, state observations, and available actions.
"""

from enum import IntEnum

import numpy as np


class Action(IntEnum):
    """Valid movement actions available to the pirate."""

    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class TreasureMaze:
    """Represents the maze environment used by the intelligent agent."""

    VISITED_MARK = 0.8
    PIRATE_MARK = 0.5

    REWARD_GOAL = 1.0
    REWARD_REVISIT = -0.25
    REWARD_INVALID = -0.75
    REWARD_STEP = -0.04

    def __init__(
        self,
        maze: np.ndarray,
        pirate: tuple[int, int] = (0, 0),
    ) -> None:
        """Initialize and validate the maze environment."""

        self._validate_maze(maze)

        self._original_maze = np.array(
            maze,
            dtype=float,
            copy=True,
        )

        self.nrows, self.ncols = self._original_maze.shape

        self.target = (
            self.nrows - 1,
            self.ncols - 1,
        )

        self.free_cells = [
            (row, col)
            for row in range(self.nrows)
            for col in range(self.ncols)
            if self._original_maze[row, col] == 1.0
        ]

        if self.target not in self.free_cells:
            raise ValueError(
                "The target cell must be an open cell in the maze."
            )

        self.free_cells.remove(self.target)

        self._validate_position(pirate)

        self.maze: np.ndarray
        self.state: tuple[int, int, str]
        self.visited: set[tuple[int, int]]
        self.total_reward: float
        self.min_reward: float

        self.reset(pirate)

    @staticmethod
    def _validate_maze(maze: np.ndarray) -> None:
        """Validate the maze before creating the environment."""

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

        if maze.shape[0] < 2 or maze.shape[1] < 2:
            raise ValueError(
                "maze must contain at least two rows and two columns."
            )

        if not np.all(np.isin(maze, [0, 1])):
            raise ValueError(
                "maze cells must contain only 0 for blocked cells "
                "or 1 for open cells."
            )

    def _validate_position(
        self,
        position: tuple[int, int],
    ) -> None:
        """Validate a starting position within the maze."""

        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or not all(
                isinstance(value, int)
                for value in position
            )
        ):
            raise TypeError(
                "pirate position must be a tuple "
                "containing two integers."
            )

        row, col = position

        if not (
            0 <= row < self.nrows
            and 0 <= col < self.ncols
        ):
            raise ValueError(
                "pirate position must be within "
                "the maze boundaries."
            )

        if self._original_maze[row, col] != 1.0:
            raise ValueError(
                "pirate position must be located "
                "on an open maze cell."
            )

        if position == self.target:
            raise ValueError(
                "pirate starting position cannot "
                "be the target cell."
            )
            
    def reset(
        self,
        pirate: tuple[int, int] = (0, 0),
    ) -> None:
        """Reset the maze and place the pirate at a valid starting cell."""

        self._validate_position(pirate)

        self.maze = self._original_maze.copy()

        row, col = pirate
        self.maze[row, col] = self.PIRATE_MARK

        self.state = (row, col, "start")
        self.visited = set()

        self.total_reward = 0.0
        self.min_reward = -0.5 * self.maze.size

    def observe(self) -> np.ndarray:
        """Return the current maze state as a flattened model input."""

        canvas = self.draw_environment()

        return np.reshape(
            canvas,
            (1, -1),
        )

    def draw_environment(self) -> np.ndarray:
        """Create a copy of the maze representing the current state."""

        canvas = self.maze.copy()

        row, col, _ = self.state

        for visited_row, visited_col in self.visited:
            if (visited_row, visited_col) != (row, col):
                canvas[visited_row, visited_col] = self.VISITED_MARK

        canvas[row, col] = self.PIRATE_MARK

        return canvas

    def game_status(self) -> str:
        """Return the current status of the maze episode."""

        if self.total_reward < self.min_reward:
            return "lose"

        row, col, _ = self.state

        if (row, col) == self.target:
            return "win"

        return "not_over"

    def valid_actions(self) -> list[Action]:
        """Return the actions currently available to the pirate."""

        row, col, _ = self.state

        actions = [
            Action.LEFT,
            Action.UP,
            Action.RIGHT,
            Action.DOWN,
        ]

        if col == 0 or self._original_maze[row, col - 1] == 0:
            actions.remove(Action.LEFT)

        if row == 0 or self._original_maze[row - 1, col] == 0:
            actions.remove(Action.UP)

        if (
            col == self.ncols - 1
            or self._original_maze[row, col + 1] == 0
        ):
            actions.remove(Action.RIGHT)

        if (
            row == self.nrows - 1
            or self._original_maze[row + 1, col] == 0
        ):
            actions.remove(Action.DOWN)

        return actions

    def update_state(self, action: Action) -> None:
        """Update the pirate's position based on the selected action."""

        row, col, _ = self.state
        self.visited.add((row, col))

        valid = self.valid_actions()

        if not valid:
            next_mode = "blocked"

        elif action not in valid:
            next_mode = "invalid"

        else:
            next_mode = "valid"

            if action == Action.LEFT:
                col -= 1

            elif action == Action.UP:
                row -= 1

            elif action == Action.RIGHT:
                col += 1

            elif action == Action.DOWN:
                row += 1

        self.state = (row, col, next_mode)

    def get_reward(self) -> float:
        """Calculate the reward for the pirate's current state."""

        row, col, mode = self.state

        if (row, col) == self.target:
            return self.REWARD_GOAL

        if mode == "blocked":
            return self.min_reward - 1.0

        if mode == "invalid":
            return self.REWARD_INVALID

        if (row, col) in self.visited:
            return self.REWARD_REVISIT

        return self.REWARD_STEP

    def act(
        self,
        action: Action,
    ) -> tuple[np.ndarray, float, str]:
        """Perform an action and return the resulting state information."""

        try:
            validated_action = Action(action)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Unsupported action: {action}"
            ) from error

        self.update_state(validated_action)

        reward = self.get_reward()
        self.total_reward += reward

        environment_state = self.observe()
        status = self.game_status()

        return environment_state, reward, status