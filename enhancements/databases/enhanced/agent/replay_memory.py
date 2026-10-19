"""
Experience replay memory for the enhanced Pirate Intelligent Agent.

This module manages reinforcement-learning experiences independently
from the neural-network model and training process.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    """Represents a single reinforcement-learning experience."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    status: str


class ReplayMemory:
    """Stores and samples experiences used during model training."""

    def __init__(
        self,
        capacity: int = 1000,
        random_seed: int = 42,
    ) -> None:
        """Initialize replay memory with a fixed capacity."""

        if capacity <= 0:
            raise ValueError(
                "capacity must be greater than zero."
            )

        self.capacity = capacity

        self._memory: deque[Experience] = deque(
            maxlen=capacity
        )

        self._random_generator = np.random.default_rng(
            random_seed
        )

    def __len__(self) -> int:
        """Return the number of stored experiences."""

        return len(self._memory)

    def add(self, experience: Experience) -> None:
        """Add an experience to replay memory."""

        if not isinstance(experience, Experience):
            raise TypeError(
                "experience must be an Experience object."
            )

        self._memory.append(experience)

    def sample(
        self,
        batch_size: int,
    ) -> list[Experience]:
        """Return a random sample of stored experiences."""

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if not self._memory:
            raise ValueError(
                "cannot sample from empty replay memory."
            )

        sample_size = min(
            batch_size,
            len(self._memory),
        )

        indices = self._random_generator.choice(
            len(self._memory),
            size=sample_size,
            replace=False,
        )

        memory_list = list(self._memory)

        return [
            memory_list[int(index)]
            for index in indices
        ]

    def clear(self) -> None:
        """Remove all stored experiences."""

        self._memory.clear()