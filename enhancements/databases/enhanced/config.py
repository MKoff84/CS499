"""
Configuration settings for the enhanced Pirate Intelligent Agent.

This module centralizes training settings that were previously
defined throughout the original Jupyter notebook.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Stores and validates reinforcement-learning configuration values."""

    epochs: int = 15 
    max_memory: int = 1000
    batch_size: int = 50

    discount_factor: float = 0.95

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995

    target_update_frequency: int = 10
    random_seed: int = 42

    def validate(self) -> None:
        """Validate configuration values before training begins."""

        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if self.max_memory <= 0:
            raise ValueError("max_memory must be greater than zero.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError(
                "epsilon_start must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.epsilon_min <= 1.0:
            raise ValueError(
                "epsilon_min must be between 0.0 and 1.0."
            )

        if self.epsilon_min > self.epsilon_start:
            raise ValueError(
                "epsilon_min cannot be greater than epsilon_start."
            )

        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be greater than 0.0 and no more than 1.0."
            )

        if self.target_update_frequency <= 0:
            raise ValueError(
                "target_update_frequency must be greater than zero."
            )