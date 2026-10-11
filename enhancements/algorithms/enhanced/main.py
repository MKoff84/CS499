"""
Entry point for the enhanced Pirate Intelligent Agent.

This module creates the maze environment, learning agent, training
configuration, and trainer before running a complete training session.
"""

import numpy as np

from agent.pirate_agent import PirateAgent
from config import TrainingConfig
from environment.treasure_maze import TreasureMaze
from training.trainer import Trainer


def create_maze() -> np.ndarray:
    """Create the original CS 370 maze used by the Pirate Intelligent Agent."""

    return np.array(
        [
            [1., 0., 1., 1., 1., 1., 1., 1.],
            [1., 0., 1., 1., 1., 0., 1., 1.],
            [1., 1., 1., 1., 0., 1., 0., 1.],
            [1., 1., 1., 0., 1., 1., 1., 1.],
            [1., 1., 0., 1., 1., 1., 1., 1.],
            [1., 1., 1., 0., 1., 0., 0., 0.],
            [1., 1., 1., 0., 1., 1., 1., 1.],
            [1., 1., 1., 1., 0., 1., 1., 1.],
        ],
        dtype=float,
    )


def main() -> None:
    """Run the enhanced Pirate Intelligent Agent training workflow."""

    maze = create_maze()

    config = TrainingConfig()
    config.validate()

    environment = TreasureMaze(
        maze=maze,
        pirate=(0, 0),
    )

    agent = PirateAgent(
        state_size=maze.size,
        action_size=4,
        random_seed=config.random_seed,
    )

    trainer = Trainer(
        environment=environment,
        agent=agent,
        config=config,
    )

    print("Enhanced Pirate Intelligent Agent")
    print("----------------------------------")
    print(f"Maze size: {environment.nrows} x {environment.ncols}")
    print(f"Training epochs: {config.epochs}")
    print(f"Replay-memory capacity: {config.max_memory}")
    print(f"Random seed: {config.random_seed}")

    print("\nTraining started...")

    metrics = trainer.train()

    print("\nTraining completed.")
    print(f"Epochs completed: {metrics.epochs_completed}")
    print(f"Wins: {metrics.wins}")
    print(f"Losses: {metrics.losses}")
    print(f"Win rate: {metrics.win_rate:.2%}")
    print(f"Final epsilon: {metrics.final_epsilon:.4f}")
    print(f"Average loss: {metrics.average_loss:.6f}")

    print("\nMetric history:")
    print(f"Reward records: {len(metrics.epoch_rewards)}")
    print(f"Loss records: {len(metrics.epoch_losses)}")
    print(f"Epsilon records: {len(metrics.epsilon_history)}")


if __name__ == "__main__":
    main()