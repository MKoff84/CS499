"""
Entry point for the enhanced Pirate Intelligent Agent.

This module creates the maze environment, learning agent, training
configuration, and trainer before running a complete training session.
Training results are persisted to a SQLite experiment database.
"""

from time import perf_counter

import numpy as np

from agent.pirate_agent import PirateAgent
from algorithms.astar import AStarPathfinder
from algorithms.bfs import BFSPathfinder
from config import TrainingConfig
from database.experiment_database import ExperimentDatabase
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

    start_time = perf_counter()
    metrics = trainer.train()
    duration_seconds = perf_counter() - start_time

    print("\nTraining completed.")
    print(f"Epochs completed: {metrics.epochs_completed}")
    print(f"Wins: {metrics.wins}")
    print(f"Losses: {metrics.losses}")
    print(f"Win rate: {metrics.win_rate:.2%}")
    print(f"Final epsilon: {metrics.final_epsilon:.4f}")
    print(f"Average loss: {metrics.average_loss:.6f}")
    print(f"Training duration: {duration_seconds:.2f} seconds")

    print("\nMetric history:")
    print(f"Reward records: {len(metrics.epoch_rewards)}")
    print(f"Loss records: {len(metrics.epoch_losses)}")
    print(f"Epsilon records: {len(metrics.epsilon_history)}")

    database = ExperimentDatabase("pirate_experiments.db")

    run_id = database.add_experiment_run(
        algorithm="DQN",
        maze_rows=environment.nrows,
        maze_columns=environment.ncols,
        epochs=metrics.epochs_completed,
        wins=metrics.wins,
        losses=metrics.losses,
        win_rate=metrics.win_rate,
        average_loss=metrics.average_loss,
        final_epsilon=metrics.final_epsilon,
        random_seed=config.random_seed,
        duration_seconds=duration_seconds,
    )

    print("\nExperiment saved to database.")
    print(f"Database run ID: {run_id}")

    # Run deterministic pathfinding algorithms for comparison.
    start = (0, 0)
    goal = (maze.shape[0] - 1, maze.shape[1] - 1)

    astar = AStarPathfinder(maze)
    bfs = BFSPathfinder(maze)

    astar_start = perf_counter()
    astar_result = astar.search(start, goal)
    astar_duration = perf_counter() - astar_start

    bfs_start = perf_counter()
    bfs_result = bfs.search(start, goal)
    bfs_duration = perf_counter() - bfs_start

    # Create a parent experiment for the pathfinding comparison.
    comparison_run_id = database.add_experiment_run(
        algorithm="Pathfinding Comparison",
        maze_rows=maze.shape[0],
        maze_columns=maze.shape[1],
        random_seed=config.random_seed,
    )

    database.add_algorithm_result(
        run_id=comparison_run_id,
        algorithm="A*",
        path_found=astar_result.found,
        path_length=astar_result.path_length,
        nodes_explored=astar_result.nodes_explored,
        total_cost=astar_result.total_cost,
        execution_time=astar_duration,
    )

    database.add_algorithm_result(
        run_id=comparison_run_id,
        algorithm="BFS",
        path_found=bfs_result.found,
        path_length=bfs_result.path_length,
        nodes_explored=bfs_result.nodes_explored,
        total_cost=bfs_result.total_cost,
        execution_time=bfs_duration,
    )

    print("\nPathfinding comparison completed.")
    print(
        f"A*: found={astar_result.found}, "
        f"path length={astar_result.path_length}, "
        f"nodes explored={astar_result.nodes_explored}, "
        f"cost={astar_result.total_cost:.0f}, "
        f"time={astar_duration:.6f} seconds"
    )
    print(
        f"BFS: found={bfs_result.found}, "
        f"path length={bfs_result.path_length}, "
        f"nodes explored={bfs_result.nodes_explored}, "
        f"cost={bfs_result.total_cost:.0f}, "
        f"time={bfs_duration:.6f} seconds"
    )

    print(
        "Pathfinding results saved to database "
        f"under run ID: {comparison_run_id}"
    )


if __name__ == "__main__":
    main()