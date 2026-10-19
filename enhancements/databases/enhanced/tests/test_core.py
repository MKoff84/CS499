"""
Automated tests for the enhanced Pirate Intelligent Agent.

These tests verify configuration validation, maze behavior,
replay-memory management, and agent input validation.
"""

import unittest

import numpy as np

from agent.pirate_agent import PirateAgent
from agent.replay_memory import Experience, ReplayMemory
from algorithms.astar import AStarPathfinder
from config import TrainingConfig
from environment.treasure_maze import Action, TreasureMaze


class TestTrainingConfig(unittest.TestCase):
    """Tests configuration validation."""

    def test_default_configuration_is_valid(self):
        config = TrainingConfig()

        # No exception should be raised.
        config.validate()

    def test_invalid_epsilon_is_rejected(self):
        config = TrainingConfig(
            epsilon_start=1.5
        )

        with self.assertRaises(ValueError):
            config.validate()


class TestTreasureMaze(unittest.TestCase):
    """Tests maze validation and environment behavior."""

    def setUp(self):
        self.maze = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def test_environment_initializes(self):
        environment = TreasureMaze(self.maze)

        self.assertEqual(
            environment.state,
            (0, 0, "start"),
        )

        self.assertEqual(
            environment.target,
            (3, 3),
        )

    def test_valid_movement(self):
        environment = TreasureMaze(self.maze)

        _, reward, status = environment.act(
            Action.RIGHT
        )

        self.assertEqual(
            environment.state,
            (0, 1, "valid"),
        )

        self.assertAlmostEqual(
            reward,
            -0.04,
        )

        self.assertEqual(
            status,
            "not_over",
        )

    def test_invalid_action_state_is_recorded(self):
        environment = TreasureMaze(self.maze)

        environment.act(Action.UP)

        self.assertEqual(
            environment.state,
            (0, 0, "invalid"),
        )

    def test_invalid_maze_value_is_rejected(self):
        invalid_maze = np.array(
            [
                [1, 2],
                [1, 1],
            ]
        )

        with self.assertRaises(ValueError):
            TreasureMaze(invalid_maze)


class TestReplayMemory(unittest.TestCase):
    """Tests bounded experience replay memory."""

    def test_capacity_discards_oldest_experience(self):
        memory = ReplayMemory(
            capacity=3,
            random_seed=42,
        )

        for value in range(4):
            state = np.array(
                [[value]],
                dtype=float,
            )

            memory.add(
                Experience(
                    state=state,
                    action=0,
                    reward=0.0,
                    next_state=state.copy(),
                    status="not_over",
                )
            )

        self.assertEqual(
            len(memory),
            3,
        )

        sampled = memory.sample(3)

        stored_values = sorted(
            int(experience.state[0][0])
            for experience in sampled
        )

        self.assertEqual(
            stored_values,
            [1, 2, 3],
        )

    def test_empty_memory_cannot_be_sampled(self):
        memory = ReplayMemory()

        with self.assertRaises(ValueError):
            memory.sample(1)


class TestPirateAgent(unittest.TestCase):
    """Tests neural-network agent validation."""

    def test_prediction_shape(self):
        agent = PirateAgent(
            state_size=16,
            random_seed=42,
        )

        state = np.ones(
            (1, 16),
            dtype=np.float32,
        )

        prediction = agent.predict(state)

        self.assertEqual(
            prediction.shape,
            (1, 4),
        )

    def test_invalid_state_shape_is_rejected(self):
        agent = PirateAgent(
            state_size=16,
            random_seed=42,
        )

        invalid_state = np.ones(
            (1, 10),
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            agent.predict(invalid_state)

class TestAStarPathfinder(unittest.TestCase):
    """Tests A* pathfinding behavior and validation."""

    def setUp(self):
        self.maze = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        self.pathfinder = AStarPathfinder(
            self.maze
        )

    def test_shortest_path_is_found(self):
        result = self.pathfinder.search(
            start=(0, 0),
            goal=(3, 3),
        )

        self.assertTrue(result.found)

        self.assertEqual(
            result.path_length,
            6,
        )

        self.assertEqual(
            result.path[0],
            (0, 0),
        )

        self.assertEqual(
            result.path[-1],
            (3, 3),
        )

        self.assertAlmostEqual(
            result.total_cost,
            6.0,
        )

    def test_blocked_start_is_rejected(self):
        with self.assertRaises(ValueError):
            self.pathfinder.search(
                start=(0, 2),
                goal=(3, 3),
            )

    def test_blocked_goal_is_rejected(self):
        with self.assertRaises(ValueError):
            self.pathfinder.search(
                start=(0, 0),
                goal=(0, 2),
            )

    def test_out_of_bounds_position_is_rejected(self):
        with self.assertRaises(ValueError):
            self.pathfinder.search(
                start=(0, 0),
                goal=(10, 10),
            )

    def test_unreachable_goal_returns_failure(self):
        unreachable_maze = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        pathfinder = AStarPathfinder(
            unreachable_maze
        )

        result = pathfinder.search(
            start=(0, 0),
            goal=(3, 3),
        )

        self.assertFalse(result.found)
        self.assertEqual(
            result.path,
            [],
        )
        self.assertEqual(
            result.path_length,
            0,
        )
        self.assertEqual(
            result.nodes_explored,
            3,
        )
        self.assertEqual(
            result.total_cost,
            float("inf"),
        )

if __name__ == "__main__":
    unittest.main()