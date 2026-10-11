"""Unit tests for pathfinding algorithms."""

import unittest

import numpy as np

from algorithms.astar import AStarPathfinder
from algorithms.bfs import BFSPathfinder


class TestPathfindingAlgorithms(unittest.TestCase):
    """Tests A* and BFS pathfinding behavior."""

    def setUp(self) -> None:
        """Create a maze used by the pathfinding tests."""

        self.maze = np.array([
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ])

        self.start = (0, 0)
        self.goal = (3, 3)

    def test_astar_finds_path(self) -> None:
        """A* should find the expected shortest path."""

        pathfinder = AStarPathfinder(self.maze)

        result = pathfinder.search(
            self.start,
            self.goal,
        )

        self.assertTrue(result.found)
        self.assertEqual(result.path_length, 6)
        self.assertEqual(result.total_cost, 6.0)
        self.assertEqual(result.path[0], self.start)
        self.assertEqual(result.path[-1], self.goal)

    def test_bfs_finds_path(self) -> None:
        """BFS should find a shortest path."""

        pathfinder = BFSPathfinder(self.maze)

        result = pathfinder.search(
            self.start,
            self.goal,
        )

        self.assertTrue(result.found)
        self.assertEqual(result.path_length, 6)
        self.assertEqual(result.total_cost, 6.0)

    def test_astar_and_bfs_find_equal_cost(self) -> None:
        """A* and BFS should find equal-cost paths."""

        astar = AStarPathfinder(self.maze)
        bfs = BFSPathfinder(self.maze)

        astar_result = astar.search(
            self.start,
            self.goal,
        )

        bfs_result = bfs.search(
            self.start,
            self.goal,
        )

        self.assertEqual(
            astar_result.path_length,
            bfs_result.path_length,
        )

        self.assertEqual(
            astar_result.total_cost,
            bfs_result.total_cost,
        )

    def test_astar_rejects_blocked_start(self) -> None:
        """A* should reject a blocked starting cell."""

        pathfinder = AStarPathfinder(self.maze)

        with self.assertRaises(ValueError):
            pathfinder.search(
                (1, 0),
                self.goal,
            )

    def test_astar_rejects_out_of_bounds_goal(self) -> None:
        """A* should reject a goal outside the maze."""

        pathfinder = AStarPathfinder(self.maze)

        with self.assertRaises(ValueError):
            pathfinder.search(
                self.start,
                (10, 10),
            )


if __name__ == "__main__":
    unittest.main()