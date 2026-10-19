"""Unit tests for the experiment database enhancement."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from database.experiment_database import ExperimentDatabase


class TestExperimentDatabase(unittest.TestCase):
    """Test SQLite experiment persistence and integrity."""

    def setUp(self) -> None:
        """Create an isolated temporary database for each test."""
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = (
            Path(self.temp_directory.name) / "test_experiments.db"
        )
        self.database = ExperimentDatabase(self.database_path)

    def tearDown(self) -> None:
        """Remove the temporary database after each test."""
        self.temp_directory.cleanup()

    def test_add_and_retrieve_experiment(self) -> None:
        """An experiment should persist and be retrievable."""
        run_id = self.database.add_experiment_run(
            algorithm="DQN",
            maze_rows=8,
            maze_columns=8,
            epochs=15,
            wins=7,
            losses=8,
            win_rate=7 / 15,
            average_loss=0.002862,
            final_epsilon=0.9276,
            random_seed=42,
        )

        stored_run = self.database.get_experiment_run(run_id)

        self.assertIsNotNone(stored_run)
        self.assertEqual(stored_run["algorithm"], "DQN")
        self.assertEqual(stored_run["maze_rows"], 8)
        self.assertEqual(stored_run["maze_columns"], 8)
        self.assertEqual(stored_run["epochs"], 15)
        self.assertEqual(stored_run["wins"], 7)
        self.assertEqual(stored_run["losses"], 8)

    def test_algorithm_results_relationship(self) -> None:
        """Multiple algorithm results should reference one experiment."""
        run_id = self.database.add_experiment_run(
            algorithm="Pathfinding Comparison",
            maze_rows=8,
            maze_columns=8,
        )

        self.database.add_algorithm_result(
            run_id=run_id,
            algorithm="A*",
            path_found=True,
            path_length=14,
            nodes_explored=25,
            total_cost=14.0,
        )

        self.database.add_algorithm_result(
            run_id=run_id,
            algorithm="BFS",
            path_found=True,
            path_length=14,
            nodes_explored=26,
            total_cost=14.0,
        )

        results = self.database.get_algorithm_results(run_id)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["algorithm"], "A*")
        self.assertEqual(results[1]["algorithm"], "BFS")
        self.assertEqual(results[0]["run_id"], run_id)
        self.assertEqual(results[1]["run_id"], run_id)

    def test_empty_algorithm_rejected(self) -> None:
        """An empty algorithm name should be rejected."""
        with self.assertRaises(ValueError):
            self.database.add_experiment_run(
                algorithm="",
                maze_rows=8,
                maze_columns=8,
            )

    def test_invalid_win_rate_rejected(self) -> None:
        """Win rates outside zero to one should be rejected."""
        with self.assertRaises(ValueError):
            self.database.add_experiment_run(
                algorithm="DQN",
                maze_rows=8,
                maze_columns=8,
                win_rate=1.5,
            )

    def test_invalid_foreign_key_rejected(self) -> None:
        """Algorithm results must reference an existing experiment."""
        with self.assertRaises(sqlite3.IntegrityError):
            self.database.add_algorithm_result(
                run_id=999999,
                algorithm="A*",
                path_found=True,
                path_length=14,
                nodes_explored=25,
                total_cost=14.0,
            )

    def test_cascade_delete(self) -> None:
        """Deleting an experiment should delete related results."""
        run_id = self.database.add_experiment_run(
            algorithm="Pathfinding Comparison",
            maze_rows=8,
            maze_columns=8,
        )

        self.database.add_algorithm_result(
            run_id=run_id,
            algorithm="A*",
            path_found=True,
            path_length=14,
            nodes_explored=25,
            total_cost=14.0,
        )

        self.assertEqual(
            len(self.database.get_algorithm_results(run_id)),
            1,
        )

        deleted = self.database.delete_experiment_run(run_id)

        self.assertTrue(deleted)
        self.assertIsNone(
            self.database.get_experiment_run(run_id)
        )
        self.assertEqual(
            self.database.get_algorithm_results(run_id),
            [],
        )


if __name__ == "__main__":
    unittest.main()