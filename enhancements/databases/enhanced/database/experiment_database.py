"""SQLite database support for Pirate Intelligent Agent experiments."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class ExperimentDatabase:
    """Manage persistent experiment data using SQLite."""

    def __init__(self, database_path: str | Path = "pirate_experiments.db") -> None:
        """Initialize the database and create required tables."""
        self.database_path = Path(database_path)
        self._create_tables()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Provide a managed SQLite connection with foreign keys enabled."""
        connection = sqlite3.connect(self.database_path)

        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _create_tables(self) -> None:
        """Create the experiment tables and indexes if they do not exist."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    algorithm TEXT NOT NULL,
                    maze_rows INTEGER NOT NULL CHECK (maze_rows > 0),
                    maze_columns INTEGER NOT NULL CHECK (maze_columns > 0),
                    epochs INTEGER CHECK (epochs IS NULL OR epochs >= 0),
                    wins INTEGER CHECK (wins IS NULL OR wins >= 0),
                    losses INTEGER CHECK (losses IS NULL OR losses >= 0),
                    win_rate REAL CHECK (
                        win_rate IS NULL OR
                        (win_rate >= 0.0 AND win_rate <= 1.0)
                    ),
                    average_loss REAL,
                    final_epsilon REAL CHECK (
                        final_epsilon IS NULL OR
                        (final_epsilon >= 0.0 AND final_epsilon <= 1.0)
                    ),
                    random_seed INTEGER,
                    duration_seconds REAL CHECK (
                        duration_seconds IS NULL OR duration_seconds >= 0.0
                    )
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS algorithm_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    algorithm TEXT NOT NULL,
                    path_found INTEGER NOT NULL CHECK (path_found IN (0, 1)),
                    path_length INTEGER NOT NULL CHECK (path_length >= 0),
                    nodes_explored INTEGER NOT NULL CHECK (nodes_explored >= 0),
                    total_cost REAL,
                    execution_time REAL CHECK (
                        execution_time IS NULL OR execution_time >= 0.0
                    ),
                    FOREIGN KEY (run_id)
                        REFERENCES experiment_runs(run_id)
                        ON DELETE CASCADE
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_algorithm
                ON experiment_runs(algorithm)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_results_run_id
                ON algorithm_results(run_id)
                """
            )

    def add_experiment_run(
        self,
        algorithm: str,
        maze_rows: int,
        maze_columns: int,
        epochs: int | None = None,
        wins: int | None = None,
        losses: int | None = None,
        win_rate: float | None = None,
        average_loss: float | None = None,
        final_epsilon: float | None = None,
        random_seed: int | None = None,
        duration_seconds: float | None = None,
    ) -> int:
        """Insert an experiment run and return its generated run ID."""
        if not algorithm.strip():
            raise ValueError("Algorithm name cannot be empty.")

        if maze_rows <= 0 or maze_columns <= 0:
            raise ValueError("Maze dimensions must be positive.")

        if win_rate is not None and not 0.0 <= win_rate <= 1.0:
            raise ValueError("Win rate must be between 0.0 and 1.0.")

        if final_epsilon is not None and not 0.0 <= final_epsilon <= 1.0:
            raise ValueError("Final epsilon must be between 0.0 and 1.0.")

        query = """
            INSERT INTO experiment_runs (
                algorithm,
                maze_rows,
                maze_columns,
                epochs,
                wins,
                losses,
                win_rate,
                average_loss,
                final_epsilon,
                random_seed,
                duration_seconds
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            algorithm.strip(),
            maze_rows,
            maze_columns,
            epochs,
            wins,
            losses,
            win_rate,
            average_loss,
            final_epsilon,
            random_seed,
            duration_seconds,
        )

        with self._connect() as connection:
            cursor = connection.execute(query, values)
            run_id = cursor.lastrowid

        if run_id is None:
            raise RuntimeError("Database did not return a run ID.")

        return int(run_id)

    def add_algorithm_result(
        self,
        run_id: int,
        algorithm: str,
        path_found: bool,
        path_length: int,
        nodes_explored: int,
        total_cost: float | None,
        execution_time: float | None = None,
    ) -> int:
        """Store a pathfinding result associated with an experiment run."""
        if run_id <= 0:
            raise ValueError("Run ID must be positive.")

        if not algorithm.strip():
            raise ValueError("Algorithm name cannot be empty.")

        if path_length < 0:
            raise ValueError("Path length cannot be negative.")

        if nodes_explored < 0:
            raise ValueError("Nodes explored cannot be negative.")

        query = """
            INSERT INTO algorithm_results (
                run_id,
                algorithm,
                path_found,
                path_length,
                nodes_explored,
                total_cost,
                execution_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        values = (
            run_id,
            algorithm.strip(),
            int(path_found),
            path_length,
            nodes_explored,
            total_cost,
            execution_time,
        )

        with self._connect() as connection:
            cursor = connection.execute(query, values)
            result_id = cursor.lastrowid

        if result_id is None:
            raise RuntimeError("Database did not return a result ID.")

        return int(result_id)

    def get_experiment_run(self, run_id: int) -> dict[str, Any] | None:
        """Retrieve one experiment run by its ID."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM experiment_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        return dict(row) if row is not None else None

    def get_algorithm_results(self, run_id: int) -> list[dict[str, Any]]:
        """Retrieve all algorithm results associated with a run."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM algorithm_results
                WHERE run_id = ?
                ORDER BY result_id
                """,
                (run_id,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_all_experiment_runs(self) -> list[dict[str, Any]]:
        """Retrieve all stored experiment runs, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM experiment_runs
                ORDER BY run_id DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_experiment_run(self, run_id: int) -> bool:
        """Delete an experiment and its related algorithm results."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM experiment_runs
                WHERE run_id = ?
                """,
                (run_id,),
            )

        return cursor.rowcount > 0