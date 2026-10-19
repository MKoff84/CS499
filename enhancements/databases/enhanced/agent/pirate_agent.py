"""
Deep Q-learning agent for the enhanced Pirate Intelligent Agent.

This module encapsulates neural-network creation, prediction, target-network
management, and action selection independently from the maze environment
and training workflow.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from environment.treasure_maze import Action


class PirateAgent:
    """Neural-network agent used to navigate the treasure maze."""

    def __init__(
        self,
        state_size: int,
        action_size: int = 4,
        random_seed: int = 42,
    ) -> None:
        """Initialize the learning agent and its target network."""

        if state_size <= 0:
            raise ValueError(
                "state_size must be greater than zero."
            )

        if action_size <= 0:
            raise ValueError(
                "action_size must be greater than zero."
            )

        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed

        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        self.model = self._build_model()

        self.target_model = keras.models.clone_model(
            self.model
        )

        self.update_target_network()

    def _build_model(self) -> keras.Model:
        """Build and compile the deep Q-learning neural network."""

        model = keras.Sequential(
            [
                keras.Input(shape=(self.state_size,)),
                keras.layers.Dense(self.state_size),
                keras.layers.PReLU(),
                keras.layers.Dense(self.state_size),
                keras.layers.PReLU(),
                keras.layers.Dense(self.action_size),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="mse",
        )

        return model

    def predict(
        self,
        state: np.ndarray,
        use_target: bool = False,
    ) -> np.ndarray:
        """Predict Q-values for a maze state."""

        state_array = np.asarray(
            state,
            dtype=np.float32,
        )

        if state_array.ndim == 1:
            state_array = np.expand_dims(
                state_array,
                axis=0,
            )

        if (
            state_array.ndim != 2
            or state_array.shape[1] != self.state_size
        ):
            raise ValueError(
                f"state must contain {self.state_size} values."
            )

        selected_model = (
            self.target_model
            if use_target
            else self.model
        )

        prediction = selected_model(
            state_array,
            training=False,
        )

        return prediction.numpy()

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: list[Action],
        epsilon: float,
        random_generator: np.random.Generator,
    ) -> Action:
        """Select an action using an epsilon-greedy policy."""

        if not valid_actions:
            raise ValueError(
                "valid_actions cannot be empty."
            )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0.0 and 1.0."
            )

        if random_generator.random() < epsilon:
            return random_generator.choice(
                valid_actions
            )

        q_values = self.predict(state)[0]

        best_action = max(
            valid_actions,
            key=lambda action: q_values[int(action)],
        )

        return Action(best_action)

    def update_target_network(self) -> None:
        """Copy the learning model weights to the target network."""

        self.target_model.set_weights(
            self.model.get_weights()
        )