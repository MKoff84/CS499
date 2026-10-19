"""
Training workflow for the enhanced Pirate Intelligent Agent.

This module coordinates the maze environment, neural-network agent,
experience replay memory, epsilon-greedy exploration, and target-network
updates while keeping those responsibilities in separate components.
"""

from dataclasses import dataclass

import numpy as np

from agent.pirate_agent import PirateAgent
from agent.replay_memory import Experience, ReplayMemory
from config import TrainingConfig
from environment.treasure_maze import TreasureMaze


@dataclass
class TrainingMetrics:
    """Stores summary information and history from a training run."""

    epochs_completed: int
    wins: int
    losses: int
    win_rate: float
    final_epsilon: float
    average_loss: float

    epoch_rewards: list[float]
    epoch_losses: list[float]
    epsilon_history: list[float]


class Trainer:
    """Coordinates deep Q-learning for the Pirate Intelligent Agent."""

    def __init__(
        self,
        environment: TreasureMaze,
        agent: PirateAgent,
        config: TrainingConfig,
    ) -> None:
        """Initialize and validate the training workflow."""

        if not isinstance(environment, TreasureMaze):
            raise TypeError(
                "environment must be a TreasureMaze object."
            )

        if not isinstance(agent, PirateAgent):
            raise TypeError(
                "agent must be a PirateAgent object."
            )

        if not isinstance(config, TrainingConfig):
            raise TypeError(
                "config must be a TrainingConfig object."
            )

        config.validate()

        expected_state_size = (
            environment.nrows * environment.ncols
        )

        if agent.state_size != expected_state_size:
            raise ValueError(
                "agent state size must match the maze size."
            )

        self.environment = environment
        self.agent = agent
        self.config = config

        self.memory = ReplayMemory(
            capacity=config.max_memory,
            random_seed=config.random_seed,
        )

        self.random_generator = np.random.default_rng(
            config.random_seed
        )

        self.epsilon = config.epsilon_start

    def _build_training_batch(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create model inputs and Q-learning targets from replay memory."""

        experiences = self.memory.sample(
            self.config.batch_size
        )

        states = np.vstack(
            [
                experience.state
                for experience in experiences
            ]
        ).astype(np.float32)

        next_states = np.vstack(
            [
                experience.next_state
                for experience in experiences
            ]
        ).astype(np.float32)

        targets = self.agent.predict(states)

        next_q_values = self.agent.predict(
            next_states,
            use_target=True,
        )

        for index, experience in enumerate(experiences):
            target_value = experience.reward

            if experience.status != "win":
                if experience.status != "lose":
                    target_value += (
                        self.config.discount_factor
                        * np.max(next_q_values[index])
                    )

            targets[
                index,
                experience.action,
            ] = target_value

        return states, targets

    def _train_from_memory(self) -> float:
        """Train the learning network using a replay-memory batch."""

        states, targets = self._build_training_batch()

        loss = self.agent.model.train_on_batch(
            states,
            targets,
        )

        return float(loss)

    def train(self) -> TrainingMetrics:
        """Run the configured deep Q-learning training process."""

        wins = 0
        losses = 0

        loss_history: list[float] = []
        epoch_rewards: list[float] = []
        epoch_losses: list[float] = []
        epsilon_history: list[float] = []

        for epoch in range(self.config.epochs):
            epoch_loss_values: list[float] = []

            start_cell = self.random_generator.choice(
                len(self.environment.free_cells)
            )

            pirate = self.environment.free_cells[
                int(start_cell)
            ]

            self.environment.reset(pirate)

            state = self.environment.observe()
            status = "not_over"

            while status == "not_over":
                valid_actions = (
                    self.environment.valid_actions()
                )

                if not valid_actions:
                    status = "lose"
                    break

                action = self.agent.select_action(
                    state=state,
                    valid_actions=valid_actions,
                    epsilon=self.epsilon,
                    random_generator=self.random_generator,
                )

                next_state, reward, status = (
                    self.environment.act(action)
                )

                experience = Experience(
                    state=state.copy(),
                    action=int(action),
                    reward=float(reward),
                    next_state=next_state.copy(),
                    status=status,
                )

                self.memory.add(experience)

                state = next_state

                loss = self._train_from_memory()

                loss_history.append(loss)
                epoch_loss_values.append(loss)

            epoch_rewards.append(
                float(self.environment.total_reward)
            )

            epoch_losses.append(
                float(np.mean(epoch_loss_values))
                if epoch_loss_values
                else 0.0
            )

            if status == "win":
                wins += 1
            else:
                losses += 1

            self.epsilon = max(
                self.config.epsilon_min,
                self.epsilon
                * self.config.epsilon_decay,
            )

            epsilon_history.append(self.epsilon)

            if (
                (epoch + 1)
                % self.config.target_update_frequency
                == 0
            ):
                self.agent.update_target_network()

        completed = wins + losses

        win_rate = (
            wins / completed
            if completed
            else 0.0
        )

        average_loss = (
            float(np.mean(loss_history))
            if loss_history
            else 0.0
        )

        return TrainingMetrics(
            epochs_completed=completed,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            final_epsilon=self.epsilon,
            average_loss=average_loss,
            epoch_rewards=epoch_rewards,
            epoch_losses=epoch_losses,
            epsilon_history=epsilon_history,
        )