import re
import numpy as np
from typing import Tuple
from src.policy.policy import Policy
import tensorflow as tf
from collections import deque


def make_q_network(state_shape, action_size, learning_rate) -> tf.keras.models.Model:
    """Makes a simple q network for experimentation"""
    state_input = tf.keras.layers.Input(state_shape, name="states")
    actions_input = tf.keras.layers.Input((action_size,), name="actions_mask")

    hidden_1 = tf.keras.layers.Dense(10, activation="relu")(state_input)
    hidden_2 = tf.keras.layers.Dense(10, activation="relu")(hidden_1)
    q_values = tf.keras.layers.Dense(action_size)(hidden_2)

    masked_q_values = tf.keras.layers.Multiply()([q_values, actions_input])

    model = tf.keras.models.Model(
        inputs=[state_input, actions_input], outputs=masked_q_values
    )
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(loss="mse", optimizer=optimizer)
    return model


class MemoryBuffer:
    def __init__(self, size: int, batch_size: int, state_shape: Tuple) -> None:
        self.size = size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=size)
        self.state_shape = state_shape

    def add(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """Add an experience to memory.

        Args:
            experience (Tuple[np.ndarray, int, float, np.ndarray, bool]): A tuple containing
              state, action, reward, next_state, done_flag
        """
        self.buffer.append(experience)

    def sample(self):
        """
        Generate a batch of experiences from memory.
        """
        buffer_size = len(self.buffer)

        indices = np.random.choice(
            np.arange(buffer_size), size=self.batch_size, replace=False
        )

        states = np.zeros(shape=(self.batch_size,) + self.state_shape, dtype=np.float32)
        actions = np.zeros((self.batch_size,), dtype=np.int32)
        rewards = np.zeros((self.batch_size,), dtype=np.int32)
        next_states = np.zeros((self.batch_size, self.state_shape), dtype=np.float32)
        done_flags = np.empty((self.batch_size,), dtype=np.bool8)
        for i in indices:
            state, action, reward, next_state, done = buffer_size[i]
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            done_flags[i] = done

        return states, actions, rewards, next_states, done_flags


class DQNPolicy(Policy):
    def __init__(self, state_shape: int, action_size: int, gamma: float) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma

        self.q_network = make_q_network(self.state_shape, self.action_size, 0.01)
        self.q_network.predict()

    def get_greedy_action(self, state) -> int:
        return super().get_greedy_action(state)
