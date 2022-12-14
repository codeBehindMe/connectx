import random
import numpy as np
from collections import defaultdict
from typing import Tuple, Union
from functools import wraps


def where_equals(a: Union[list, tuple], val: int, inverse=False) -> np.ndarray:
    if inverse is True:
        return np.where(np.asarray(a) != val)[0]

    return np.where(np.asarray(a) == val)[0]


def argmax(a: np.ndarray) -> int:
    max_val = np.max(a)
    mask = where_equals(a, max_val)
    if mask.size == 0:
        return np.argmax(a)

    tie_breaker_index = np.random.choice(mask)
    return tie_breaker_index


def tuplise(m):
    @wraps(m)
    def _impl(self, *m_args, **m_kwargs):
        return m(self, tuple(m_args[0]), **m_kwargs)

    return _impl


class Policy:
    def __init__(
        self, num_columns: int, epsilon: float, alpha: float = 0.01, gamma=0.9
    ) -> None:

        self.num_columns = num_columns
        self.q = defaultdict(lambda: np.zeros(num_columns))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def _make_callback(self, board: list, action: int) -> callable:
        board = tuple(board)

        def _callback(reward, next_board):
            next_board = tuple(next_board)
            max_q_s_prime_a = self.q[next_board][self.get_greedy_action(next_board)]
            # print(f"pre update q(s,a): {self.q[board][action]}")
            self.q[board][action] += self.alpha * (
                reward + self.gamma * (max_q_s_prime_a - self.q[board][action])
            )
            # print(f"post update q(s,a): {self.q[board][action]}")
            return

        return _callback

    @tuplise
    def get_greedy_action(self, board: list) -> int:
        # We set the values of the unavailable functions to be negative inf
        unavailable_actions_mask = where_equals(board[0 : self.num_columns], 0, True)
        self.q[board][unavailable_actions_mask] = np.NINF
        action = int(argmax(self.q[board]))

        return action

    @tuplise
    def get_epsilon_greedy_action(self, board: list) -> Tuple[int, callable]:

        if board[0] != 0:
            _ = 1
        if self.epsilon < random.uniform(0, 1):
            # We limit the random pick to only actions that are allowed.
            available_actions_mask = where_equals(board[0 : self.num_columns], 0)
            # Pick a random action from available actions.
            action = int(np.random.choice(available_actions_mask))
        else:
            action = self.get_greedy_action(board)

        return action, self._make_callback(board, action)


class Agent:
    def __init__(self, configuration, epsilon=0.1) -> None:
        self.config = configuration
        self.epsilon = epsilon

        self.policy = Policy(configuration.columns, self.epsilon)
