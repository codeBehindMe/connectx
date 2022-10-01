from abc import ABC
from abc import abstractmethod
from typing import Tuple



class Policy(ABC):
    
    @abstractmethod
    def get_epsilon_greedy_action(self, state) -> Tuple[int, callable]:
        raise NotImplementedError()

    @abstractmethod
    def get_greedy_action(self, state) -> int:
        raise NotImplementedError()