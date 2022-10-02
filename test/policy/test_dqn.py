import numpy as np
import pytest
import gym
from src.policy.dqn import MemoryBuffer

@pytest.fixture(scope="module")
def cartpole_env():
  
  env = gym.make("CartPole-v1")
  return env

@pytest.mark.usefixtures("cartpole_env")
class TestMemoryBuffer:
  
  def test_memory_buffer_allocation(self, cartpole_env: gym.Env):
    current_state : np.ndarray

    current_state , _ = cartpole_env.reset()
    mem_buffer = MemoryBuffer(20,10, current_state.shape)

    for _ in range(100):
      action = cartpole_env.action_space.sample()
      next_state, reward, done, trunc , _ = cartpole_env.step(action)

      mem_buffer.add((current_state,action,reward,next_state,done))
      current_state = next_state

      if done or trunc:
        current_state, _ = cartpole_env.reset()
      
    cartpole_env.close()