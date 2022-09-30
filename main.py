from kaggle_environments import evaluate, make, utils
from agent import Policy
import numpy as np


if __name__ == "__main__":

    env = make("connectx", debug=True)
    env.render(mode="ipython")

    trainer = env.train([None, "random"])

    policy = Policy(env.configuration.columns, 0.1)
    steps_per_game_sum = 0
    game_count = 0
    with open("average_steps.hist", "w") as f:

        for _ in range(1000000):
            observation = trainer.reset()
            steps = 0
            while not env.done:
                action, update_policy_fn = policy.get_epsilon_greedy_action(
                    observation.board
                )
                observation, reward, done, info = trainer.step(action)
                update_policy_fn(reward, observation.board)
                steps += 1
            print(f"reward : {reward}")
            steps_per_game_sum += steps
            game_count += 1
            print(steps_per_game_sum / game_count)
            f.write(f"{steps_per_game_sum / game_count},{reward}\n")
