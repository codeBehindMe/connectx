from kaggle_environments import evaluate, make, utils
from agent import Policy


if __name__ == "__main__":

    env = make("connectx", debug=True)
    env.render(mode="ipython")

    trainer = env.train([None, "random"])

    policy = Policy(env.configuration.columns, 0.1)

    for _ in range(1000):
        observation = trainer.reset()
        steps = 0
        while not env.done:
            action, update_policy_fn = policy.get_epsilon_greedy_action(
                observation.board
            )
            observation, reward, done, info = trainer.step(action)
            update_policy_fn(reward, observation.board)
            steps += 1
        print(f"episode steps: {steps}")
