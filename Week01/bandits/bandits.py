# TODO: Add Recodex IDs.

#!/usr/bin/env python3
import argparse

import numpy as np


# A class providing MultiArmedBandits environment.
# You should not modify it or access its private attributes.
class MultiArmedBandits():
    def __init__(self, bandits: int, seed: int) -> None:
        self.__generator = np.random.RandomState(seed)
        self.__bandits = [None] * bandits
        self.reset()

    def reset(self) -> None:
        for i in range(len(self.__bandits)):
            self.__bandits[i] = self.__generator.normal(0., 1.)

    def step(self, action: int) -> float:
        return self.__generator.normal(self.__bandits[action], 1.)

    def greedy(self, epsilon: float) -> bool:
        return self.__generator.uniform() >= epsilon


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0, type=float, help="Learning rate to use or 0 for averaging.")
parser.add_argument("--bandits", default=10, type=int, help="Number of bandits.")
parser.add_argument("--episode_length", default=1000, type=int, help="Number of trials per episode.")
parser.add_argument("--episodes", default=100, type=int, help="Episodes to perform.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor (if applicable).")
parser.add_argument("--initial", default=0, type=float, help="Initial estimation of values.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(env: MultiArmedBandits, args: argparse.Namespace) -> float:
    current_estimates = np.full(args.bandits, fill_value=args.initial, dtype=float)
    current_hit_counts = np.zeros(args.bandits, dtype=int)
    rewards = 0
    for step in range(args.episode_length):
        if env.greedy(args.epsilon):
            action = np.random.choice(np.flatnonzero(current_estimates == current_estimates.max()))  # Break ties.
            # action = np.argmax(current_estimates)
        else:
            action = np.random.randint(0, args.bandits)

        # Perform the action.
        reward = env.step(action)
        current_hit_counts[action] += 1
        rewards += reward

        difference = reward - current_estimates[action]
        if args.alpha == 0:  # Averaging.
            current_estimates[action] += (1 / current_hit_counts[action]) * difference
        else:  # Learning rate.
            current_estimates[action] += args.alpha * difference

    return rewards / args.episode_length


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = MultiArmedBandits(args.bandits, seed=args.seed)

    # Set random seed
    np.random.seed(args.seed)

    returns = []
    for _ in range(args.episodes):
        returns.append(main(env, args))

    # Print the mean and std
    print("{:.2f} {:.2f}".format(np.mean(returns), np.std(returns)))
