#!/usr/bin/env python3
#
# Team members:
# Dominik Chodounský
# Martin Lank
# Juraj Kmec
#
# ReCodExIDs:
# 882a1f6f-99a2-48df-aee6-1b62d6d0d2df
# b503f10b-77cf-41be-a787-371a69cfa66a
# 8c8b5f62-9f3e-4825-9966-185987537e3f
import argparse

import gym
import numpy as np

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.7, type=float, help="Learning rate.")
parser.add_argument("--alpha_final", default=0.05, type=float, help="Final learning rate.")
parser.add_argument("--alpha_final_at", default=2000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.6, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.05, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=1500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.985, type=float, help="Discounting factor.")
parser.add_argument("--tiles", default=8, type=int, help="Number of tiles.")
parser.add_argument("--episodes", default=4000, type=int, help="Number of episodes to train for.")
parser.add_argument("--use_pretrained", default=True, action="store_true", help="Load pre-trained model.")


def greedy_action(state, W):
    return np.argmax(np.sum(W[state], axis=0))


def q_hat(state, action, W):
    return np.sum(W[state, action])


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Implement Q-learning RL algorithm, using linear approximation.
    W = np.zeros([env.observation_space.nvec[-1], env.action_space.n])
    epsilon = args.epsilon
    alpha = args.alpha / args.tiles

    episode = 0

    # training = True
    # while training:
    for _ in range(args.episodes):
        if args.use_pretrained:
            break
        episode += 1
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # Choose an action.
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(state, W)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update the action-value estimates
            if done:
                W[state, action] += alpha * (reward - q_hat(state, action, W))
            else:
                next_state_q_hat = q_hat(next_state, greedy_action(next_state, W), W)
                W[state, action] += alpha * (reward + args.gamma * next_state_q_hat - q_hat(state, action, W))
                state = next_state

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
        
        if args.alpha_final_at:
            alpha = np.interp(env.episode + 1, [0, args.alpha_final_at], [args.alpha, args.alpha_final])
            alpha /= args.tiles
                
    if args.use_pretrained:
        W = np.load('W_dump.npy', allow_pickle=True)
    else:
        W.dump('W_dump.npy')


    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Choose (greedy) action
            action = greedy_action(state, W)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0"), tiles=args.tiles),
                                 args.seed, args.render_each)

    main(env, args)
