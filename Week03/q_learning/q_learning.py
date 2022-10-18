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
parser.add_argument("--alpha", default=0.25, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.5, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    state_cnt = env.observation_space.n
    action_cnt = env.action_space.n

    # TODO: Variable creation and initialization
    Q = np.zeros(shape=(state_cnt, action_cnt))

    training = True
    iteration = 0
    episode_100_mean = 0
    while training:
        break # use pre-trained Q
        
        # Perform episode
        state, done = env.reset()[0], False
        while not done:
            # With probability of epislon, generate a random uniform action
            if np.random.uniform(low=0.0, high=1.0) < args.epsilon:
                action = env.action_space.sample()
            
            # Otherwise select the action with highest estimated reward in Q
            else:
                Q_action = Q[state] 
                action = np.argmax(Q_action)

            # Perform the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_100_mean += reward

            # Update the action-value estimates
            Q[state][action] = Q[state][action] + args.alpha * (reward + args.gamma * np.max(Q[next_state]) - Q[state][action])

            state = next_state              
        
        iteration += 1
        
        # Reset mean 100-episode return (and if it is good enough, end training)
        if iteration % 100 == 0:
            if episode_100_mean/100.0 > -140:
                training = False
            else:
                episode_100_mean = 0
        
        # Gradually decrease epsilon as we get closer to better solutions
        if iteration % 10 == 0 and args.epsilon > 0.05:
            if args.epsilon > 0.15:
                args.epsilon *= 0.9
            else:
                args.epsilon *= 0.95
        
        # Gradually adjust learning rate alpha to accelerate convergence but avoid overshooting the optimal action-value function
        if iteration % 1000 == 0 and args.alpha > 0.05:
            args.alpha *= 0.95
            
            

    #Q.dump('Q_dump')
    Q = np.load('Q_dump', allow_pickle=True)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Choose a greedy action
            Q_action = Q[state] 
            action = np.argmax(Q_action)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCar1000-v0", render_mode='human')), args.seed, args.render_each)

    main(env, args)