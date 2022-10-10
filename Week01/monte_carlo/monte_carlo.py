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
# #TODO

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
parser.add_argument("--episodes", default=5000, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=1, type=float, help="Discount factor.")


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # - Create Q, a zero-filled NumPy array with shape [number of states, number of actions],
    #   representing estimated Q value of a given (state, action) pair.
    # - Create C, a zero-filled NumPy array with the same shape,
    #   representing number of observed returns of a given (state, action) pair.
    
    state_cnt = env.observation_space.n
    action_cnt = env.action_space.n
    
    Q = np.zeros(shape=(state_cnt, action_cnt)) # return estimates
    C = np.zeros(shape=(state_cnt, action_cnt)) # observed returns

    state, done = env.reset()[0], False
    
    for _ in range(args.episodes):
        # Perform an episode, collecting states, actions and rewards.
        episode_collection = []

        state, done = env.reset()[0], False
        G = 0
        while not done:
            # Compute `action` using epsilon-greedy policy.
            
            # With probability of epison, generate a random uniform action
            if np.random.uniform(low=0.0, high=1.0) < args.epsilon:
                action = env.action_space.sample()
            
            # Otherwise select the action with highest estimated reward in Q
            else:
                Q_action = Q[state] 
                action = np.argmax(Q_action)
                
            # Perform the action.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Collect states, actions and rewards.
            episode_collection.append([state, action, reward])
            
            state = next_state
        
        # Update estimates and observations.
        for state, action, reward in episode_collection[::-1]:            
            G = args.gamma * G + reward
            C[state][action] += 1
            Q[state][action] = Q[state][action] + (1.0/C[state][action])*(G - Q[state][action])
            
        

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
        wrappers.DiscreteCartPoleWrapper(gym.make("CartPole-v1")), args.seed, args.render_each)

    main(env, args)
