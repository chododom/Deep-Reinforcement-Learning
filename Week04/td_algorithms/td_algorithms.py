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
parser.add_argument("--alpha", default=0.1, type=float, help="Learning rate alpha.")
parser.add_argument("--episodes", default=10, type=int, help="Training episodes.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration epsilon factor.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor gamma.")
parser.add_argument("--mode", default="sarsa", type=str, help="Mode (sarsa/expected_sarsa/tree_backup).")
parser.add_argument("--n", default=1, type=int, help="Use n-step method.")
parser.add_argument("--off_policy", default=False, action="store_true", help="Off-policy; use greedy as target")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed.")
# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a fixed seed
    generator = np.random.RandomState(args.seed)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("Taxi-v3"), seed=args.seed, report_each=min(200, args.episodes))

    Q = np.zeros((env.observation_space.n, env.action_space.n))

    # The next action is always chosen in the epsilon-greedy way.
    def choose_next_action(Q: np.ndarray) -> tuple[int, float]:
        greedy_action = argmax_with_tolerance(Q[next_state])
        next_action = greedy_action if generator.uniform() >= args.epsilon else env.action_space.sample()
        return next_action, args.epsilon / env.action_space.n + (1 - args.epsilon) * (greedy_action == next_action)

    # The target policy is either the behavior policy (if not args.off_policy),
    # or the greedy policy (if args.off_policy).
    def compute_target_policy(Q: np.ndarray) -> np.ndarray:
        target_policy = np.eye(env.action_space.n)[argmax_with_tolerance(Q, axis=-1)]
        if not args.off_policy:
            target_policy = (1 - args.epsilon) * target_policy + args.epsilon / env.action_space.n
        return target_policy

    # Run the TD algorithm
    for _ in range(args.episodes):
        next_state, done = env.reset()[0], False

        T = np.inf
        t = 0
        tau = 0

        A = np.zeros(args.n + 1) # actions
        S = np.zeros(args.n + 1) # states 
        R = np.zeros(args.n + 1) # rewards
        b = np.zeros(args.n + 1) # behaviour

        # Generate episode and update Q using the given TD method
        
        next_action, next_action_prob = choose_next_action(Q)
        A[0] = next_action
        S[0] = next_state
        b[0] = next_action_prob

        while tau != T - 1:
            if t < T:
                # Take action A_t; observe and store the next reward and state
                action, action_prob, state = next_action, next_action_prob, next_state
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                S[(t + 1) % (args.n + 1)] = next_state
                R[(t + 1) % (args.n + 1)] = reward
                
                if not done:
                    # Choose an action A_t+1 arbitrarily and store it
                    next_action, next_action_prob = choose_next_action(Q)
                    A[(t + 1) % (args.n + 1)] = next_action
                    b[(t + 1) % (args.n + 1)] = next_action_prob
                else:
                    T = t + 1


            tau = t + 1 - args.n # tau is the time whose estimate is being updated
            if tau >= 0:
                
                # n-step Tree Backup:
                if args.mode == 'tree_backup':
                    if t + 1 >= T:
                        G = R[T % (args.n + 1)]
                    else:
                        sum_pi = 0
                        for a in range(env.action_space.n):                            
                            S_ind = S[(t + 1) % (args.n + 1)].astype(np.int32)
                            target_policy = compute_target_policy(Q)
                            sum_pi += target_policy[S_ind, a] * Q[S_ind, a]

                        G = R[(t + 1) % (args.n + 1)] + args.gamma * sum_pi
        

                    k = min(t, T - 1)
                    while k >= tau + 1:
                        A_k = A[k % (args.n + 1)].astype(np.int32)
                        sum_pi_except_Ak = 0
                        for a in range(env.action_space.n):
                            if a == A_k:
                                continue
                            S_ind = S[k % (args.n + 1)].astype(np.int32)
                            target_policy = compute_target_policy(Q)
                            sum_pi_except_Ak += target_policy[S_ind, a] * Q[S_ind, a] 
                            
                        G = R[k % (args.n + 1)] + args.gamma * sum_pi_except_Ak + args.gamma * target_policy[S_ind, A_k] * G
                        k -= 1
                    
                    S_tau = S[tau % (args.n + 1)].astype(np.int32)
                    A_tau = A[tau % (args.n + 1)].astype(np.int32)
                    Q[S_tau, A_tau] += args.alpha * (G - Q[S_tau, A_tau])
                
            
                # n-step SARSA:
                elif args.mode == 'sarsa':
                    prob = 1
                    if args.off_policy:
                        for i in range(tau + 1, min(tau + args.n, T - 1) + 1):
                            s_i = S[i % (args.n + 1)].astype(np.int32)
                            a_i = A[i % (args.n + 1)].astype(np.int32)
                            prob *= compute_target_policy(Q)[s_i, a_i] / b[i % (args.n + 1)]

                    G = 0
                    for i in range(tau + 1, min(tau + args.n, T) + 1):
                        G += args.gamma ** (i - tau - 1) * R[i % (args.n + 1)]

                    if tau + args.n < T:
                        s_tau_n = S[(tau + args.n) % (args.n + 1)].astype(np.int32)
                        a_tau_n = A[(tau + args.n) % (args.n + 1)].astype(np.int32)
                        G += (args.gamma ** args.n) * Q[s_tau_n, a_tau_n]

                    s_tau = S[tau % (args.n + 1)].astype(np.int32)
                    a_tau = A[tau % (args.n + 1)].astype(np.int32)
                    Q[s_tau, a_tau] += args.alpha * prob * (G - Q[s_tau, a_tau])

            t += 1
    return Q


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
