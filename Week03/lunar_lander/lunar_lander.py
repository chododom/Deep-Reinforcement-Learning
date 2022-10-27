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
parser.add_argument("--alpha", default=0.2, type=float, help="Learning rate.")
parser.add_argument("--epsilon", default=0.1, type=float, help="Exploration factor.")
parser.add_argument("--gamma", default=1, type=float, help="Discounting factor.")
parser.add_argument("--episodes", default=1000, type=float, help="Number of episodes.")
parser.add_argument("--n", default=5, type=float, help="Number of steps in n-step TD algorithm.")
parser.add_argument("--off_policy", default=False, type=float, help="Whether to use off or on policy.")


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


class RemainderList:
    def __init__(self, n):
        self._n = n
        self._list = [None] * (self._n + 1)

    def __getitem__(self, item):
        return self._list[item % (self._n + 1)]

    def __setitem__(self, key, value):
        self._list[key % (self._n + 1)] = value

    def __str__(self):
        return str(self._list)

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seed
    np.random.seed(args.seed)
    
    generator = np.random.RandomState(args.seed)
    
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
    
    
    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    args.recodex = False
    if args.recodex:
        # Load the agent
        Q_lander = np.load('Q_lander.npy')
        
        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # Choose a greedy action
                Q_action = Q_lander[state]
                action = np.argmax(Q_action)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated            

    else:
        Q = np.load('Q_lander.npy')
        use_expert = False
        from tqdm import trange
        for j in trange(args.episodes):
            
            if j % 3 == 0:
                use_expert = True
            
            next_state, done = env.reset()[0], False

            T = np.inf # terminal state time
            t = 0
            tau = 0

            if not use_expert:
                A = np.zeros(args.n + 1) # actions
                S = np.zeros(args.n + 1) # states 
                R = np.zeros(args.n + 1) # rewards
                b = np.zeros(args.n + 1) # behaviour
                
                # Generate episode and update Q using the given TD method
                
                next_action, next_action_prob = choose_next_action(Q)
                A[0] = next_action
                S[0] = next_state
                b[0] = next_action_prob
            else:
                A = []
                S = []
                R = []
                
                exp_trajectory = env.expert_trajectory()
                for triple in exp_trajectory:
                    A.append(triple[1])
                    S.append(triple[0])
                    R.append(triple[2])
                    

            while tau != T - 1:
                # If no expert trajectory is provided, generate episode and keep storing states
                if not use_expert:
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
                            
                # If expert trajectory is provided, there is no need to generate the episodes, we can just set the time variables to their final values
                else:
                    if t < T:
                        t = len(R)
                        T = t + 1

                tau = t + 1 - args.n # tau is the time whose estimate is being updated
                if tau >= 0:
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
                        A_k = int(A[k % (args.n + 1)])
                        sum_pi_except_Ak = 0
                        for a in range(env.action_space.n):
                            if a == A_k:
                                continue
                            S_ind = S[k % (args.n + 1)].astype(np.int32)
                            target_policy = compute_target_policy(Q)
                            sum_pi_except_Ak += target_policy[S_ind, a] * Q[S_ind, a] 
                            
                        G = R[k % (args.n + 1)] + args.gamma * sum_pi_except_Ak + args.gamma * target_policy[S_ind, A_k] * G
                        k -= 1
                    
                    S_tau = int(S[tau % (args.n + 1)])
                    A_tau = int(A[tau % (args.n + 1)])
                    Q[S_tau, A_tau] += args.alpha * (G - Q[S_tau, A_tau])
                t += 1
            
            if use_expert:
                use_expert = False
                
            # Gradually adjust learning rate alpha to accelerate convergence without overshooting the optimal Q function
            if j % 50 == 0 and args.alpha > 0.08:
                args.alpha *= 0.95
            
            np.save('Q_lander', Q)
        
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
        wrappers.DiscreteLunarLanderWrapper(gym.make("LunarLander-v2")), args.seed, args.render_each)

    main(env, args)