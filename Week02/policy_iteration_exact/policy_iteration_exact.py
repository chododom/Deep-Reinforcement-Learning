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

import numpy as np


class GridWorld:
    # States in the gridworld are the following:
    # 0 1 2 3
    # 4 x 5 6
    # 7 8 9 10

    # The rewards are +1 in state 10 and -100 in state 6

    # Actions are ↑ → ↓ ←; with probability 80% they are performed as requested,
    # with 10% move 90° CCW is performed, with 10% move 90° CW is performed.
    states: int = 11

    actions: list[str] = ["↑", "→", "↓", "←"]

    @staticmethod
    def step(state: int, action: int) -> list[tuple[float, float, int]]:
        return [GridWorld._step(0.8, state, action),
                GridWorld._step(0.1, state, (action + 1) % 4),
                GridWorld._step(0.1, state, (action + 3) % 4)]

    @staticmethod
    def _step(probability: float, state: int, action: int) -> tuple[float, float, int]:
        state += (state >= 5)
        x, y = state % 4, state // 4
        offset_x = -1 if action == 3 else action == 1
        offset_y = -1 if action == 0 else action == 2
        new_x, new_y = x + offset_x, y + offset_y
        if not (new_x >= 4 or new_x < 0 or new_y >= 3 or new_y < 0 or (new_x == 1 and new_y == 1)):
            state = new_x + 4 * new_y
        state -= (state >= 5)
        return (probability, +1 if state == 10 else -100 if state == 6 else 0, state)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--gamma", default=0.95, type=float, help="Discount factor.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--steps", default=5, type=int, help="Number of policy evaluation/improvements to perform.")
# If you add more arguments, ReCodEx will keep them with your default values.


def argmax_with_tolerance(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Argmax with small tolerance, choosing the value with smallest index on ties"""
    x = np.asarray(x)
    return np.argmax(x + 1e-6 >= np.max(x, axis=axis, keepdims=True), axis=axis)


def find_best_action(state, value_function, args):
    sums = []
    for action in range(len(GridWorld.actions)):
        sum_a = 0
        outcomes = GridWorld.step(state, action)

        for next_action in outcomes:
            probability = next_action[0]
            reward = next_action[1]
            next_state = next_action[2]

            sum_a += probability * (reward + args.gamma * value_function[next_state])

        sums.append(sum_a)

    return argmax_with_tolerance(sums)



def main(args: argparse.Namespace) -> tuple[list[float], list[int]]:
    # Start with zero value function and "go North" policy
    value_function = [0.0] * GridWorld.states
    policy = [0] * GridWorld.states

    # Implement policy iteration algorithm, with `args.steps` steps of
    # policy evaluation/policy improvement. During policy evaluation, compute
    # the value function exactly by solving the system of linear equations.
    # During the policy improvement, use the `argmax_with_tolerance` to
    # choose the best action.

    for step in range(args.steps):
        # init matrices
        I = np.eye(GridWorld.states)
        P = np.zeros((GridWorld.states,GridWorld.states))
        R = np.zeros(GridWorld.states)

        for st in range(GridWorld.states):
            outcomes = GridWorld.step(st, policy[st])

            for outcome in outcomes:
                prob, rew, new_st = outcome

                P[st, new_st] += prob
                R[st] += rew * prob


        value_function = np.linalg.solve(I-args.gamma*P, R)

        # Policy improvement
        for st in range(GridWorld.states):
            policy[st] = find_best_action(st, value_function, args)

    # The final value function should be in `value_function` and final greedy policy in `policy`.
    return value_function, policy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    value_function, policy = main(args)

    # Print results
    for r in range(3):
        for c in range(4):
            state = 4 * r + c
            state -= (state >= 5)
            print("        " if r == 1 and c == 1 else "{:-8.2f}".format(value_function[state]), end="")
            print(" " if r == 1 and c == 1 else GridWorld.actions[policy[state]], end="")
        print()