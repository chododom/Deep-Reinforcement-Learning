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
from __future__ import annotations

import argparse
import collections
import os
import random
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=1, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.1, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=100, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=32, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=16, type=int, help="Target update frequency.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace, model_path: str = None) -> None:
        # TODO: Create a suitable model. The rest of the code assumes
        # it is stored as `self._model` and has been `.compile()`-d.
        # You can use `self._model.summary()` to see a summary of the model.
        # print(env.action_space.n)
        if model_path is None:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(env.observation_space.shape, batch_size=args.batch_size),
                tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu),
                tf.keras.layers.Dense(env.action_space.n),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss=tf.keras.losses.MeanSquaredError()
            )
            model.summary()
        else:
            model = tf.keras.models.load_model(model_path)

        self._model = model

    # Define a training method. Generally you have two possibilities
    # - pass new q_values of all actions for a given state; all but one are the same as before
    # - pass only one new q_value for a given state, and include the index of the action to which
    #   the new q_value belongs
    # The code below implements the first option, but you can change it if you want.
    # Also note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, q_values: np.ndarray) -> None:
        self._model.optimizer.minimize(
            lambda: self._model.compiled_loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    @tf.function
    def copy_weights_from(self, other: Network) -> None:
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)

    def predict_q(self, state):
        return self.predict([state])[0]


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    target_network = Network(env, args)
    target_network.copy_weights_from(network)

    epsilon = args.epsilon
    iteration = 0
    episode_100_mean = 0
    training = args.recodex is False

    c = 0
    while training:
        # Perform episode
        state, done = env.reset()[0], False
        while not done:

            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(target_network.predict_q(state))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_100_mean += reward

            # Append state, action, reward, done and next_state to replay_buffer
            replay_buffer.append(Transition(state, action, reward, done, next_state))

            # If the replay_buffer is large enough, perform a training batch
            # from `args.batch_size` uniformly randomly chosen transitions.
            if len(replay_buffer) > args.batch_size:
                samples = random.sample(replay_buffer, args.batch_size)
                states = np.asarray([s.state for s in samples])
                actions = [s.action for s in samples]

                targets = [s.reward + (1 - s.done) * args.gamma * target_network.predict_q(s.next_state)[
                    np.argmax(network.predict_q(s.next_state))] for s in samples]

                q_values = network.predict(states)
                q_values[np.arange(len(actions)), actions] = targets

                # print(q_values)

                # After you choose `states` and suitable targets, you can train the network as
                network.train(states, q_values)

            state = next_state

            if c % args.target_update_freq:
                target_network.copy_weights_from(network)
            else:
                c = (c + 1) % args.target_update_freq

        if args.epsilon_final_at:
            epsilon = np.interp(env.episode + 1, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])

        # Reset mean 100-episode return (and if it is good enough, end training)
        if iteration % 100 == 0:
            if episode_100_mean / 100.0 > 470:
                network._model.save("model_470")
                training = False
            else:
                episode_100_mean = 0

        iteration += 1
        network._model.save("model")

    if args.recodex:
        with zipfile.ZipFile("model_470.zip", 'r') as zip_ref:
            zip_ref.extractall("./")
            target_network = Network(env, args, "model_470")

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Choose (greedy) action
            action = np.argmax(target_network.predict_q(state))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)
