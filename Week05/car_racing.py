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
import os
import collections
import random
import shutil

import gym
import numpy as np
import tensorflow as tf

import car_racing_environment
import wrappers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--frame_skip", default=1, type=int, help="Frame skip.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epsilon", default=0.3, type=float, help="Exploration factor.")
parser.add_argument("--epsilon_final", default=0.02, type=float, help="Final exploration factor.")
parser.add_argument("--epsilon_final_at", default=100, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--target_update_freq", default=512, type=int, help="Target update frequency.")
parser.add_argument("--min_buffer_size", default=1000, type=int, help="Minimum replay buffer size before sampling.")
parser.add_argument("--max_buffer_size", default=100_000, type=int, help="Maximum replay buffer size.")
parser.add_argument("--model_path", default='model_547.zip', type=str, help="Path to a trained model.")


class Network:
    def __init__(self, num_actions: int, env, args):
        # Create a suitable model. The rest of the code assumes
        # it is stored as `self._model` and has been `.compile()`-d.
        # You can use `self._model.summary()` to see a summary of the model.
        model_path = args.model_path if 'model_path' in args else None
        if model_path is None:
            model = tf.keras.Sequential([
                tf.keras.layers.Input(env.observation_space.shape, batch_size=args.batch_size),
                tf.keras.layers.Conv2D(32, 8, 4, activation="relu"),
                # tf.keras.layers.Dropout(0.3),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, 4, 2, activation="relu"),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(64, 3, 1, activation="relu"),
                # tf.keras.layers.BatchNormalization(),
                # tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(num_actions)
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss=tf.keras.losses.MeanSquaredError()
            )
            model.summary()
        else:
            shutil.unpack_archive(model_path)
            model = tf.keras.models.load_model(os.path.splitext(model_path)[0])

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
    def train(self, states, q_values):
        self._model.optimizer.minimize(
            lambda: self._model.compiled_loss(q_values, self._model(states, training=True)),
            var_list=self._model.trainable_variables
        )

    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)

    # If you want to use target network, the following method copies weights from
    # a given Network to the current one.
    @tf.function
    def copy_weights_from(self, other):
        for var, other_var in zip(self._model.variables, other._model.variables):
            var.assign(other_var)

    # Compute greedy action.
    # @wrappers.typed_np_function(np.float32)
    # @tf.function
    def get_greedy_actions(self, states):
        action_value_function = self.predict(states)
        greedy_actions = np.argmax(action_value_function, axis=1)
        return greedy_actions, action_value_function[np.arange(len(greedy_actions)), greedy_actions]


def main(env, args):
    # Set random seeds and the number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    actions_discrete = np.array([
        [+0.0, 0.0, 0.0],  # Do nothing.
        [-1.0, 0.0, 0.0],  # Steer left.
        [+1.0, 0.0, 0.0],  # Steer right.
        # [-0.5, 0.0, 0.0],  # Steer left slightly.
        # [+0.5, 0.0, 0.0],  # Steer right slightly.
        [+0.0, 1.0, 0.0],  # Accelerate.
        [+0.0, 0.0, 0.8],  # Break.
        # [+0.0, 0.0, 0.4],  # Break slightly.
    ])
    num_actions = actions_discrete.shape[0]

    # Assuming you have pre-trained your agent locally, perform only evaluation in ReCodEx
    if args.recodex:
        network = Network(num_actions, env, args)
        # Final evaluation
        while True:
            state, done = env.reset(start_evaluation=True)[0], False
            while not done:
                # Choose a greedy action
                action_idx = network.get_greedy_actions([state])[0][0]
                state, reward, terminated, truncated, _ = env.step(actions_discrete[action_idx])
                done = terminated or truncated

    # Implement a suitable RL algorithm and train the agent.
    #
    # If you want to create N multiprocessing parallel environments, use
    # vector_env = gym.vector.make("CarRacingFS{}-v2".format(args.frame_skip), 4, asynchronous=True)
    # vector_env.reset(seed=args.seed) # The individual environments get incremental seeds

    # Construct the network
    network = Network(num_actions, env, args)

    target_network = Network(num_actions, env, args)
    target_network.copy_weights_from(network)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=args.max_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    training = True
    epsilon = args.epsilon
    episode = 0
    episode_returns = []
    best_mean_return = -np.inf
    steps_since_update = 0
    try:
        while training:
            episode += 1
            # Perform episode
            state, done = env.reset()[0], False
            episode_return = 0
            while not done:
                if np.random.uniform() < epsilon:
                    action_idx = np.random.randint(0, num_actions)
                else:
                    action_idx = network.get_greedy_actions([state])[0][0]  # Batch of 1.
                action = actions_discrete[action_idx]
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
                if done:
                    episode_returns.append(episode_return)

                # Append state, action, reward, done and next_state to replay_buffer
                replay_buffer.append(Transition(state, action_idx, reward, done, next_state))

                # If the replay_buffer is large enough, perform a training batch
                # from `args.batch_size` uniformly randomly chosen transitions.
                if len(replay_buffer) < args.min_buffer_size:
                    continue

                samples = random.sample(replay_buffer, args.batch_size)
                states = [s.state for s in samples]
                actions = [s.action for s in samples]
                rewards = np.asarray([s.reward for s in samples])
                dones = np.asarray([s.done for s in samples], dtype=np.float32)
                next_states = [s.next_state for s in samples]

                # next_state_q_values = target_network.get_greedy_actions(next_states)[1]
                # DDQN learning:
                next_state_q_values = target_network.predict(next_states)[
                    np.arange(len(samples)),
                    network.get_greedy_actions(next_states)[0]
                ]
                targets = rewards + (1 - dones) * args.gamma * next_state_q_values
                q_values = network.predict(states)
                q_values[np.arange(len(samples)), actions] = targets

                network.train(states, q_values)
                state = next_state

                if steps_since_update % args.target_update_freq == 0:
                    target_network.copy_weights_from(network)
                    steps_since_update = 0
                else:
                    steps_since_update += 1

            if args.epsilon_final_at:
                epsilon = np.interp(episode, [0, args.epsilon_final_at], [args.epsilon, args.epsilon_final])
            if len(episode_returns) >= 15:
                mean_return = np.mean(episode_returns[-15:])
                if mean_return > 300 and mean_return > best_mean_return:
                    best_mean_return = mean_return
                    model_name = f'model_{int(best_mean_return)}'
                    target_network._model.save(model_name)
                    shutil.make_archive(model_name, 'zip')
    except KeyboardInterrupt:
        pass
    if best_mean_return > 300:
        args.recodex = True
        args.model_path = f'model_{int(best_mean_return)}'
        main(env, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    assert args.min_buffer_size >= args.batch_size

    # Create the environment
    env = wrappers.EvaluationEnv(
        gym.make("CarRacingFS{}-v2".format(args.frame_skip)), args.seed, args.render_each,
        evaluate_for=15, report_each=1)

    main(env, args)
