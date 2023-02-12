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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import memory_game_environment
import wrappers

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=4, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=16, type=int, help="Number of episodes to train on.")
parser.add_argument("--gradient_clipping", default=1.0, type=float, help="Gradient clipping.")
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=1024, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--replay_buffer", default=None, type=int, help="Max replay buffer size; default batch_size")


def masked_sparse_categorical_crossentropy(y_true, y_pred, mask, sample_weight=None):
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    sample_weight = tf.boolean_mask(sample_weight, mask)
    return tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred, sample_weight=sample_weight)


def masked_categorical_crossentropy(y_true, y_pred, mask):
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        # Define the agent inputs: a memory and a state.
        memory = tf.keras.layers.Input(shape=[args.memory_cells, args.memory_cell_size], dtype=tf.float32)
        state = tf.keras.layers.Input(shape=env.observation_space.shape, dtype=tf.int32)

        # Encode the input state, which is a (card, observation) pair,
        # by representing each element as one-hot and concatenating them, resulting
        # in a vector of length `sum(env.observation_space.nvec)`.
        encoded_input = tf.keras.layers.Concatenate()(
            [tf.one_hot(state[:, i], dim) for i, dim in enumerate(env.observation_space.nvec)])

        # Generate a read key for memory read from the encoded input, by using
        # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
        # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
        # content in limited range).
        x = tf.keras.layers.Dense(args.hidden_layer, activation='relu')(encoded_input)
        read_key = tf.keras.layers.Dense(args.memory_cell_size, activation='tanh')(x)

        # Read the memory using the generated read key. Notably, compute cosine
        # similarity of the key and every memory row, apply softmax to generate
        # a weight distribution over the rows, and finally take a weighted average of
        # the memory rows.
        normalized_memory = tf.math.l2_normalize(memory, axis=-1)
        normalized_read_keys = tf.math.l2_normalize(read_key, axis=-1)
        matvec = tf.linalg.matvec(normalized_memory, normalized_read_keys)
        softmax = tf.nn.softmax(matvec, axis=-1)
        read_value = tf.linalg.matvec(memory, softmax, transpose_a=True)

        # Using concatenated encoded input and the read value, use a ReLU hidden
        # layer of size `args.hidden_layer` followed by a dense layer with
        # `env.action_space.n` units and `softmax` activation to produce a policy.
        policy = tf.keras.layers.Concatenate(axis=1)([encoded_input, read_value])
        policy = tf.keras.layers.Dense(args.hidden_layer, activation='relu')(policy)
        policy = tf.keras.layers.Dense(env.action_space.n, activation='softmax')(policy)
        policy = tf.squeeze(policy)  # QOL when calling with batch_size of 1.

        # Perform memory write. For faster convergence, append directly
        # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
        # the last memory row to keep memory size constant.
        updated_memory = tf.concat([tf.expand_dims(encoded_input, 1), memory[:, :-1]], axis=1)
        updated_memory = tf.squeeze(updated_memory)  # To avoid adding extra dimension when batch_size = 1

        # Create the agent
        self._agent = tf.keras.Model(inputs=[memory, state], outputs=[updated_memory, policy])
        self._agent.compile(
            optimizer=tf.optimizers.Adam(clipnorm=args.gradient_clipping),
            loss=masked_sparse_categorical_crossentropy,
        )

    def zero_memory(self):
        # Return an empty memory. It should be a TF tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]`.
        return tf.zeros(shape=[self.args.memory_cells, self.args.memory_cell_size])

    @tf.function
    def _train(self, states, actions, returns, episode_lengths, max_length):
        # Train the network given a batch of sequences of `states`
        # (each being a (card, symbol) pair), sampled `actions` and observed `returns`.
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `actions` as actions.
        #
        # Use the REINFORCE algorithm, optionally with a baseline. Note that
        # I use a baseline, but not a baseline computed by a neural network;
        # instead, for every time step, I track exponential moving average of
        # observed returns, with momentum 0.01. Furthermore, I use entropy regularization
        # with coefficient `args.entropy_regularization`.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        batch_size = len(states)
        memory = tf.stack([self.zero_memory() for _ in range(batch_size)])
        baseline = tf.reduce_mean(returns[:, 0])
        momentum = 0.01
        for step in range(max_length):
            state = states[:, step]
            action = actions[:, step]
            return_ = returns[:, step]
            mask = step < episode_lengths
            baseline_t = tf.reduce_mean(tf.boolean_mask(returns[:, step], mask))
            baseline = (1 - momentum) * baseline_t + momentum * baseline
            # baseline = momentum * baseline_t + (1 - momentum) * baseline
            with tf.GradientTape() as tape:
                memory, policy = self._agent([memory, state])
                reinforce_loss = self._agent.loss(action, policy, mask, sample_weight=return_ - baseline)
                entropy_loss = masked_categorical_crossentropy(policy, policy, mask)
                loss = reinforce_loss - self.args.entropy_regularization * entropy_loss
            grads = tape.gradient(loss, self._agent.trainable_variables)
            self._agent.optimizer.apply_gradients(zip(grads, self._agent.trainable_weights))

    def train(self, episodes):
        # Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        state_batches, action_batches, return_batches, episode_lengths = [], [], [], [len(e) - 1 for e in episodes]
        max_len = max(episode_lengths)

        for e in episodes:
            states, actions, returns = [], [], []
            for step in range(max_len):
                if step < len(e) - 1:  # Last action is None.
                    states.append(e[step][0])
                    actions.append(e[step][1])
                    returns.append(e[step][2])
                else:
                    states.append([-1, -1])
                    actions.append(-1)
                    returns.append(-1)
            state_batches.append(states)
            action_batches.append(actions)
            return_batches.append(returns)

        self._train(np.array(state_batches),
                    np.array(action_batches),
                    np.array(return_batches),
                    np.array(episode_lengths),
                    max_len)

    @wrappers.typed_np_function(np.float32, np.int32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, memory, state):
        return self._agent([memory, state])


def main(env, args):
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    if args.replay_buffer is None:
        args.replay_buffer = args.batch_size
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], network.zero_memory()
        rewards, done = 0, False
        while not done:
            # Find out which action to use
            memory, policy = network.predict([memory], [state])
            action = np.argmax(policy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Training
    replay_buffer = wrappers.ReplayBuffer(max_length=args.replay_buffer)
    training = True
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each):
            state, memory, episode, done = env.reset()[0], network.zero_memory(), [], False
            while not done:
                # Choose an action according to the generated distribution.
                memory, policy = network.predict([memory], [state])
                action = np.random.choice(env.action_space.n, p=policy)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode.append([state, action, reward])
                state = next_state

            # In the `episode`, compute returns from the rewards.
            # From the REINFORCE assignment.
            rewards = [t[2] for t in episode]
            gamma_exp = np.arange(len(rewards))[::-1]
            gammas = [0.99 ** gamma_exp[i] for i in range(len(rewards))]
            returns = [np.sum([gammas[k] * rewards[k] for k in range(t, len(rewards))]) for t in range(len(rewards))]
            for i, t in enumerate(episode):
                # Rewrite the reward with return.
                t[2] = returns[i]
            replay_buffer.append(episode)

            # Train the network if enough data is available
            if len(replay_buffer) >= args.batch_size:
                network.train(replay_buffer.sample(args.batch_size, np.random, replace=False))

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]
        # if np.mean(returns) - 1 * np.std(returns) > 0:
        if np.mean(returns) > 1:
                training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("MemoryGame-v0", cards=args.cards), args.seed, args.render_each,
                                 evaluate_for=args.evaluate_for, report_each=args.evaluate_for)

    main(env, args)
