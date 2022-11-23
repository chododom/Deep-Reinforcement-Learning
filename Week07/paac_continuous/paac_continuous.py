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
os.environ.setdefault("VERBOSE", "1")

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=16, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=16, type=int, help="Tiles to use.")
parser.add_argument("--embedding_dim", default=64, type=int, help="Dimension of the embedding layer.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Analogously to paac, your model should contain two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # The given states are tile encoded, so they are integral indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with with `args.tiles` ones).
        # (Or you can even use embeddings for better efficiency.)
        #
        # The actor computes `mus` and `sds`, each of shape `[batch_size, actions]`.
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and a ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `tf.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.

        inputs = tf.keras.layers.Input(shape=env.observation_space.shape)
        # x = tf.keras.layers.Embedding(input_dim=env.observation_space.nvec[-1],
        #                               output_dim=args.embedding_dim,
        #                               input_length=env.observation_space.shape[0])(inputs)
        # x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.CategoryEncoding(num_tokens=env.observation_space.nvec[-1])(inputs)
        x = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(x)
        mus = tf.keras.layers.Dense(env.action_space.shape[0])(x)
        mus = tf.math.tanh(mus)  # No further scaling necessary since the action's range is (-1, 1).
        sigmas = tf.keras.layers.Dense(env.action_space.shape[0])(x)
        sigmas = tf.math.softplus(sigmas)
        actor_network = tf.keras.Model(inputs=inputs, outputs=[mus, sigmas])
        actor_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        self._actor_network = actor_network

        # x = tf.keras.layers.Embedding(input_dim=env.observation_space.nvec[-1],
        #                               output_dim=args.embedding_dim,
        #                               input_length=env.observation_space.shape[0])(inputs)
        # x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.CategoryEncoding(num_tokens=env.observation_space.nvec[-1])(inputs)
        x = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(x)
        output = tf.keras.layers.Dense(1)(x)
        critic_network = tf.keras.Model(inputs=inputs, outputs=output)
        critic_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
        self._critic_network = critic_network

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # Run the model on given `states` and compute
        # `sds`, `mus` and predicted values. Then create `action_distribution` using
        # `tfp.distributions.Normal` class and the computed `mus` and `sds`.
        # In PyTorch, the corresponding class is `torch.distributions.Normal`.
        #
        # Train the actor using the sum of the following two losses:
        # - REINFORCE loss, i.e., the negative log likelihood of the `actions` in the
        #   `action_distribution` (using the `log_prob` method). You then need to sum
        #   the log probabilities of the action components in a single batch example.
        #   Finally, multiply the resulting vector by `(returns - predicted values)`
        #   and compute its mean. Note that the gradient must not flow through
        #   the predicted values (you can use `tf.stop_gradient` if necessary).
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.

        with tf.GradientTape() as actor_tape:
            mus, sigmas = self._actor_network(states)
            # distributions = tfp.distributions.TruncatedNormal(mus, sigmas, -1, 1)
            distributions = tfp.distributions.Normal(mus, sigmas)
            advantages = tf.stop_gradient(returns - tf.squeeze(self._critic_network(states, training=True)))
            log_likelihood = tf.squeeze(distributions.log_prob(actions))
            reinforce_loss = log_likelihood * advantages

            entropy_loss = args.entropy_regularization * distributions.entropy()

            actor_loss = -tf.reduce_mean(reinforce_loss + entropy_loss)
        actor_grads = actor_tape.gradient(actor_loss, self._actor_network.trainable_weights)
        self._actor_network.optimizer.apply_gradients(zip(actor_grads, self._actor_network.trainable_weights))

        with tf.GradientTape() as critic_tape:
            predicted_values = self._critic_network(states, training=True)
            critic_loss = tf.reduce_mean((predicted_values - returns)**2)
        critic_grads = critic_tape.gradient(critic_loss, self._critic_network.trainable_weights)
        self._critic_network.optimizer.apply_gradients(zip(critic_grads, self._critic_network.trainable_weights))

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Return predicted action distributions (mus and sds).
        return self._actor_network(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # Return predicted state-action values.
        return tf.squeeze(self._critic_network(states))


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # Predict the action using the greedy policy.
            mu, sigma = network.predict_actions([state])
            state, reward, terminated, truncated, _ = env.step(mu[0])
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.make("MountainCarContinuous-v0", args.envs, asynchronous=True,
                                 wrappers=lambda env: wrappers.DiscreteMountainCarWrapper(env, tiles=args.tiles))
    states = vector_env.reset(seed=args.seed)[0]

    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # Predict action distribution using `network.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            mus, sigmas = network.predict_actions(states)
            # distributions = tfp.distributions.TruncatedNormal(mus, sigmas, env.action_space.low, env.action_space.high)
            distributions = tfp.distributions.Normal(mus, sigmas)
            actions = distributions.sample().numpy()
            actions = np.clip(actions, env.action_space.low, env.action_space.high)

            # Perform steps in the vectorized environment
            next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
            dones = terminated | truncated

            # Compute estimates of returns by one-step bootstrapping
            estimated_returns = rewards + args.gamma * (1 - dones) * network.predict_values(next_states)

            # Train network using current states, chosen actions and estimated returns
            network.train(states, actions, estimated_returns)

            states = next_states

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(
        wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles),
        args.seed, args.render_each)

    main(env, args)
