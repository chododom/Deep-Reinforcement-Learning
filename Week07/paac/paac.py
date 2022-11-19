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
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="CartPole-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=None, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=16, type=int, help="Number of parallel environments.")
parser.add_argument("--evaluate_each", default=500, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Similarly to reinforce with baseline, define two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # Use independent networks for both of them, each with
        # `args.hidden_layer_size` neurons in one ReLU hidden layer,
        # and train them using Adam with given `args.learning_rate`.
        inputs = tf.keras.Input(shape=env.observation_space.shape, name='Input')
        x = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        output = tf.keras.layers.Dense(env.action_space.n, activation='softmax')(x)
        actor_network = tf.keras.Model(inputs=inputs, outputs=output)
        actor_network.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                              optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

        inputs = tf.keras.Input(shape=env.observation_space.shape, name='Input')
        x = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        output = tf.keras.layers.Dense(1)(x)
        critic_network = tf.keras.Model(inputs=inputs, outputs=output)
        critic_network.compile(loss=tf.keras.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate))

        self._actor_network = actor_network
        self._critic_network = critic_network

    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # Train the policy network using policy gradient theorem
        # and the value network using MSE.
        #
        # The `args.entropy_regularization` might be used to include actor
        # entropy regularization -- however, the assignment can be solved
        # quite easily without it (my reference solution does not use it).
        # In any case, `tfp.distributions.Categorical` is the suitable distribution;
        # in PyTorch, it is `torch.distributions.Categorical`.

        # Train the actor network with advantages as sample weights.
        advantages = returns - tf.squeeze(self._critic_network(states))
        with tf.GradientTape() as actor_tape:
            y_true = actions
            y_pred = self._actor_network(states, training=True)
            actor_loss = self._actor_network.loss(y_true, y_pred, sample_weight=advantages)

        # Train the critic network to predict value function.
        with tf.GradientTape() as critic_tape:
            y_true = returns
            y_pred = tf.squeeze(self._critic_network(states, training=True))
            critic_loss = self._critic_network.loss(y_true, y_pred)

        self._actor_network.optimizer.minimize(actor_loss,
                                               self._actor_network.trainable_variables, tape=actor_tape)
        self._critic_network.optimizer.minimize(critic_loss,
                                               self._critic_network.trainable_variables, tape=critic_tape)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # Return predicted action probabilities.
        return self._actor_network(states)

    @wrappers.typed_np_function(np.float32)
    def sample_actions(self, probs: np.ndarray) -> np.ndarray:
        # Sample actions according to predicted action probabilities.
        actions = tf.random.categorical(np.log(probs), num_samples=1)
        return tf.squeeze(actions)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # Return estimates of value function.
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
            action = np.argmax(network.predict_actions([state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.make(args.env, args.envs, asynchronous=True)
    try:
        states = vector_env.reset(seed=args.seed)[0]
        training = True
        while training:
            # Training
            for _ in range(args.evaluate_each):
                # Choose actions using `network.predict_actions`.
                probs = network.predict_actions(states)
                actions = network.sample_actions(probs)

                # Perform steps in the vectorized environment
                next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
                dones = terminated | truncated

                # Compute estimates of returns by one-step bootstrapping
                estimated_returns = rewards + args.gamma * (1 - dones) * network.predict_values(next_states)

                # Train network using current states, chosen actions and estimated returns
                network.train(states, actions, estimated_returns)

                states = next_states

            # Periodic evaluation
            returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
            # print(np.mean(returns), '+-', np.std(returns))
            if np.mean(returns) - np.std(returns) > 450:
                training = False
    except KeyboardInterrupt:
        pass
    finally:
        vector_env.close()

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
