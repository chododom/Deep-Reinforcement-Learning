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
import collections
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="HalfCheetah-v4", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--envs", default=12, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=5000, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=512, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="cheetah_8252.model", type=str, help="Model path")
parser.add_argument("--resume_training_path", default=None, type=str, help="Model path")
parser.add_argument("--replay_buffer_size", default=1_000_000, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Create an actor. Because we will be sampling (and `sample()` from
        # `tfp.distributions` does not play nice with functional models) and because
        # we need the `alpha` variable, we use subclassing to create the actor.
        self.target_entropy = args.target_entropy * env.action_space.shape[0]
        self._log_alpha = tf.Variable(np.log(0.1), dtype=tf.float32)
        self._alpha_optimizer = tf.optimizers.Adam(args.learning_rate)

        class Actor(tf.keras.Model):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                # Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generating means with `env.action_space.shape[0]` units and no activation
                # - a layer for generating sds with `env.action_space.shape[0]` units and `tf.math.exp` activation
                # - finally, create a variable representing a logarithm of alpha, using for example the following:
                self.hidden_layers = [
                    tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu),
                    tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu)
                ]
                self.mean_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation=None)
                self.sd_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.math.exp)

            def call(self, inputs: tf.Tensor, sample: bool):
                # Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                # - Then, create the action distribution using `tfp.distributions.Normal`
                #   with the `mus` and `sds`. Note that to support computation without
                #   sampling, the easiest is to pass zeros as standard deviations when
                #   `sample == False`.
                # - We then bijectively modify the distribution so that the actions are
                #   in the given range. Luckily, `tfp.bijectors` offers classes that
                #   can transform a distribution.
                #   - first run
                #       tfp.bijectors.Tanh()(actions_distribution)
                #     to squash the actions to [-1, 1] range,
                #   - then run
                #       tfp.bijectors.Scale((env.action_space.high - env.action_space.low) / 2)(actions_distribution)
                #     to scale the action ranges to [-(high-low)/2, (high-low)/2],
                #   - finally,
                #       tfp.bijectors.Shift((env.action_space.high + env.action_space.low) / 2)(actions_distribution)
                #     shifts the ranges to [low, high].
                #   In case you wanted to do this manually, sample from a normal distribution, pass the samples
                #   through the `tanh` and suitable scaling, and then compute the log-prob by using `log_prob`
                #   from the normal distribution and manually accounting for the `tanh` as shown in the slides.
                #   However, the formula from the slides is not numerically stable, for a better variant see
                #   https://github.com/tensorflow/probability/blob/ef1f64a434/tensorflow_probability/python/bijectors/tanh.py#L70-L81
                # - Sample the actions by a `sample()` call.
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                # - Return actions, log_prob, and alpha.
                x = inputs
                for hidden_layer in self.hidden_layers:
                    x = hidden_layer(x)
                mus = self.mean_layer(x)
                sds = self.sd_layer(x)
                if sample:
                    action_dist = tfp.distributions.Normal(loc=mus, scale=sds)
                else:
                    action_dist = tfp.distributions.Normal(loc=mus, scale=tf.zeros_like(sds))
                action_dist = tfp.bijectors.Tanh()(action_dist)
                action_dist = tfp.bijectors.Scale((env.action_space.high - env.action_space.low) / 2)(action_dist)
                action_dist = tfp.bijectors.Shift((env.action_space.high + env.action_space.low) / 2)(action_dist)
                actions = action_dist.sample()
                log_probs = action_dist.log_prob(actions)
                log_probs = tf.reduce_mean(log_probs, axis=1)
                # alpha = tf.math.exp(self.log_alpha)
                # return actions, log_probs, alpha
                return actions, log_probs

        # Instantiate the actor as `self._actor` and compile it.
        self._actor = Actor(args.hidden_layer_size)
        self._actor.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

        # Create a critic, which
        # - takes observations and actions as inputs,
        # - concatenates them,
        # - passes the result through two dense layers with `args.hidden_layer_size` units
        #   and ReLU activation,
        # - finally, using a last dense layer produces a single output with no activation
        # This critic needs to be cloned so that two critics and two target critics are created.
        observations = tf.keras.Input(shape=env.observation_space.shape)
        actions = tf.keras.Input(shape=env.action_space.shape)
        x = tf.keras.layers.Concatenate()([observations, actions])
        x = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(x)
        output = tf.keras.layers.Dense(1)(x)
        critic = tf.keras.Model(inputs=[observations, actions], outputs=output)
        self._critic1 = critic
        self._critic2 = tf.keras.models.clone_model(critic)

        self._target_critic1 = tf.keras.models.clone_model(critic)
        self._target_critic2 = tf.keras.models.clone_model(critic)

        self._critic1.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                              loss=tf.keras.losses.MeanSquaredError())
        self._critic2.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                              loss=tf.keras.losses.MeanSquaredError())

    def save_actor(self, path: str):
        # Because we use subclassing for creating the actor, the easiest way of
        # serializing an actor is just to save weights.
        self._actor.save_weights(path, save_format="h5")

    def load_actor(self, path: str, env: wrappers.EvaluationEnv):
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        self.predict_mean_actions([env.observation_space.sample()])
        self._actor.load_weights(path)

    def touch_actor(self):
        for var in self._actor.trainable_weights:
            var.assign(np.random.normal(1, 0.001) * var)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # Separately train:
        # - the actor, by using two objectives:
        #   - the objective for the actor itself; in this objective, `tf.stop_gradient(alpha)`
        #     should be used (for the `alpha` returned by the actor) to avoid optimizing `alpha`,
        #   - the objective for `alpha`, where `tf.stop_gradient(log_prob)` should be used
        #     to avoid computing gradient for other variables than `alpha`.
        #     Use `args.target_entropy` as the target entropy (the default of -1 per action
        #     component is fine and does not need to be tuned for the agent to train).
        # - the critics using MSE loss.
        #
        # Finally, update the two target critic networks exponential moving
        # average with weight `args.target_tau`, using something like
        #   for var, target_var in zip(critic.trainable_variables, target_critic.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        with tf.GradientTape() as critic1_tape:
            predicted_values = self._critic1([states, actions])
            critic1_loss = tf.keras.losses.MeanSquaredError()(returns, predicted_values)
        critic1_grads = critic1_tape.gradient(critic1_loss, self._critic1.trainable_weights)
        self._critic1.optimizer.apply_gradients(zip(critic1_grads, self._critic1.trainable_weights))

        with tf.GradientTape() as critic2_tape:
            predicted_values = self._critic2([states, actions])
            critic2_loss = tf.keras.losses.MeanSquaredError()(returns, predicted_values)
        critic2_grads = critic2_tape.gradient(critic2_loss, self._critic2.trainable_weights)
        self._critic2.optimizer.apply_gradients(zip(critic2_grads, self._critic2.trainable_weights))

        alpha = tf.math.exp(self._log_alpha)
        with tf.GradientTape() as actor_tape:
            act, log_probs = self._actor(states, sample=True)
            value = tf.minimum(
                self._critic1([states, act]),
                self._critic2([states, act])
            )
            actor_loss = tf.reduce_mean(alpha * log_probs - value)
        actor_grads = actor_tape.gradient(actor_loss, self._actor.trainable_weights)
        self._actor.optimizer.apply_gradients(zip(actor_grads, self._actor.trainable_weights))

        with tf.GradientTape() as alpha_tape:
            alpha = tf.math.exp(self._log_alpha)
            act, log_probs = self._actor(states, sample=True)
            alpha_loss = tf.reduce_mean(-alpha * (log_probs + self.target_entropy))
        alpha_grads = alpha_tape.gradient(alpha_loss, [self._log_alpha])
        self._alpha_optimizer.apply_gradients(zip(alpha_grads, [self._log_alpha]))

        for var, target_var in zip(self._critic1.trainable_variables, self._target_critic1.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)
        for var, target_var in zip(self._critic2.trainable_variables, self._target_critic2.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)

    # Predict actions without sampling.
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_mean_actions(self, states: np.ndarray) -> np.ndarray:
        # Return predicted actions, assuming the actor is in `self._actor`.
        return self._actor(states, sample=False)[0]

    # Predict actions with sampling.
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_sampled_actions(self, states: np.ndarray) -> np.ndarray:
        # Return predicted actions, assuming the actor is in `self._actor`.
        return self._actor(states, sample=True)[0]

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # Produce the predicted returns, which are the minimum of
        #    target_critic(s, a) - alpha * log_prob
        #  considering both target critics and actions sampled from the actor.
        actions, log_probs = self._actor(states, sample=True)
        alpha = tf.math.exp(self._log_alpha)
        return tf.minimum(
            tf.squeeze(self._target_critic1([states, actions])) - alpha * log_probs,
            tf.squeeze(self._target_critic2([states, actions])) - alpha * log_probs
        )


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
            action = network.predict_mean_actions([state])[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        network.load_actor(args.model_path, env)
        while True:
            evaluate_episode(True)
    if args.resume_training_path:
        network.load_actor(args.resume_training_path, env)

    # Create the asynchroneous vector environment for training.
    venv = gym.vector.make(args.env, args.envs, asynchronous=True)

    # Replay memory of a specified maximum size.
    replay_buffer = wrappers.ReplayBuffer(max_length=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state, training = venv.reset(seed=args.seed)[0], True
    current_best = -np.inf
    while training:
        for _ in range(args.evaluate_each):
            # Predict actions by calling `network.predict_sampled_actions`.
            action = network.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = venv.step(action)
            done = terminated | truncated
            for i in range(args.envs):
                replay_buffer.append(Transition(state[i], action[i], reward[i], done[i], next_state[i]))
            state = next_state

            # Training
            if len(replay_buffer) >= 4 * args.batch_size:
                # Randomly uniformly sample transitions from the replay buffer.
                batch = replay_buffer.sample(args.batch_size, np.random)
                states, actions, rewards, dones, next_states = map(np.array, zip(*batch))
                # Perform the training.
                returns = rewards + (1 - dones) * args.gamma * network.predict_values(next_states)
                network.train(states, actions, returns)
        # Periodic evaluation
        scores = [evaluate_episode() for _ in range(args.evaluate_for)]
        score = np.mean(scores)
        print(f'Average return: {score}')
        if score > 8000:
            model_path = args.model_path + f'_{int(score)}'
            network.save_actor(model_path)
            print(f'Saved model at "{model_path}".')

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
