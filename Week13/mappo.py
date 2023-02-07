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

import multi_collect_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--agents", default=2, type=int, help="Agents to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=16, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--clip_epsilon", default=0.15, type=float, help="Clipping epsilon.")
parser.add_argument("--entropy_regularization", default=0.05, type=float, help="Entropy regularization weight.")
parser.add_argument("--envs", default=12, type=int, help="Workers during experience collection.")
parser.add_argument("--epochs", default=8, type=int, help="Epochs to train each iteration.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each given number of iterations.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--trace_lambda", default=0.95, type=float, help="Traces factor lambda.")
parser.add_argument("--worker_steps", default=80, type=int, help="Steps for each worker to perform.")
parser.add_argument("--model_path", default='mappo_model_504', type=str, help="Path to a pre-trained model.")


# We use the exactly same Network as in the `ppo` assignment.
class Network(tf.keras.Model):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, args: argparse.Namespace) -> None:
        self.args = args

        # Create a suitable model for the given observation and action spaces.
        inputs = tf.keras.layers.Input(observation_space.shape)

        # Using a single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a policy with `action_space.n` discrete actions.
        policy = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        policy = tf.keras.layers.Dense(action_space.n, activation='softmax')(policy)

        # Using an independent single hidden layer with args.hidden_layer_size and ReLU activation,
        # produce a value function estimate. It is best to generate it as a scalar, not
        # a vector of length one, to avoid broadcasting errors later.
        value = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        value = tf.keras.layers.Dense(1)(value)
        value = tf.squeeze(value)

        # Construct the model
        super().__init__(inputs=inputs, outputs=[policy, value])

        # Compile using Adam optimizer with the given learning rate.
        self.compile(optimizer=tf.optimizers.Adam(args.learning_rate))

    # Define a training method `train_step`, which is automatically used by Keras.
    def train_step(self, data):
        # Unwrap the data. The targets is a dictionary of several tensors, containing keys
        # - "actions"
        # - "action_probs"
        # - "advantages"
        # - "returns"
        states, targets = data
        with tf.GradientTape() as tape:
            # Compute the policy and the value function
            policy, value = self(states, training=True)

            # Sum the following three losses
            # - the PPO loss, where `self.args.clip_epsilon` is used to clip the probability ratio
            # - the MSE error between the predicted value function and target regurns
            # - the entropy regularization with coefficient `self.args.entropy_regularization`.
            #   You can compute it for example using `tf.losses.CategoricalCrossentropy()`
            #   by realizing that entropy can be computed using cross-entropy.
            new_probs = tf.gather(policy, targets['actions'], batch_dims=1, axis=1)
            # Division with probabilities can lead to NaNs.
            ppo_ratios = tf.math.exp(tf.math.log(new_probs) - tf.math.log(targets['action_probs']))
            ppo_ratios_clipped = tf.clip_by_value(ppo_ratios, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon)
            ppo_loss = -tf.reduce_mean(tf.minimum(
                ppo_ratios * targets['advantages'],
                ppo_ratios_clipped * targets['advantages']
            ))

            critic_loss = tf.reduce_mean(tf.square(value - targets['returns']))

            entropy_loss = tf.keras.losses.CategoricalCrossentropy()(policy, policy)
            # The following is very numerically unstable, using CategoricalCrossentropy works better.
            # entropy_loss = -tf.reduce_mean(-tf.reduce_sum(policy * tf.math.log(policy), axis=1))

            loss = ppo_loss + critic_loss - args.entropy_regularization * entropy_loss

        # Perform an optimizer step and return the loss for reporting and visualization.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return {"loss": loss}

    # Predict method, with @wrappers.raw_tf_function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the networks, each for the same observation space and corresponding action space.
    networks = [Network(env.observation_space, env.action_space[i], args) for i in range(args.agents)]
    if args.model_path:
        for i in range(args.agents):
            filename = args.model_path + f'_{i}.h5'
            networks[i].load_weights(filename)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # Predict the vector of actions using the greedy policy
            action = []
            for i in range(args.agents):
                policy = networks[i].predict([state])[0]
                action.append(np.argmax(policy))

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Create the vectorized environment
    venv = gym.vector.make(env.spec.id, num_envs=args.envs, asynchronous=True, agents=args.agents)

    # Training
    state = venv.reset(seed=args.seed)[0]
    training = True if not args.recodex else False
    iteration = 0
    current_best = -np.inf
    try:
        while training:
            # Collect experience. Notably, we collect the following quantities
            # as tensors with the first two dimensions `[args.worker_steps, args.envs]`,
            # and the third dimension being `args.agents` for `action*`, `rewards`, `values`.
            states, actions, action_probs, rewards, dones, values = [], [], [], [], [], []
            for _ in range(args.worker_steps):
                # Choose `action`, which is a vector of `args.agents` actions for each worker,
                # each action sampled from the corresponding policy generated by the `predict` of the
                # networks executed on the vector `state`.
                action = []
                action_prob = []
                value = []
                for i in range(args.agents):
                    policy_i, value_i = networks[i].predict(state)
                    action_i = [np.random.choice(len(probs), p=probs) for probs in policy_i]
                    action_prob_i = policy_i[np.arange(len(action_i)), action_i]
                    action.append(action_i)
                    action_prob.append(action_prob_i)
                    value.append(value_i)

                # Transpose the lists
                action = list(map(list, zip(*action)))
                action_prob = list(map(list, zip(*action_prob)))
                value = list(map(list, zip(*value)))

                # Perform the step, extracting the per-agent rewards for training
                next_state, _, terminated, truncated, info = venv.step(action)
                reward = np.array([*info["agent_rewards"]])
                done = terminated | truncated

                # Collect the required quantities
                for collection, new_entry in zip([states, actions, action_probs, rewards, dones, values],
                                                 [state, action, action_prob, reward, done, value]):
                    collection.append(new_entry)

                state = next_state

            # Convert to NumPy arrays for easier indexing
            states, actions, action_probs, rewards, dones, values = (
                np.array(collection) for collection in [states, actions, action_probs, rewards, dones, values]
            )

            # Artificially mark the last state as done (truncated) for purposes of computing advantages.
            dones[-1] = True

            for a in range(args.agents):
                # For the given agent, estimate `advantages` and `returns` (they differ only by the value
                # function estimate) using lambda-return with coefficients `args.trace_lambda` and `args.gamma`.
                # You need to process episodes of individual workers independently, and note that
                # each worker might have generated multiple episodes, the last one probably unfinished.
                deltas = rewards[:, :, a] - values[:, :, a]
                deltas[:-1] += (1 - dones[:-1]) * args.gamma * values[1:, :, a]

                advantages = np.zeros((args.worker_steps, args.envs))
                for i in range(args.envs):
                    ep_end = np.argmax(dones[:, i])
                    for t in range(args.worker_steps):
                        coeffs = (args.gamma * args.trace_lambda) ** (np.arange(ep_end - t))
                        advantages[t, i] = np.dot(deltas[t:ep_end, i], coeffs)
                        if dones[t, i] and t != args.worker_steps - 1:  # No more data left.
                            ep_end = ep_end + np.argmax(dones[t + 1:, i]) + 1
                returns = advantages + values[:, :, a]

                # Train the agent `a` using the Keras API.
                # - The below code assumes that the first two dimensions of the used quantities are
                #   `[args.worker_steps, args.envs]` and concatenates them together.
                # - The code further assumes `actions` and `action_probs` have shape
                #   `[args.worker_steps, args.envs, args.agents]`, and uses only values of agent `a`.
                #   If you use a different shape, please update the code accordingly.
                # - We do not log the training by passing `verbose=0`; feel free to change it.
                networks[a].fit(
                    np.concatenate(states),
                    {"actions": np.concatenate(actions)[:, a],
                     "action_probs": np.concatenate(action_probs)[:, a],
                     "advantages": np.concatenate(advantages),
                     "returns": np.concatenate(returns)},
                    batch_size=args.batch_size, epochs=args.epochs, verbose=0,
                )
            # Periodic evaluation
            iteration += 1
            if iteration % args.evaluate_each == 0:
                scores = [evaluate_episode() for _ in range(args.evaluate_for)]
                scores_avg = np.mean(scores)
                if scores_avg > max(current_best, 0):
                    current_best = scores_avg
                    for i in range(args.agents):
                        model_name = f'mappo_model_{int(current_best)}_{i}.h5'
                        networks[i].save_weights(model_name)
    except KeyboardInterrupt:
        print('Keyboard interrupt detected, starting evaluation.')

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("MultiCollect-v0", agents=args.agents), args.seed, args.render_each)

    main(env, args)
