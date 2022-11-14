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

import gym
import numpy as np
import tensorflow as tf

import wrappers

tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=50, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Create a suitable model.
        #
        # Apart from the policy network defined in `reinforce` assignment, you
        # also need a value network for computing the baseline (it can be for
        # example another independent model with a single hidden layer and
        # an output layer with a single output and no activation).
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
                
        
        # REINFORCE policy network
        inputs = tf.keras.Input(shape=env.observation_space.shape, name='Input')
        x = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu', name='Hidden-dense1')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(units=env.action_space.n, activation='softmax', name='Output')(x) # action space consists of movement to the left and to the right
        policy_model = tf.keras.Model(inputs=inputs, outputs=x, name="REINFORCE")
        
        policy_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(tf.keras.losses.Reduction.NONE), # get raw per-sample losses, which will then be weighted
                      optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
                      )
        
        # Baseline value network
        inputs = tf.keras.Input(shape=env.observation_space.shape, name='Input')
        y = tf.keras.layers.Dense(units=args.hidden_layer_size, activation='relu', name='Hidden-dense1')(inputs)
        y = tf.keras.layers.Dropout(0.3)(y)
        y = tf.keras.layers.Dense(units=1, activation=None,name='Output')(y)
        value_model = tf.keras.Model(inputs=inputs, outputs=y, name="Baseline")
        
        value_model.compile(loss=tf.keras.losses.MeanSquaredError(tf.keras.losses.Reduction.NONE),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
                      )
        
        
        self._model = policy_model
        self._baseline = value_model


    # Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:

        # Compute predicted baseline using the baseline model
        baseline_preds = tf.squeeze(self._baseline(states, training=True))
        delta = returns - baseline_preds
       
        # Train the policy model, using 'returns - predicted baseline' as advantage estimate
        with tf.GradientTape() as model_tape:
            y_true = actions
            y_pred = self._model(states, training=True)
            reinforce_loss = self._model.loss(y_true, y_pred, sample_weight=delta)
            
        self._model.optimizer.minimize(reinforce_loss, self._model.trainable_variables, tape=model_tape)
        
        # Train the baseline model to predict 'returns'
        with tf.GradientTape() as baseline_tape:
            y_true = returns
            y_pred = tf.squeeze(self._baseline(states, training=True))
            baseline_loss = self._baseline.loss(y_true, y_pred)
            
        self._baseline.optimizer.minimize(baseline_loss, self._baseline.trainable_variables, tape=baseline_tape)


    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)


    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []

        # Generate a training batch which consists of a number of episodes and their actions and overall returns
        
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset()[0], False
            
            while not done:
                # Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `network.predict` and current `state`.
            
                action = np.random.choice(env.action_space.n, p=network.predict([state]).flatten())

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # Compute returns by summing rewards (with discounting)

            returns = []
            gamma_exp = np.arange(len(rewards))[::-1]
            gammas = [args.gamma ** gamma_exp[i] for i in range(len(rewards))]

            """ Before efficiency optimization:
            
            for t in range(len(rewards)):
                G = 0
                for k in range(t, len(rewards)):
                    G += gammas[k] * rewards[k]
                returns.append(G)
            """
            
            returns = [np.sum([gammas[k] * rewards[k] for k in range(t, len(rewards))]) for t in range(len(rewards))]
    
            # Add states, actions and returns to the training batch
            # We need the batch to consist of a list of states on a "flat" level
            
            batch_states.extend(states) # batch_states += states
            batch_actions.extend(actions)
            batch_returns.extend(returns)
        
        network.train(batch_states, batch_actions, batch_returns)
            
        
    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True)[0], False
        while not done:
            # Choose a greedy action
            preds = network.predict([state]).flatten()
            action = np.argmax(preds)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPole-v1"), args.seed, args.render_each)

    main(env, args)