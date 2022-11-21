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

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="Pendulum-v1", type=str, help="Environment.")
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=64, type=int, help="Size of hidden layer.")
parser.add_argument("--actor_lr", default=0.001, type=float, help="Learning rate of actor.")
parser.add_argument("--critic_lr", default=0.003, type=float, help="Learning rate of critic.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Create:
        # - an actor, which starts with states and returns actions.
        #   Usually, one or two hidden layers are employed. As in the
        #   paac_continuous, to keep the actions in the required range, you
        #   should apply properly scaled `tf.tanh` activation.
        #
        # - a target actor as the copy of the actor using `tf.keras.models.clone_model`.
        #
        # - a critic, starting with given states and actions, producing predicted
        #   returns. The states and actions are usually concatenated and fed through
        #   two more hidden layers, before computing the returns with the last output layer.
        #
        # - a target critic as the copy of the critic using `tf.keras.models.clone_model`.
    
        input_states = tf.keras.layers.Input(env.observation_space.shape[0])
        actor = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(input_states)
        actor = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(actor)
        output_actor = tf.keras.layers.Dense(1, activation='tanh')(actor)
        output_actor = tf.multiply(output_actor, 2) # rescale tanh to range -2. : 2.
        self.actor = tf.keras.Model(input_states, output_actor)
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(args.actor_lr))
        
        self.target_actor= tf.keras.models.clone_model(self.actor)
        
        states_critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(input_states)
        actions_critic = tf.keras.layers.Input(1)
        critic = tf.keras.layers.concatenate([states_critic, actions_critic])
        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(critic)
        critic = tf.keras.layers.Dense(1)(critic)
        self.critic = tf.keras.Model(inputs=[input_states, actions_critic], outputs=critic)
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        
        self.target_critic = tf.keras.models.clone_model(self.critic)
        

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # Separately train:
        # - the actor using the DPG loss,
        # - the critic using MSE loss.
        
        # Train actor using DPG loss
        with tf.GradientTape() as atape:
            critic_value = self.critic([states, self.actor(states, training=True)], training=True)
            # Invert sign as we want to maximize the value given by the critic for our actions 
            dpg_loss = -tf.math.reduce_mean(critic_value)
        actor_grads = atape.gradient(dpg_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Train critic using MSE loss
        with tf.GradientTape() as ctape:
            critic_value = self.critic([states, actions], training=True)
            mse_loss = self.critic.loss(returns, critic_value)
        self.critic.optimizer.minimize(mse_loss, self.critic.trainable_variables, tape=ctape)
        
        """ Equivalent to:
        self.critic.optimizer.minimize(
            lambda: self.critic.loss(returns, self.critic([states, actions], training=True)),
            var_list=self.critic.trainable_variables
        )
        """
        
        # Furthermore, update the target actor and critic networks by
        # exponential moving average with weight `args.target_tau`. A possible
        # way to implement it inside a `tf.function` is the following:
        #   for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        
        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)  

    
    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # Return predicted actions by the actor.
        return self.actor(states)

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # Return predicted returns -- predict actions by the target actor
        # and evaluate them using the target critic.
        return self.target_critic([states, self.target_actor(states)])
    

class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


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

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        rewards, state, done = 0, env.reset(start_evaluation=start_evaluation, logging=logging)[0], False
        while not done:
            # Predict the action using the greedy policy.
            action = network.target_actor(np.asarray([state]))[0]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    if args.recodex:
        print("Eval:")
        
        network.actor = tf.keras.models.load_model('actor-156.h5')
        network.target_actor = tf.keras.models.load_model('actor-156.h5')
        network.critic = tf.keras.models.load_model('critic-156.h5')
        network.target_critic = tf.keras.models.load_model('critic-156.h5')
        
        while True:
            evaluate_episode(start_evaluation=True)
        
        # For local evaluation of pretrained model
        #returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        #mean_returns = np.mean(returns)
        #print("Evaluation after episode {}: {:.2f}".format(env.episode, mean_returns))


    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = True
    best_returns = -np.inf
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset()[0], False
            noise.reset()
            while not done:
                # Predict actions by calling `network.predict_actions`
                # and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                # clip the actions to the `env.action_space.{low,high}` range.
                action = network.predict_actions([state])[0] + noise.sample()
                action = np.clip(action, -2.0, 2.0) # env.action_space.low/high

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) < args.batch_size:
                    continue
                batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                # Perform the training
                returns = rewards + args.gamma * network.predict_values(next_states).flatten()             
                network.train(states, actions, returns)

        # Periodic evaluation
        returns = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
        mean_returns = np.mean(returns)
        print("Evaluation after episode {}: {:.2f}".format(env.episode, mean_returns))
        if mean_returns >= best_returns:
            best_returns = mean_returns
            actor_name = f"actor{int(best_returns)}.h5"
            critic_name = f"critic{int(best_returns)}.h5"
            network.target_actor.save(actor_name)
            network.target_critic.save(critic_name)


    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env, render_mode='human'), args.seed, args.render_each)

    main(env, args)
