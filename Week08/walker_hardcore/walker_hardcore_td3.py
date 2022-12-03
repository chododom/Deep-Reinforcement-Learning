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
parser.add_argument("--env", default="BipedalWalkerHardcore-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--actor_lr", default=0.001, type=float, help="Learning rate of actor.")
parser.add_argument("--critic_lr", default=0.001, type=float, help="Learning rate of critic.")
parser.add_argument("--noise_sigma", default=0.1, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.2, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--use_pretrain", default=False, type=bool, help="Whether to use pretrained model.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        
        # Actor network        
        input_states = tf.keras.layers.Input(env.observation_space.shape)
        actor = tf.keras.layers.Dense(512, activation='relu')(input_states)
        actor = tf.keras.layers.Dense(256, activation='relu')(input_states)
        #actor = tf.keras.layers.Dense(128, activation='relu')(actor)
        output_actor = tf.keras.layers.Dense(env.action_space.shape[0], activation='tanh')(actor)
        self.actor = tf.keras.Model(input_states, output_actor)
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(args.actor_lr))
        
        self.target_actor= tf.keras.models.clone_model(self.actor)
        
        # Critic network
        input_states_critic = tf.keras.layers.Input(env.observation_space.shape)
        actions_critic = tf.keras.layers.Input(env.action_space.shape)
        critic = tf.keras.layers.concatenate([input_states_critic, actions_critic])
        critic = tf.keras.layers.Dense(512, activation='relu')(critic)
        critic = tf.keras.layers.Dense(256, activation='relu')(critic)
        #critic = tf.keras.layers.Dense(128, activation='relu')(critic)
        critic = tf.keras.layers.Dense(1)(critic)
        self.critic = tf.keras.Model(inputs=[input_states_critic, actions_critic], outputs=critic)
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        
        self.target_critic = tf.keras.models.clone_model(self.critic)
        
        # Second critic network
        input_states_critic2 = tf.keras.layers.Input(env.observation_space.shape)
        actions_critic2 = tf.keras.layers.Input(env.action_space.shape)
        critic2 = tf.keras.layers.concatenate([input_states_critic2, actions_critic2])
        critic2 = tf.keras.layers.Dense(512, activation='relu')(critic2)
        critic2 = tf.keras.layers.Dense(256, activation='relu')(critic2)
        #critic2 = tf.keras.layers.Dense(128, activation='relu')(critic2)
        critic2 = tf.keras.layers.Dense(1)(critic2)
        self.critic2 = tf.keras.Model(inputs=[input_states_critic2, actions_critic2], outputs=critic2)
        self.critic2.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        
        self.target_critic2 = tf.keras.models.clone_model(self.critic2)
        
        
    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:

        # Train actor using DPG loss
        with tf.GradientTape() as atape:
            critic_value = self.critic([states, self.actor(states, training=True)], training=True)
            # Invert sign as we want to maximize the value given by the critic for our actions 
            dpg_loss = -tf.math.reduce_mean(critic_value)
        actor_grads = atape.gradient(dpg_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Train critics using MSE loss
        with tf.GradientTape() as ctape:
            critic_value = self.critic([states, actions], training=True)
            mse_loss = self.critic.loss(returns, critic_value)
        self.critic.optimizer.minimize(mse_loss, self.critic.trainable_variables, tape=ctape)
        
        with tf.GradientTape() as ctape2:
            critic_value2 = self.critic2([states, actions], training=True)
            mse_loss2 = self.critic2.loss(returns, critic_value2)
        self.critic2.optimizer.minimize(mse_loss2, self.critic2.trainable_variables, tape=ctape2)
        
        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)
        for var, target_var in zip(self.critic2.trainable_variables, self.target_critic2.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)
    
    def compile_nets(self):
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(args.actor_lr))   
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(args.actor_lr))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        self.critic2.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        self.target_critic2.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())

    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # Return predicted actions by the actor.
        return self.actor(states)

    @wrappers.typed_np_function(np.float32)
    #@wrappers.raw_tf_function(dynamic_dims=1) Has to be commented out for the random function to work for some reason
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # Return predicted returns -- predict actions by the target actor
        # and evaluate them using the target critics.
        actor = self.target_actor(states)
        
        rng = np.random.normal(loc=0, scale=args.noise_sigma, size=actor.shape)
        noise = np.clip(rng, -0.6, 0.6)
        
        critic = self.target_critic([states, actor + noise]) 
        critic2 = self.target_critic2([states, actor + noise])
        return tf.minimum(critic, critic2)
    

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
        prefix = 'td3__'
        number = 261
        network.actor = tf.keras.models.load_model(f"{prefix}actor{number}.h5")
        network.critic = tf.keras.models.load_model(f"{prefix}critic{number}.h5")
        network.critic2 = tf.keras.models.load_model(f"{prefix}2critic{number}.h5")
        
        network.target_actor = tf.keras.models.load_model(f"{prefix}tactor{number}.h5")
        network.target_critic2 = tf.keras.models.load_model(f"{prefix}tcritic{number}.h5")
        network.target_critic2 = tf.keras.models.load_model(f"{prefix}2tcritic{number}.h5")
        
        while True:
            evaluate_episode(start_evaluation=True)


    else:
        noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
        training = True
        best_returns = -np.inf
        while training:
            # Training
            for _ in range(args.evaluate_each):
                state, done = env.reset()[0], False
                noise.reset()
                t_step = 0
                while not done:
                    action = network.predict_actions([state])[0] + noise.sample()
                    action = np.clip(action, env.action_space.low, env.action_space.high)
    
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    reward = 0 if reward == -100 else reward
                    
                    replay_buffer.append(Transition(state, action, reward, done, next_state))
                    state = next_state
                    t_step += 1
    
                    if len(replay_buffer) < args.batch_size:
                        continue
                    
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                    
                    # Perform the training
                    returns = rewards + args.gamma * (1 - dones) * network.predict_values(next_states).flatten()          
                
                    network.train_critics(states, actions, returns)
                    
                    # Update actor and target networks once every D t_step (TD3)
                    d = 2
                    if t_step % d == 0:
                        network.train_actor_and_targets(states, actions, returns)
                       
    
            # Periodic evaluation
            rets = [evaluate_episode(logging=False) for _ in range(args.evaluate_for)]
            mean_returns = np.mean(rets)
            print("Evaluation after episode {}: {:.2f}".format(env.episode, mean_returns))
            
            if mean_returns >= best_returns:
                best_returns = mean_returns
                
                prefix = 'td3__'
                actor_name = f"{prefix}actor{int(best_returns)}.h5"
                critic_name = f"{prefix}critic{int(best_returns)}.h5"
                critic_name2 = f"{prefix}2critic{int(best_returns)}.h5"
                tactor_name = f"{prefix}tactor{int(best_returns)}.h5"
                tcritic_name = f"{prefix}tcritic{int(best_returns)}.h5"
                tcritic_name2 = f"{prefix}2tcritic{int(best_returns)}.h5"
                network.actor.save(actor_name)
                network.critic.save(critic_name)
                network.critic2.save(critic_name2)
                network.target_critic.save(tcritic_name)
                network.target_actor.save(tactor_name)
                network.target_critic.save(tcritic_name)
                network.target_critic2.save(tcritic_name2)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    render = False
    if render:
        env = wrappers.EvaluationEnv(gym.make(args.env, render_mode='human'), args.seed, args.render_each)
    else:
        env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
