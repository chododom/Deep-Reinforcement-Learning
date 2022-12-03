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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--envs", default=8, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=300, type=int, help="Size of hidden layer.")
parser.add_argument("--actor_lr", default=0.001, type=float, help="Learning rate of actor.")
parser.add_argument("--critic_lr", default=0.003, type=float, help="Learning rate of critic.")
parser.add_argument("--model_path", default="walker.model", type=str, help="Model path")
parser.add_argument("--replay_buffer_size", default=1000000, type=int, help="Replay buffer size")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.01, type=float, help="Target network update weight.")


class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # Create an actor. Because we will be sampling (and `sample()` from
        # `tfp.distributions` does not play nice with functional models) and because
        # we need the `alpha` variable, we use subclassing to create the actor.
        class Actor(tf.keras.Model):
            def __init__(self):
                super().__init__()
                # Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generaing means with `env.action_space.shape[0]` units and no activation
                # - a layer for generaing sds with `env.action_space.shape[0]` units and `tf.math.exp` activation
                # - finally, create a variable represeting a logarithm of alpha, using for example the following:
               
                self.hidden1 = tf.keras.layers.Dense(512, tf.nn.relu)
                self.hidden2 = tf.keras.layers.Dense(256, tf.nn.relu)
                self.means = tf.keras.layers.Dense(env.action_space.shape[0])
                self.sds = tf.keras.layers.Dense(env.action_space.shape[0]) #, activation=tf.keras.activations.exponential)
                self.exp = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))      
                
                self._log_alpha = tf.Variable(np.log(0.1), dtype=tf.float32)
                
            # Forward pass
            def call(self, inputs: tf.Tensor, sample: bool):
                # Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                
                actor = self.hidden1(inputs)
                actor = self.hidden2(actor)
                
                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                
                mus = self.means(actor)
                sds = self.sds(actor)
                sds = self.exp(sds)
                
                # - Then, create the action distribution using `tfp.distributions.Normal`
                #   with the `mus` and `sds`. Note that to support computation without
                #   sampling, the easiest is to pass zeros as standard deviations when
                #   `sample == False`.
                
                
                # TODO 0 mozna musi byt vektor urcity delky?
                #x = mus.shape[1]
                #zeros = np.zeros((x))
                actions_distribution = tfp.distributions.Normal(mus, sds) if sample else tfp.distributions.Normal(mus, 0)
                
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
                
                actions_distribution = tfp.bijectors.Tanh()(actions_distribution)
                actions_distribution = tfp.bijectors.Scale((env.action_space.high - env.action_space.low) / 2)(actions_distribution)
                actions_distribution = tfp.bijectors.Shift((env.action_space.high + env.action_space.low) / 2)(actions_distribution)
                
                # - Sample the actions by a `sample()` call.
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                
                actions = actions_distribution.sample()
                log_probs = actions_distribution.log_prob(actions)
                
                # VUBEC NEVIM
                # tohle jsem ted pridal, nevim jestli je to spravne, ale snazim se tim splnit posledni vetu v tom zadani v komentari
                log_probs = tf.math.reduce_mean(log_probs, axis=1)
                
                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                
                alpha = tf.math.exp(self._log_alpha)
                
                # - Return actions, log_probs and alpha.
                
                return actions, log_probs, alpha
                

        # Instantiate the actor as `self._actor` and compile it.
        
        self._actor = Actor()
        self._actor.compile(optimizer=tf.keras.optimizers.Adam(args.actor_lr))

        # Create a critic, which
        # - takes observations and actions as inputs,
        # - concatenates them,
        # - passes the result through two dense layers with `args.hidden_layer_size` units
        #   and ReLU activation,
        # - finally, using a last dense layer produces a single output with no activation
        # This critic needs to be cloned so that two critics and two target critics are created.
        
        observations_input = tf.keras.layers.Input(env.observation_space.shape)
        #observations_input2 = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(observations_input)
        actions_input = tf.keras.layers.Input(env.action_space.shape)
        critic = tf.keras.layers.concatenate([observations_input, actions_input])
        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(critic)
        critic = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(critic)
        critic = tf.keras.layers.Dense(1)(critic)
        self._critic = tf.keras.Model(inputs=[observations_input, actions_input], outputs=critic)
        self._critic.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        
        self._critic2 = tf.keras.models.clone_model(self._critic)
        self._critic2.compile(optimizer=tf.keras.optimizers.Adam(args.critic_lr),
                            loss=tf.keras.losses.MeanSquaredError())
        
        self._target_critic = tf.keras.models.clone_model(self._critic)
        self._target_critic2 = tf.keras.models.clone_model(self._critic2)


    def save_actor(self, path: str):
        # Because we use subclassing for creating the actor, the easiest way of
        # serializing an actor is just to save weights.
        self._actor.save_weights(path, save_format="h5")

    def load_actor(self, path: str, env: wrappers.EvaluationEnv):
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        self.predict_mean_actions([env.observation_space.sample()])
        self._actor.load_weights(path)

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
        
        # Train actor (second formula on slide 13 of SAC lecture)
        with tf.GradientTape() as actor_tape:
            #actions, log_probs, alpha = self.predict_sampled_actions(states)
            actions, log_probs, alpha = self._actor(states, sample=True)
            
            # NETUSIM
            alpha = tf.stop_gradient(alpha) # Where should this be?
            
            #critic_value = self.predict_values(states)
            critic = self._target_critic([states, actions]) - alpha * log_probs        
            critic2 = self._target_critic2([states, actions]) - alpha * log_probs
            critic_value = tf.minimum(critic, critic2)
            
            actor_loss = tf.math.reduce_mean(alpha * log_probs - critic_value)
        actor_grads = actor_tape.gradient(actor_loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(actor_grads, self._actor.trainable_variables))
        
        # Train alpha (formula on slide 16 of SAC lecture)
        with tf.GradientTape() as alpha_tape:
            #actions, log_probs, alpha = self.predict_sampled_actions(states)
            actions, log_probs, alpha = self._actor(states, sample=True)
            
            # NETUSIM
            log_probs = tf.stop_gradient(log_probs)
            alpha_loss = -alpha * log_probs - alpha * args.target_entropy
        alpha_grads = alpha_tape.gradient(alpha_loss, self._actor.trainable_variables)
        self._actor.optimizer.apply_gradients(zip(alpha_grads, self._actor.trainable_variables))
        
        #returns = tf.reshape(returns, (args.batch_size, -1))
        
        # Train critics (first formula on slide 12 of SAC lecture)
        self._critic.optimizer.minimize(
            lambda: self._critic.loss(returns, self._critic([states, actions], training=True)),
            var_list=self._critic.trainable_variables
        )
        self._critic2.optimizer.minimize(
            lambda: self._critic2.loss(returns, self._critic2([states, actions], training=True)),
            var_list=self._critic2.trainable_variables
        )
        
        # Finally, update the two target critic networks exponential moving
        # average with weight `args.target_tau`, using something like
        #   for var, target_var in zip(critic.trainable_variables, target_critic.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        
        for var, target_var in zip(self._critic.trainable_variables, self._target_critic.trainable_variables):
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
        
        actions, log_probs, alpha = self._actor(states, sample=True)
        critic = self._target_critic([states, actions]) - alpha * log_probs        
        critic2 = self._target_critic2([states, actions]) - alpha * log_probs
        return tf.minimum(critic, critic2)


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
            action = network.predict_mean_actions(np.asarray([state]))[0] # TODO is it ok?
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        network.load_actor(args.model_path, env)
        while True:
            evaluate_episode(True)

    # Create the asynchroneous vector environment for training.
    venv = gym.vector.make(args.env, args.envs, asynchronous=False)

    # Replay memory of a specified maximum size.
    replay_buffer = collections.deque(maxlen=args.replay_buffer_size)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state, training = venv.reset(seed=args.seed)[0], True
    best_returns = -np.inf
    while training:
        for _ in range(args.evaluate_each):
            # Predict actions by calling `network.predict_sampled_actions`.
            action = network.predict_sampled_actions(state)

            next_state, reward, terminated, truncated, _ = venv.step(action)
            done = terminated | truncated
            
            #def clip_reward(r):
            #    return 0 if r == -100 else r
            
            #reward = [clip_reward(x) for x in reward]
            
            for i in range(args.envs):
                replay_buffer.append(Transition(state[i], action[i], reward[i], done[i], next_state[i]))
            state = next_state

            # Training
            if len(replay_buffer) >= 4 * args.batch_size:
                # Note that until now we used `np.random.choice` with `replace=False` to generate
                # batch indices. However, this call is extremely slow for large buffers, because
                # it generates a whole permutation. With `np.random.randint`, indices may repeat,
                # but once the buffer is large, it happend with little probability.
                batch = np.random.randint(len(replay_buffer), size=args.batch_size)
                states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                # Perform the training
          
                # #TODO Toto je hodne wild reshape, vubec nevim jestli spravnej,
                # zatim to tu mam jenom aby prosel vypocet returns kvuli shapingu
                
                preds = network.predict_values(next_states)  #.reshape(-1, args.batch_size)
                returns =  rewards + args.gamma * (~dones) * preds
                #returns = rewards + args.gamma * (~dones) * network.predict_values(next_states)
                
                network.train(states, actions, returns)
                
                
        # Periodic evaluation
        rets = [evaluate_episode() for _ in range(args.evaluate_for)]
        mean_returns = np.mean(rets)
        if mean_returns >= best_returns:
            best_returns = mean_returns
            
            prefix = 'sac_normal__'
            actor_name = f"{prefix}actor{int(best_returns)}.h5"
            critic_name = f"{prefix}critic{int(best_returns)}.h5"
            critic_name2 = f"{prefix}2critic{int(best_returns)}.h5"
            tcritic_name = f"{prefix}tcritic{int(best_returns)}.h5"
            tcritic_name2 = f"{prefix}2tcritic{int(best_returns)}.h5"
            network.save_actor(actor_name)
            network._critic.save(critic_name)
            network._critic2.save(critic_name2)
            network._target_critic.save(tcritic_name)
            network._target_critic2.save(tcritic_name2)

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed, args.render_each)

    main(env, args)
