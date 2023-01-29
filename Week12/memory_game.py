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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=4, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=1, type=int, help="Number of episodes to train on.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=5, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")


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
        attention = tf.linalg.matvec(normalized_memory, normalized_read_keys)
        softmax = tf.nn.softmax(attention, axis=-1)
        read_value = tf.linalg.matvec(memory, softmax, transpose_a=True)
            
    
        # Using concatenated encoded input and the read value, use a ReLU hidden
        # layer of size `args.hidden_layer` followed by a dense layer with
        # `env.action_space.n` units and `softmax` activation to produce a policy.
        policy = tf.keras.layers.Concatenate(axis=0)([encoded_input, read_value])
        policy = tf.keras.layers.Dense(args.hidden_layer, activation='relu')(encoded_input)
        policy = tf.keras.layers.Dense(env.action_space.n, activation='softmax')(policy)
        
        
        # Perform memory write. For faster convergence, append directly
        # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
        # the last memory row to keep memory size constant.
        
        #TODO is the memory[0, :-1] ok?
        updated_memory = tf.keras.layers.Concatenate(axis=0)([encoded_input, memory[0, :-1]])

        # Create the agent
        self._agent = tf.keras.Model(inputs=[memory, state], outputs=[updated_memory, policy])    
        self._agent.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.SparseCategoricalCrossentropy())
        #self._agent.summary()


    def zero_memory(self):
        # Return an empty memory. It should be a TF tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]`.
        return tf.zeros(shape=[self.args.memory_cells, self.args.memory_cell_size])

    #@tf.function
    def _train(self, states, targets, lengths):
        # Given a batch of sequences of `states` (each being a (card, symbol) pair),
        # train the network to predict the required `targets`.
        #
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `targets` as gold labels.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        # It is possible to use Ragged Tensors.
     
        """
        with tf.GradientTape() as tape:
            memories = [tf.zeros(shape=[self.args.memory_cells, self.args.memory_cell_size]) for x in range(len(states))]
            overall_losses = [0.0 for x in range(len(states))]
            for step in range(max(lengths)-1):
                memories[step], policies = self.predict([memories[step]], states[step])
                curr_losses = self._agent.loss(targets[step], policies)
                overall_losses = tf.math.add(overall_losses, curr_losses)
            loss = tf.math.reduce_mean(overall_losses)
            
        self._agent.optimizer.minimize(loss, self._agent.trainable_variables, tape=tape)
        """
        """
        memories = [tf.zeros(shape=[self.args.memory_cells, self.args.memory_cell_size]) for x in range(len(states))]
        overall_losses = [0.0 for x in range(len(states))]
        for step in range(max(lengths)):
            memories[step], policies = self.predict([memories[step]], states[step])
            curr_losses = self._agent.loss(targets[step], policies)
            overall_losses = tf.math.add(overall_losses, curr_losses)"""
            
            
            
        """
        with tf.GradientTape() as tape:
            memory = tf.zeros(shape=[self.args.memory_cells, self.args.memory_cell_size])
            loss = 0 
            for state, target_action in zip(states, targets):
                if target_action is None:
                    break
                memory, policy = self.predict([memory], [state])
                loss += self._agent.loss(target_action, policy)
            
        self._agent.optimizer.minimize(loss, self._agent.trainable_variables, tape=tape)"""
            

    def train(self, episodes):
        # Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        
        state_batches, action_batches, episode_lengths = [], [], [len(e) for e in episodes]
        max_len = max(episode_lengths)
        for step in range(max_len):
            states, actions = [], []
            for episode in episodes:
                if step < len(episode):
                    s, a = episode[step]
                    states.append(s)
                    actions.append(a)
                else:
                    states.append(None)
                    actions.append(None)
            state_batches.append(states)
            action_batches.append(actions)
                    
        self._train(state_batches, action_batches, episode_lengths)
        raise Exception('STOP after one episode')
        

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
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation: bool = False, logging: bool = True) -> float:
        state, memory = env.reset(start_evaluation=start_evaluation, logging=logging)[0], network.zero_memory()
        rewards, done = 0, False
        #print('-----')
        while not done:
            # Find out which action to use
            #print('Before: \n', memory)
            memory, policy = network.predict([memory], [state])
            #print('After: \n', memory)
            action = np.argmax(policy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards += reward
        return rewards

    # Training
    training = True
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each):   # // args.batch_size):
            episodes = []
            for _ in range(args.batch_size):
                episodes.append(env.expert_episode())

            # Train the network
            network.train(episodes)

        # Periodic evaluation
        returns = [evaluate_episode() for _ in range(args.evaluate_for)]

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("MemoryGame-v0", cards=args.cards), args.seed, args.render_each,
                                 evaluate_for=args.evaluate_for, report_each=args.evaluate_for)

    main(env, args)
