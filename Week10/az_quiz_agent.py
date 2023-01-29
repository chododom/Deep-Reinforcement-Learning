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

from __future__ import annotations
import argparse
import collections
import math
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from az_quiz import AZQuiz
import az_quiz_evaluator
import az_quiz_player_simple_heuristic
#import az_quiz_player_fork_heuristic
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=1, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="fork_az_92", type=str, help="Model path")
parser.add_argument("--num_simulations", default=150, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=False, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=1, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=1, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100000, type=int, help="Replay buffer max length.")


#########
# Agent #
#########
class Agent:
    def __init__(self, args: argparse.Namespace):
        #  Define an agent network in `self._model`.
        #
        # A possible architecture known to work consists of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer with softmax
        #   activation to produce the policy,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.
        
        input_layer = tf.keras.layers.Input((7, 7, 4))
        x = tf.keras.layers.Conv2D(20, (3,3), activation='relu', padding='same')(input_layer)        
        x = tf.keras.layers.Conv2D(20, (3,3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(20, (3,3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(20, (3,3), activation='relu', padding='same')(x)
        x = tf.keras.layers.Conv2D(20, (3,3), activation='relu', padding='same')(x)
        
        policy = tf.keras.layers.Conv2D(2, (3,3), activation='relu', padding='same')(x)
        policy = tf.keras.layers.Flatten()(policy)
        policy = tf.keras.layers.Dense(28, activation='softmax')(policy)
        
        value = tf.keras.layers.Conv2D(2, (3,3), activation='relu', padding='same')(x)
        value = tf.keras.layers.Flatten()(value)
        value = tf.keras.layers.Dense(1, activation='tanh')(value)
        
        self._model = tf.keras.Model(inputs=input_layer, outputs=[policy, value])
        self._model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate))

        

    @classmethod
    def load(cls, path: str) -> Agent:
        # A static method returning a new Agent loaded from the given path.
        agent = Agent.__new__(Agent)
        agent._model = tf.keras.models.load_model(path)
        return agent

    def save(self, path: str, include_optimizer=True) -> None:
        # Save the agent model as a h5 file, possibly with/without the optimizer.
        self._model.save(path, include_optimizer=include_optimizer, save_format="h5")
        
    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def train(self, boards: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> None:
        # Train the model based on given boards, target policies and target values.
        with tf.GradientTape() as tape:
            policies, values = self._model(boards)
            loss = tf.keras.losses.huber(target_values, values) + tf.keras.losses.categorical_crossentropy(target_policies, policies)            
        grads = tape.gradient(loss, self._model.trainable_variables)
        self._model.optimizer.apply_gradients(zip(grads, self._model.trainable_variables))


    @wrappers.typed_np_function(np.float32)
    @wrappers.raw_tf_function(dynamic_dims=1)
    def predict(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function.
        policies, values = self._model(boards)
        return policies, values
        

    def turn_to_play(self, game: AZQuiz):
        if game.winner is not None:
            if game.winner == 1:
                return True
            else:
                return False
        else:
            if game.to_play == 0:
                return True
            else:
                return False

    def board(self, game: AZQuiz) -> np.ndarray:
        # Generate the boards from the current `AZQuiz` game.
        #
        # The `game.board` returns a board representation, but you also need to
        # somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `AZQuiz.swap_players` might come handy);
        # - indicate the current player by adding channels to the representation.
        
        # Change the game so that the current player is always the same one
        if not self.turn_to_play(game):
            game.swap_players()
            board = game.board
            game.swap_players()
        else:
            board = game.board
        return board


########
# MCTS #
########
class MCTNode:
    def __init__(self, prior: float, turn_to_play=None):
        self.prior = prior  # Prior probability from the agent.
        self.game = None    # If the node is evaluated, the corresponding game instance.
        self.children = {}  # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0
        self.turn_to_play = turn_to_play

    def value(self) -> float:
        # Return the value of the current node, handling the
        # case when `self.visit_count` is 0.
        if self.visit_count == 0:
            return 0
        else:
            return self.total_value / self.visit_count

    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.
        
        # No winner yet
        if self.game.winner is None:
            policy, value = agent.predict([agent.board(self.game)])
            for action in self.game.valid_actions():
                self.children[action] = MCTNode(policy[0][action])
        else:
            value = -1
            
        self.total_value = value if self.game.winner is not None else value[0][0] 
        self.visit_count = 1
        

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior
        
        alphas = [alpha for x in self.children]
        noise = np.random.dirichlet(alphas)
        
        for i, ch in enumerate(self.children):
            orig_prior = self.children[ch].prior
            self.children[ch].prior = epsilon * noise[i] + (1 - epsilon) * orig_prior
            

    def select_child(self) -> tuple[int, MCTNode]:
        # Select a child according to the PUCT formula.
        def ucb_score(child):
            # For a given child, compute the UCB score as
            #   Q(s, a) + C(s) * P(s, a) * (sqrt(N(s)) / (N(s, a) + 1)),
            # where:
            # - Q(s, a) is the estimated value of the action stored in the
            #   `child` node. However, the value in the `child` node is estimated
            #   from the view of the player playing in the `child` node, which
            #   is usually the other player than the one playing in `self`,
            #   and in that case the estimated value must be "inverted";
            # - C(s) in AlphaZero is defined as
            #     log((1 + N(s) + 19652) / 19652) + 1.25
            #   Personally I used 1965.2 to account for shorter games, but I do not
            #   think it makes any difference;
            # - P(s, a) is the prior computed by the agent;
            # - N(s) is the number of visits of state `s`;
            # - N(s, a) is the number of visits of action `a` in state `s`.
            
            Q_s_a = -child.value()
            C_s = np.log((1 + self.visit_count + 1965) / 1965) + 1.25
            P_s_a = child.prior
            ucb = Q_s_a + C_s * P_s_a * (np.sqrt(self.visit_count) / (child.visit_count + 1))
            return ucb

        # Return the (action, child) pair with the highest `ucb_score`.
        best_score = -np.inf
        best_action = -1
        for ch in self.children:
            ch_score = ucb_score(self.children[ch])
            if ch_score > best_score:
                best_score = ch_score
                best_action = ch
                
        return best_action, self.children[best_action]


def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    # Run the MCTS search and return the policy proportional to the visit counts,
    # optionally including exploration noise to the root children.
    root = MCTNode(None)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Perform the `args.num_simulations` number of MCTS simulations.
    for _ in range(args.num_simulations):
        # Starting in the root node, traverse the tree using `select_child()`,
        # until a `node` without `children` is found.
        
        game_clone = game.clone()
        node = root
        path = []
        while node.children:
            path.append(node)
            action, node = node.select_child()
            if node.children:
                game_clone.move(action)

        # If the node has not been evaluated, evaluate it.
        if not node.is_evaluated():
            # Evaluate the `node` using the `evaluate` method. To that
            # end, create a suitable `AZQuiz` instance for this node by cloning
            # the `game` from its parent and performing a suitable action.
            game_clone.move(action)
            node.evaluate(game_clone, agent)
        else:
            # If the node has been evaluated but has no children, the
            # game ends in this node. Update it appropriately.
            node.total_value += node.value()
            node.visit_count += 1

        # Get the value of the node.
        node_val = node.value()

        # For all parents of the `node`, update their value estimate,
        # i.e., the `visit_count` and `total_value`.
        for parent in reversed(path):
            if parent.turn_to_play == node.turn_to_play:
                parent.total_value += node_val
            else:
                parent.total_value -= node_val
            parent.visit_count += 1

    # Compute a policy proportional to visit counts of the root children.
    # Note that invalid actions are not the children of the root, but the
    # policy should still return 0 for them.
    
    policy = np.zeros(28)
    for action, child in root.children.items():
        policy[action] = child.visit_count
    return policy / policy.sum()


############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])

def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz(randomized=False)
    boards_buf, policies_buf, player_turns_buf = [], [], []
    
    play_cnt = 0
    while game.winner is None:
        play_cnt += 1
        
        # Run the `mcts` with exploration.
        player_turns_buf.append(agent.turn_to_play(game))
        boards_buf.append(agent.board(game))
        
        policy = mcts(game, agent, args, explore=True)
        policies_buf.append(policy)

        # Select an action, either by sampling from the policy or greedily,
        # according to the `args.sampling_moves`.
        if play_cnt < args.sampling_moves:
            sampled_action = np.random.choice(range(28), p=policy)
            game.move(sampled_action)
        else:
            greedy_action = np.argmax(policy)
            game.move(greedy_action)

    # Return all encountered game states, each consisting of
    # - the board (probably via `agent.board`),
    # - the policy obtained by MCTS,
    # - the outcome based on the outcome of the whole game.
    
    if game.winner == 0:
        value = 1
    else:
        value = -1
        
    replays = []
    for player_turn, board, pol in zip(player_turns_buf, boards_buf, policies_buf):
        if player_turn == False:
            value *= -1
        replays.append(ReplayBufferEntry(board, pol, value))
    
    return replays


def train(args: argparse.Namespace) -> Agent:
    # Perform training
    agent = Agent(args)
    
    pretrained = True
    if pretrained:
        load_score = 95
        agent.load(f"az_{load_score}")
    
    replay_buffer = wrappers.ReplayBuffer(max_length=args.window_length)

    iteration = 0
    training = True
    best_score = 0
    while training:
        iteration += 1

        # Generate simulated games
        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)

            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games:
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append("Move {}, result {}".format(i, outcome).center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                "{:>3.0f} ".format(policy[action] * 100))
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                print(*["".join(line) for line in log], sep="\n")

        # Train
        for _ in range(args.train_for):
            # Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            
            batch = replay_buffer.sample(args.batch_size, np.random)
            boards, policies, vals = map(np.array, zip(*batch))
            agent.train(boards, policies, vals.reshape(-1, 1))

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            score = az_quiz_evaluator.evaluate(
                [Player(agent, argparse.Namespace(num_simulations=0)), az_quiz_player_simple_heuristic.Player()],
                games=56, randomized=False, first_chosen=False, render=False, verbose=False)
            print("Evaluation after iteration {}: {:.1f}%".format(iteration, 100 * score), flush=True)
            
            if score > best_score or iteration % 150 == 0:
                best_score = score
                path = f"fork_az_{100*score:.0f}"
                agent.save(path)
    return agent


#####################
# Evaluation Player #
#####################
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            
            game_clone = game.clone()
            if game_clone.to_play != 0:
                game_clone.swap_players()
            policy = self.agent.predict([game_clone.board])[0][0]
        else:
            # Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            
            policy = mcts(game, self.agent, self.args, explore=False)

        # Now select a valid action with the largest probability.
        best_prob = -np.inf
        best_action = -1
        for a in game.valid_actions():
            prob = policy[a]
            if prob > best_prob:
                best_prob = prob
                best_action = a
            
        return best_action


########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    # Set random seeds and number of threads
    if args.seed is not None:
        tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    az_quiz_evaluator.evaluate(
        [player, az_quiz_player_simple_heuristic.Player()],
        games=56, randomized=False, first_chosen=False, render=False, verbose=True,
    )
    