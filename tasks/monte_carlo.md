### Assignment: monte_carlo
#### Date: Deadline: Oct 17, 7:59 a.m.
#### Points: 5 points

Solve the discretized [CartPole-v1 environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
from the [Gym library](https://www.gymlibrary.dev/) using the Monte Carlo
reinforcement learning algorithm. The `gym` environments have the followng
methods and properties:
- `observation_space`: the description of environment observations
- `action_space`: the description of environment actions
- `reset() → new_state, info`: starts a new episode, returning the new
  state and additional environment-specific information
- `step(action) → new_state, reward, terminated, truncated, info`: perform the
  chosen action in the environment, returning the new state, obtained reward,
  boolean flags indicating a terminal state and episode truncation, and
  additional environment-specific information

We additionaly extend the `gym` environment by:
- `episode`: number of the current episode (zero-based)
- `reset(start_evaluation=False) → new_state, info`: if `start_evaluation` is
  `True`, an evaluation is started

Once you finish training (which you indicate by passing `start_evaluation=True`
to `reset`), your goal is to reach an average return of 490 during 100
evaluation episodes. Note that the environment prints your 100-episode
average return each 10 episodes even during training.

Start with the [monte_carlo.py](https://github.com/ufal/npfl122/tree/master/labs/01/monte_carlo.py)
template, which parses several useful parameters, creates the environment
and illustrates the overall usage.

During evaluation in ReCodEx, three different random seeds will be employed, and
you need to reach the required return on all of them. Time limit for each test
is 5 minutes.
