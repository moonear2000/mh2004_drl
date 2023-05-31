import random
import math
from hanabi_learning_environment.rl_env import Agent


class GreedyAgent(Agent):
  """Agent that takes random legal actions."""

  def __init__(self, config, epsilon=0, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    # epsilon determines how much exploration is done
    self.epsilon = epsilon

  def act(self, encoded_state, q_table):
    """Act based on an observation, and q_table."""
    # Explore with probability epsilon
    if random.uniform(0,1)<self.epsilon:
        random_action = random.choice(q_table[encoded_state]['actions'])
        while 'COLOR' in random_action['action']['action_type']:
          random_action = random.choice(q_table[encoded_state]['actions'])
        return random_action['action']
    else:
        best_actions = []
        best_value = -math.inf
        for a in q_table[encoded_state]['actions']:
          if 'COLOR' in a['action']['action_type']:
            continue
          if a['value']>best_value:
            best_actions = [a['action']]
            best_value = a['value']
          elif a['value'] == best_value:
            best_actions.append(a['action'])
        return random.choice(best_actions)
        