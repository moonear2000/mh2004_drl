import random
from hanabi_learning_environment.rl_env import Agent

class Q_agent(Agent):
  """Agent that takes random legal actions."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config

  def act(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] == 0:
      print(len(observation['vectorized']))
      return random.choice(observation['legal_moves'])
    else:
      return None