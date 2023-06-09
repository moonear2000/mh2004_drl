from __future__ import print_function

import sys
import getopt

import rl_env
from myAgents.random_agent import RandomAgent
from myAgents.simple_agent import SimpleAgent

AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent}

class Runner(object):
  """Runner class"""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.environment = rl_env.make('Hanabi-Very-Small', num_players=flags['players'])
    self.agent_config = {'players': flags['players'], 'information_tokens': 3}
    
    # Note in Hanabi-very-small, each player has only 2 cards, there are 3 information_tokens and 1 life token. The deck has size 10 (1 of each card up to 5).

    self.agent_class = AGENT_CLASSES[flags['agent_class']]

  def run(self):
    """Run episodes."""
    rewards = []
    for episode in range(flags['num_episodes']):
      observations = self.environment.reset()
      agents = [self.agent_class(self.agent_config)
                for _ in range(self.flags['players'])]
      done = False
      episode_reward = 0
      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]
          action = agent.act(observation)
          if observation['current_player'] == agent_id:
            assert action is not None
            current_player_action = action
          else:
            assert action is None
        # Make an environment step.
        print('Agent: {} action: {}'.format(observation['current_player'],
                                            current_player_action))
        observations, reward, done, unused_info = self.environment.step(
            current_player_action)
        episode_reward += reward
      print(observations)
      rewards.append(episode_reward)
      print('Running episode: %d' % episode)
      print('Max Reward: %.3f' % max(rewards))
    return rewards

if __name__ == "__main__":

  flags = {'players': 2, 'num_episodes': 1, 'agent_class': 'RandomAgent'}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  if arguments:
    sys.exit('usage: rl_env_example.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  runner = Runner(flags)
  runner.run()