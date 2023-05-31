from __future__ import print_function

import sys
import getopt
import pickle
import os
from tqdm import tqdm
import numpy as np
import random


import rl_env
from myAgents.epsilon_greedy import GreedyAgent

"""Some notes regarding this script:
    1) define an epsilon greedy agent that starts exploring using random states and then explores more common states
    2) define the tabular method for maintaining q values
    3) define update rules for q-values"""

class Trainer(object):
  """Runner class"""
  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.environment = rl_env.make('Hanabi-Very-Small', num_players=flags['players'])
    self.agent_config = {'players': flags['players'], 'information_tokens': 3}
    self.epsilon_initial = 0.9 # explores 'epsilon' of the time
    self.agent_class = GreedyAgent
    self.gamma = 0.9
    self.lr_initial = 0.3 #fast learning
    dirname = os.path.dirname(__file__)
    self.filename = os.path.join(dirname, 'tables/q_table.pickle')
    try:
        with open(self.filename, 'rb') as fp:
            self.q_table = pickle.load(fp)
    except IOError:
        print('No old table found, starting a new table')
        self.q_table = {}
    
  def learning_rate(self, iteration, total_iterations):
    alpha = self.lr_initial * np.exp(-(iteration)*4/total_iterations)
    if alpha < 0.1:
        return 0.1
    else:
        return alpha

  def encode_state(self, observation):
    # The purpose of this function is to encode the state into a string that can be used as a key to the q_dict

    state= ''
    color = 'R'
    # life tokens [0,1]
    state += str(observation['life_tokens'])
    # information tokens [0,1,2,3]
    state += str(observation['information_tokens'])
    # fireworks [0,1,2,3,4]
    state += str(observation['fireworks'][color])
    # discard pile: list of discarded numbers
    discard_pile = ''
    for card in observation['discard_pile']:
        discard_pile += str(card['rank'])
    state += 'X'*(10 - len(discard_pile)) + ''.join(sorted(discard_pile))

    # current agent hand knowledge (must be sorted)
    hand_knowledge = ''
    for card in observation['card_knowledge'][0]:
        if card['rank'] == None:
            hand_knowledge += 'X'
            continue
        hand_knowledge += str(card['rank'])
    if len(hand_knowledge)<2:
        hand_knowledge += 'X'
    state += hand_knowledge

    # his hand (unsorted)
    hand = ''
    for card in observation['observed_hands'][-1]:
        hand += str(card['rank'])
    if len(hand) <2:
        hand += 'X'
    state += ''.join(sorted(hand))

    # his hand knowledge (unsorted)
    hand_knowledge = ''
    for card in observation['card_knowledge'][1]:
        if card['rank'] == None:
            hand_knowledge += 'X'
            continue
        hand_knowledge += str(card['rank'])
    if len(hand_knowledge) <2:
        hand_knowledge += 'X'
    state += ''.join(sorted(hand_knowledge))
    
    # the length of state string should be 19
    if len(state) != 19:
        print(observation)
    assert len(state) == 19
    return state

  def find_q(self, state, action = None):

    if action == None: # Find over all actions
        max_value = 0
        for a in self.q_table[state]['actions']:
            max_value = max(a['value'], max_value)
        return max_value
    else:
        for a in self.q_table[state]['actions']:
            if a['action'] == action:
                return a['value']
        return 0

  def update_q_table(self, old_state, new_state, action, reward, lr=0.1):
    
    old_q = self.find_q(old_state, action)
    new_q = self.find_q(new_state)
    q = old_q + lr *(reward + self.gamma*(new_q) - old_q)

    for a in self.q_table[old_state]['actions']:
        if a['action'] == action:
            a['value'] = q

  def add_state_to_table(self, state, legal_moves):
    self.q_table[state] = {'actions':[], 'e':1}
    for a in legal_moves:
        if 'COLOR' != a['action_type'][-5:]: # Do not add revealing color as a legal action (as it is irrelevant in this game)
            self.q_table[state]['actions'].append({'action': a, 'value': 0})

  def evaluate(self):

    num_games = 20000
    rewards = np.zeros(num_games)
    unvisited_states = 0
    for game in range(num_games):
        observations = self.environment.reset()
        agents = [self.agent_class(self.agent_config, 0) for _ in range(self.flags['players'])]
        done = False
        episode_reward = 0
        reward = 0
        while not done:
            
            for agent_id, agent in enumerate(agents):
                if observations['player_observations'][agent_id]['current_player_offset'] == 0:
                    observation = observations['player_observations'][agent_id]
                    state = self.encode_state(observation)
                    if state not in self.q_table:
                        unvisited_states += 1
                        action = random.choice(observation['legal_moves'])
                    else:
                        action = agent.act(state, self.q_table)
                    observations, reward, done, unused_info = self.environment.step(action)
                    break
            
            if reward == 1: episode_reward += reward

        rewards[game] = episode_reward
    
    average_score = np.mean(rewards)
    print('Initial state value = {}'.format(self.q_table['X']))
    print('The average score across {} games is {}'.format(num_games, average_score))
    print('{} unvisited states'.format(unvisited_states))

  def play_games(self, num_games = 5):
    "Run episodes"
    rewards = []
    for episode in range(num_games):
        # Reset game
        observations = self.environment.reset()
        # Create agents
        agents = [self.agent_class(self.agent_config, 0) for _ in range(self.flags['players'])]
        done = False
        episode_reward = 0
        # q-parameters
        old_state = 'X'
        reward = 0
        action = 'Start Game'
        # Start game
        print('Start Game')
        while not done:
            # At each turn, iterate through all agents
            for agent_id, agent in enumerate(agents):
                
                # Only proceed for current agent
                if observations['player_observations'][agent_id]['current_player_offset'] == 0:
                    # Find that agents observations
                    observation = observations['player_observations'][agent_id]
                    new_state = self.encode_state(observation)
                    if new_state not in self.q_table:
                        self.add_state_to_table(new_state, observation['legal_moves'])
                    action = agent.act(new_state, self.q_table)
                    for a in self.q_table[new_state]['actions']:
                        if a['action'] == action:
                            None
                    assert action is not None
                    print('Agent: {} action: {}'.format(observation['current_player'],
                                            action))
                    #print('Legal Moves: {}'.format(observation['legal_moves']))
                    #print('His knowledge: {}'.format(observation['card_knowledge'][1]))

                    observations, reward, done, unused_info = self.environment.step(action)
                    break

            # Note when playing a correct card, reward is +1
            if reward == 1: episode_reward += reward
        rewards.append(episode_reward)
        print('Episode Reward = {}'.format(episode_reward))

  def train(self):

    # Improvements to make to q-learning training:
        # 1) The reward in the update step should be the cumulative reward between actionable states (shared reward)
        # 2) When a game is lost and no correct plays have been made, update with a penelty of -100
        # 3) When a game is lost after x correct plays, update terminal states with a reward of x
        # 4) Epsilon should be determined by the number of times that state has been visited. 
        #    It should decay with 1/visits. When a state has been visited multiple times, take the greedy route
    # The idea behind 1 is that if a player makes an action to facilitate another player playing a card, he should be rewarded
    # The idea behind 2 is to discourage playing moves that may result in a loss (i.e playing cards you don't know). Instead, only play cards you know, otherwise discard
    # The idea behind 3 is to add a terminal reward at terminal states that can backpropogate through the state tree. Rewards given during gameplay are guidance towards terminal states.
    
    

    if 'X' not in self.q_table:
            self.add_state_to_table('X', ['Start Game'])
    
    if 'Terminal' not in self.q_table:
            self.add_state_to_table('Terminal', ['Invalid'])

    num_episodes = 1
    # Initial fully exploration phase
    for episode in tqdm(range(num_episodes)):
        # Reset game
        lr = 0.1
        observations = self.environment.reset()

        # Create agents
        # TO-DO: Change so that epsilon is set based on the number of times a state has been visited
        agents = [self.agent_class(self.agent_config, 0.2) for _ in range(self.flags['players'])]
        state_log = {0: [('X','Start Game')], 1:[('X','Start Game')]}
        done = False
        episode_reward = 0
        rewards = []
        reward = 0

        while not done:
            rewards.append(reward)
            for agent_id, agent in enumerate(agents):
                if observations['player_observations'][agent_id]['current_player_offset'] == 0:
                    observation = observations['player_observations'][agent_id]
                    new_state = self.encode_state(observation)
                    if new_state not in self.q_table:
                        self.add_state_to_table(new_state, observation['legal_moves'])
                    self.update_q_table(state_log[agent_id][-1][0], new_state, state_log[agent_id][-1][1], sum(rewards[-2:]), lr=lr)
                    action = agent.act(new_state, self.q_table)
                    state_log[agent_id].append((new_state, action))
                    observations, reward, done, unused_info = self.environment.step(action)
                    break
            if reward == 1: episode_reward += reward
        
        # Adjust final states
        if episode_reward == 0:
            final_reward = -100
        else: 
            final_reward = episode_reward*10
        
        for agent_id, agent in enumerate(agents):
            self.update_q_table(state_log[agent_id][-1][0], 'Terminal', state_log[agent_id][-1][1], final_reward, lr=lr)

    self.evaluate()

    epsilon_decay = np.exp(np.log(0.05)/num_episodes)
    epsilon = 1
    num_episodes = self.flags['num_episodes']

    for episode in tqdm(range(num_episodes)):
        # Reset game
        lr = 0.001
        observations = self.environment.reset()

        # Create agents
        # TO-DO: Change so that epsilon is set based on the number of times a state has been visited
        agents = [self.agent_class(self.agent_config, 0.3) for _ in range(self.flags['players'])]
        state_log = {0: [('X','Start Game')], 1:[('X','Start Game')]}
        done = False
        episode_reward = 0
        rewards = []
        reward = 0

        while not done:
            rewards.append(reward)
            for agent_id, agent in enumerate(agents):
                if observations['player_observations'][agent_id]['current_player_offset'] == 0:
                    observation = observations['player_observations'][agent_id]
                    new_state = self.encode_state(observation)
                    if new_state not in self.q_table:
                        self.add_state_to_table(new_state, observation['legal_moves'])
                    self.q_table[new_state]['e'] *= epsilon_decay
                    self.update_q_table(state_log[agent_id][-1][0], new_state, state_log[agent_id][-1][1], sum(rewards[-2:]), lr=lr)
                    action = agent.act(new_state, self.q_table)
                    state_log[agent_id].append((new_state, action))
                    observations, reward, done, unused_info = self.environment.step(action)
                    break
            if reward == 1: episode_reward += reward
        
        # Adjust terminal states
        if episode_reward == 0:
            final_reward = -100
        else: 
            final_reward = episode_reward*10
        
        for agent_id, agent in enumerate(agents):
            self.update_q_table(state_log[agent_id][-1][0], 'Terminal', state_log[agent_id][-1][1], final_reward, lr=lr)

        epsilon *= epsilon_decay
            
        if episode%200000 == 0:
            print('Saving table')
            self.evaluate()
            try:
                with open(self.filename, 'wb') as fp:
                    pickle.dump(self.q_table, fp, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                print('Could not save')

if __name__ == "__main__":

  flags = {'players': 2, 'num_episodes': 5000000, 'agent_class': 'RandomAgent'}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  runner = Trainer(flags)
  runner.play_games(10)