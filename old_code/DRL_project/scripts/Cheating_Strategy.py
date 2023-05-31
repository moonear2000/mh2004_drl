from __future__ import print_function

import sys
import getopt
import pickle
import os
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt


import rl_env

class Runner(object):
    
    def __init__(self, flags):
        
        self.flags = flags
        self.environment = rl_env.make('Hanabi-Very-Small', num_players=flags['players'])


    def cheatingAgent(self, obs):
        hands = self.getHands(obs)
        curPlayer = obs['player_observations'][0]['current_player']
        fireworks = obs['player_observations'][0]['fireworks']
        discards = obs['player_observations'][0]['discard_pile']
        legal_moves = obs['player_observations'][curPlayer]['legal_moves']
        # Order:
        #   1) Play playable card
        #   2) Discard dead cards
        #   3) Discard duplicate cards
        #   4) Discrad dispensable card


        # 1) Playable
        hand = hands[curPlayer]
        for index, card in enumerate(hand):
            if self.isPlayable(card, fireworks):
                action = {'action_type': 'PLAY', 'card_index':index}
                return action
        
        # 2) Dead
        for index, card in enumerate(hand):
            if self.isDead(card, discards, fireworks):
                action = {'action_type': 'DISCARD', 'card_index':index}
                if action in legal_moves:
                    return action
                else:
                    action = self.giveRandomHint(legal_moves)
                    return action
                
        
        # 3) Duplicate
        hands_copy = dict(hands)
        hands_copy.pop(curPlayer)
        for i, c in enumerate(hand):
            for h in hands_copy.values():
                for index, card in enumerate(h):
                    if card == c:
                        action = {'action_type': 'DISCARD', 'card_index': i}
                        if action in legal_moves:
                            return action
                        else:
                            action = self.giveRandomHint(legal_moves)
                            return action
        
        # 4) Displensible
        for index, card in enumerate(hand):
            if self.isDispensible(card, discards):
                action = action = {'action_type': 'DISCARD', 'card_index': index}
                if action in legal_moves:
                    return action
                else:
                    action = self.giveRandomHint(legal_moves)
                    return action
                

        return {'action_type': 'DISCARD', 'card_index': 0}

    def isPlayable(self, card, fireworks):
        for color, rank in fireworks.items():
            if card['color'] == color and card['rank'] == rank:
                return True
        return False

    def isDead(self, card, discards, fireworks):
        frequency = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}

        # Card is dead because it's color firework has a higher rank
        if card['rank'] < fireworks[card['color']]:
            return True
        x = 0
        for color, rank in discards:
            if rank == card['rank'] and color == card['color']:
                x += 1
        if x == frequency[card['rank']]:
            return True
        return False
            
    def isDispensible(self, card, discards):
        frequency = {0: 3, 1: 2, 2: 2, 3: 2, 4: 1}  
        x = 0
        for color, rank in discards:
            if color == card['color'] and rank == card['rank']:
                x += 1
        if x < frequency[card['rank']]:
            return True
        return False

    def getHands(self, obs):
        numPlayers = obs['player_observations'][0]['num_players']
        hands = {}
        for observation in obs['player_observations']:
            p = observation['current_player'] + observation['current_player_offset']
            for i, hand in enumerate(observation['observed_hands'][1:]):
                if (p + i + 1)%numPlayers not in hands:
                    hands[(p + i + 1)%numPlayers] = hand
        return hands

    def giveRandomHint(self, legalActions):

        for action in legalActions:
            if 'REVEAL' in action['action_type']:
                return action

    def runCheatingStrategy(self, numIters = 100000):
        rewards = {}
        for i in range(1,6): rewards[i] = 0

        for game in tqdm(range(numIters)):
            observations = self.environment.reset()
            r = 0
            done = False

            while not done:
                action = self.cheatingAgent(observations)
                observations, reward, done, unused_info = self.environment.step(action)
                if reward == 1: r += 1
            
            if r not in rewards: rewards[r] = 1
            else: rewards[r] += 1

        return rewards

        
if __name__ == "__main__":

  flags = {'players': 2}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class='])
  for flag, value in options:
    flag = flag[2:]  # Strip leading --.
    flags[flag] = type(flags[flag])(value)
  runner = Runner(flags)
  numIters = 100000
  rewards = runner.runCheatingStrategy(numIters=numIters)
  print(rewards)
  r_sum = 0
  for reward, frequency in rewards.items():
    r_sum += reward*frequency
  print('Average reward = {}'.format(r_sum/numIters))
  y_axis = np.arange(len(rewards))[::-1]
  y_labels = np.arange(1,6)
  frequency = [rewards[i] for i in y_axis]
  plt.bar(y_axis, frequency, align='center', alpha = 0.8)
  plt.xticks(y_axis, y_labels)
  for r, f in zip(y_axis, frequency):
    plt.annotate('{}%'.format(round(f*100/numIters,2)), (r,f), ha='center', va='bottom')
  plt.ylabel('Frequency')
  plt.title('Bar chart showing scores from cheating game. Average reward = {}'.format(round(r_sum/numIters, 2)))
  plt.show()