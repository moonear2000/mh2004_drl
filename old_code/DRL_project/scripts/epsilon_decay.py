import matplotlib.pyplot as plt
import numpy as np

n = np.linspace(0,10000000,1000000)
p = 0.01
N = 10000000
epsilon_dec = 1*p**(n/N)
epsilon_lin = 1 - n * (1-p)/N
plt.plot(n,epsilon_dec, c = 'b', label = 'Exponential decay')
plt.plot(n,epsilon_lin, c = 'r', label = 'Linear decay')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Exploration vs Exploitation')
plt.savefig('exp_vs_expl.png')