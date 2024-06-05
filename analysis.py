import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('single_random_mpc.npy', 'rb') as f:
    rewards_randommpc = np.load(f)
with open('single_mppi.npy', 'rb') as f:
    rewards_mppi = np.load(f)
with open('ensemble_mppi.npy', 'rb') as f:
    rewards_emppi = np.load(f)

sns.set_theme()
fig = plt.figure()
plt.plot(range(len(rewards_randommpc)), rewards_randommpc, color='r', label='Random MPC', linewidth=2)
plt.plot(range(len(rewards_mppi)), rewards_mppi, color='b', label='MPPI', linewidth=2)
plt.plot(range(len(rewards_emppi)), rewards_emppi, color='g', label='Ensemble MPPI', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.legend()
plt.savefig("rewards_comparison.pdf")