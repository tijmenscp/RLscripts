# File name:    windygrid_glie.py
# Author:       T.S.C. Pollack
# Description:  Training of an agent that is to find the quickest way out of a discrete maze
#               baffled by winds; training is performed using the GLIE (Greedy in the Limit
#               with Infinite Exploration) algorithm

import os
import sys
import numpy as np
from random import random, randint, seed
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

parentPath = os.path.abspath("./")
sys.path.append(parentPath)

from gridworld import GridWorld

# Script settings
plotting = True

# Define agent


class Agent:

    def __init__(self, worldstates, gamma):
        self.actions = ["N", "E", "S", "W"]
        self.policy_eps = 0.1
        self.gamma = gamma
        self.k = 0
        self.N = {}
        self.Vfunc = {}
        self.Qfunc = {}
        self.action = 0
        self.decision_state = 0
        self.reset()

        # Initialize value functions (naive strategy)
        for state in worldstates:

            self.Vfunc[state] = 0
            self.N[state] = {}
            self.Qfunc[state] = {}

            for action in self.actions:
                self.N[state][action] = 0
                self.Qfunc[state][action] = 0

    def reset(self):
        self.sahist = []
        self.rhist = []
        self.G = {}

    # Define an observer
    def observe(self, reward):

        # Store everything we see until the end of the episode
        self.sahist.append((self.decision_state, self.action))
        self.rhist.append(reward)
        self.N[self.decision_state][self.action] += 1

    # Define every-visit MC policy evaluation
    def evaluate(self):

        # Calculate return for each state-action pair
        for t in range(len(self.sahist)):
            self.G[self.sahist[t]] = self.rhist[t]
            for i in range(1, len(self.sahist)-t):
                self.G[self.sahist[t]
                       ] += self.gamma**(i)*self.rhist[t+i]

        # Update Q-table with the returns observed in this episode
        for (state, action) in self.G:

            # Perform update
            self.Qfunc[state][action] += 1/self.N[state][action] * \
                (self.G[(state, action)]-self.Qfunc[state][action])

            # Update state value function
            self.Vfunc[state] = max(self.Qfunc[state].values())

        # Flush buffer from last episode
        self.reset()

        # Update episode count and epsilon
        self.k += 1
        # self.policy_eps = 1/self.k

    def policy(self, state):

        # Store decision state
        self.decision_state = tuple(state)

        # Get all known state-action values for current state
        Qvalues = self.Qfunc.get(self.decision_state, {})

        # Policy is epsilon-greedy
        if random() < self.policy_eps:
            action_arg = randint(0, len(self.actions)-1)
            self.action = self.actions[action_arg]
        else:
            self.action = max(Qvalues, key=lambda k: Qvalues[k])

        # Return proposed action
        return self.action


def Vrender(Vfunc, worldsize, start, goal):

    # Process value function for compatibility with matplotlib's imshow()
    V = np.zeros(worldsize)
    for state in Vfunc:
        V[state] = Vfunc[state]
    V = V.transpose()

    # Plot value function
    plt.figure()
    plt.imshow(V, origin='lower', vmin=V.min()*0, vmax=V.max(),cmap='magma')
    ax = plt.gca()
    plt.title("State value function Windy Grid World")
    plt.tick_params(axis='both', which='both', bottom='off',
                    top='off', left='off', right='off', labelbottom='off', labelleft='off')
    ax.set_xticks(np.arange(-.5, V.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5,  V.shape[0], 1), minor=True)
    plt.grid(b=True, which='minor', linestyle='-', linewidth=2)

    # Create overlay for start and goal states
    start, goal = tuple(start), tuple(goal)
    
    start_terminal = np.zeros(worldsize)
    start_terminal[start] = 1
    start_terminal = start_terminal > 0

    goal_terminal = np.zeros(worldsize)
    goal_terminal[goal] = 1
    goal_terminal = goal_terminal > 0

    start_terminal, goal_terminal = start_terminal.transpose(), goal_terminal.transpose()

    start_overlay = np.zeros((worldsize[1], worldsize[0], 4))
    goal_overlay = np.zeros((worldsize[1], worldsize[0], 4))

    start_overlay[..., 0], goal_overlay[..., 1] = 1, 1
    start_overlay[..., 3], goal_overlay[..., 3] = start_terminal, goal_terminal
    
    plt.imshow(start_overlay, origin='lower',)
    plt.imshow(goal_overlay, origin='lower',)


# Define hyperparameters
duration = 200
n_episodes = 1000
gamma = 0.95

# Initialize grid world
gridsize = (10, 10)
grid = GridWorld(gridsize)
start, goal = ([0, 1], [8, 2])
grid.set_task(start, goal, duration)
grid.winds.define(1, [3, 6], "N")

# Create list of all possible states in this gridworld
worldstates = []
for we in range(gridsize[0]):
    for ns in range(gridsize[1]):
        worldstates.append((we,ns))

# Define agent
bot = Agent(worldstates,gamma)

# Define auxiliary variables
epsreward = []

# Set seed
seed(9001)

# Train agent
for i in range(n_episodes):

    cumreward = 0

    # Run as long the episode is alive
    while not grid.complete and not grid.terminate:

        # Decide what action to take given our current state
        action = bot.policy(grid.s)

        # Simulate the Vironment over one step and obtain reward
        reward = grid.step(action)
        cumreward += reward

        # Observe the consequences of our actions
        bot.observe(reward)

    # Evaluate the policy from our last episode
    bot.evaluate()

    # Store total return obtained in last episode
    epsreward.append(cumreward)

    # Reset simulation
    grid.reset()

    # Print training status
    if i % 100 == 0 and i != 0:
        print("Training episode: ", i)

if plotting: 

    # Plot learning curve
    plt.figure()
    plt.plot(epsreward)
    plt.xlabel("Episode [-]")
    plt.ylabel("Cumulative reward per episode [-]")
    plt.title("Every-visit MC $\epsilon$-greedy learning curve")
    plt.grid()

    # Display value function
    Vrender(bot.Vfunc, gridsize, start, goal)

    # Show plots
    plt.show()

    print("EOF")
