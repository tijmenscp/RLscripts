# File name:    windygrid_glie.py
# Author:       T.S.C. Pollack
# Description:  Training of an agent that is to find the quickest way out of a discrete maze
#               baffled by winds; training is performed using the GLIE (Greedy in the Limit
#               with Infinite Exploration) algorithm

import os
import sys
import numpy as np
from random import random, randint, seed
from copy import copy
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
        self.alpha = 0.2
        self.gamma = gamma
        self.Vfunc = {}
        self.Qfunc = {}
        self.action = 0
        self.decision_state = 0

        # Initialize value functions (naive strategy)
        for state in worldstates:

            self.Vfunc[state] = 0
            self.Qfunc[state] = {}

            for action in self.actions:
                self.Qfunc[state][action] = 0

    # Define every-visit MC policy evaluation
    def update(self, state, reward):

        # Observe our new state as a result of our last action and world dynamics
        update_state = tuple(state)

        # Make a shallow copy of our previous decision state and action
        decision_state = copy(self.decision_state)
        action = copy(self.action)

        # Choose the action using policy derived from Q (e-greedy)
        update_action = self.policy(update_state)

        # Get Q-value corresponding to this update action
        update_Q = self.Qfunc[update_state][update_action]

        # Update Q-value for the last state-action pair by bootstrapping on future estimate
        self.Qfunc[decision_state][action] += self.alpha * \
            (reward + gamma*update_Q -
             self.Qfunc[decision_state][action])

        # Update state value function
        self.Vfunc[decision_state] = max(self.Qfunc[decision_state].values())

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
    plt.imshow(V, origin='lower', vmin=V.min()*0, vmax=V.max(), cmap='magma')
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
gridsize = (16, 16)
grid = GridWorld(gridsize)
start, goal = ([0, 1], [7, 15])
grid.set_task(start, goal, duration)
grid.winds.define(1, [3, 6], "E")

# Create list of all possible states in this gridworld
worldstates = []
for we in range(gridsize[0]):
    for ns in range(gridsize[1]):
        worldstates.append((we, ns))

# Define agent
bot = Agent(worldstates, gamma)

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

        # Evaluate our last action
        bot.update(grid.s, reward)

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
    plt.title("SARSA $\epsilon$-greedy learning curve")
    plt.grid()

    # Display value function
    Vrender(bot.Vfunc, gridsize, start, goal)

    # Show plots
    plt.show()

print("EOF")
