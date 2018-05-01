# File name:    GridWorld.py
# Author:       T.S.C. Pollack
# Description:  Grid world class definition

import numpy as np
from copy import copy

class GridWorld:

    def __init__(self, size):
        self.hsize = size[0]
        self.vsize = size[1]
        self.winds = self.Winds()
        self.start = [0, 0]
        self.goal  = [size[0]-1, size[1]-1]
        self.s     = copy(self.start)
        self.nstep = 0
        self.complete = False
        self.terminate = False

    class Winds:
        def __init__(self):
            self.strength = 0
            self.location = [0, 0]
            self.direction = "N"

        def define(self, strength, location, direction):
            self.strength = strength
            self.location = location
            self.direction = direction

    def set_task(self, start, goal, duration):
        for arg in [start, goal]:
            if not isinstance(arg, (list, tuple)) or not len(arg) == 2:
                raise ValueError('arguments must be list/tuple of length 2')

        self.start = start
        self.goal = goal
        self.s = copy(self.start)
        self.duration = duration

    def reset(self):
        self.s = copy(self.start)
        self.nstep = 0
        self.complete = False
        self.terminate = False

    def step(self, action):
        if not action in ["N", "W", "S", "E"]:
            raise ValueError('actions can only take the following values: "N","W","S","E"')
        
        # Update number of steps in episode
        self.nstep += 1

        # Check if episode needs to be terminated
        if self.nstep > self.duration:
            self.terminate = True
            r = -10
            print("Maximum number of actions exceeded")

        if not self.s == self.goal and not self.terminate:

            # Set immediate reward
            r = -1

            # Effect of taking an action
            if action == "E":
                self.s[0] = self.s[0] + 1
            elif action == "W":
                self.s[0] = self.s[0] - 1
            elif action == "N":
                self.s[1] = self.s[1] + 1
            elif action == "S":
                self.s[1] = self.s[1] - 1

            # Effect due to wind
            if self.winds.direction in ["N", "S"] and self.winds.location[0] <= self.s[0] <= self.winds.location[1]:
                dir = 1 if self.winds.direction == "N" else -1
                self.s[1] = self.s[1] + self.winds.strength*dir
            elif self.winds.direction in ["W", "E"] and self.winds.location[0] <= self.s[1] <= self.winds.location[1]:
                dir = 1 if self.winds.direction == "E" else -1
                self.s[0] = self.s[0] + self.winds.strength*dir

            # Check if we are still within the boundaries of the world
            self.s = list(np.minimum(self.s, [self.hsize-1, self.vsize-1]))
            self.s = list(np.maximum(self.s, [0, 0]))

        elif self.s == self.goal and not self.terminate:
            self.complete = True

            # Set immediate reward
            r = 100   

        # Return immediate reward
        return r

    def txt_render(self):
        world = np.chararray((self.hsize, self.vsize))
        world[:] = "'"
        world[self.goal[0],self.goal[1]] = 'X'
        world[self.s[0],self.s[1]] = 'O'
        world = world.transpose()
        world = np.flip(world,0)
        print(str(world))