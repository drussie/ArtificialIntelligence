# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/154fGck2cBxqgWNVI2j-7dX8VhB5WlNTq

#**Florida International University - CAP 4630 Artificial Intelligence**

###Programming Assignment 4 - Reinforcement Learning

This assignment was designed by Paulo Padrao
(YouTube: [@padraopaulo](https://www.youtube.com/@padraopaulo ) | ppadraol@fiu.edu)

##Installing Dependencies
"""

"""## Creating and Testing a Random Environment with OpenAI Gym"""

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
import random

"""###Scenario

Every day it's the same thing during bath time. It took us longer trying to adjust the water temperature than actually 
taking a shower.

###Goal### 
Build a reinforcement learning model to adjust the water temperature autonomously to get in the optimal range.

###Time constraint### 
Shower Length = 5 minutes (300s)

###Optimal Temperature (in °C)###
Between 37 and 39 degrees

###Actions###
Turn temperature down, Do Nothing, Turn temperature up

###Task###
Build a model that keeps us in the optimal temperature range for as long as possible during shower time.

###OpenAI Gym Environments###
In OpenAI Gym, an environment should have 4 functions:



1.   `__init__(self)` , an initilization function that defines the initial values for the environment.
2.   `step(self, action)`, a step function that provides the next observation (state) of the system, a reward, 
and stopping criteria (done) given an action.
3. `render(self)`, a render function for visualization. Not implemented here.
4. `reset(self)`, a function to reset the values of the environment after each epsiode. 


"""


class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take: temperature down, no change in temperature, temperature up
        self.action_space = Discrete(3)
        # Temperature array with continuous values. Min temp = 0, Max temp = 100
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))

        # Set start temp with some random noise
        self.state = 38 + random.randint(-3, 3)
        # Set shower length (in seconds)
        self.shower_length = 300

    def step(self, action):
        # Apply action. By default, action values are 0, 1, 2.
        # 0 -1 = -1 temperature
        # 1 -1 = 0
        # 2 -1 = 1 temperature
        self.state += action - 1
        # Reduce shower length by 1 second
        self.shower_length -= 1

        # Calculate reward. If temperature is within optimal range, reward = 1.
        # Else, reward = -1
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

            # Check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Apply temperature noise
        self.state += random.randint(-1, 1)
        # Set placeholder for info. This is just an OpenAI Gym requirement
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        # If we want to visualize the sytem, we can implement the render function
        pass

    def reset(self):
        # Reset shower temperature
        self.state = 38 + random.randint(-3, 3)
        # Reset shower time
        self.shower_length = 60
        return self.state


"""Defining the Environment"""

env = ShowerEnv()

"""Checking action space"""

# You should expect an integer representing the number of actions we define.
# In this case, 3.

#print(env.action_space.n)

"""Getting a random observation from the environment"""

#print(env.observation_space.sample())

"""Testing the environment"""

episodes = 10
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        # env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

"""## SARSA Algorithm

Here's an overview of the [SARSA Algorithm](https://www.youtube.com/watch?v=FhSaHuC0u2M). Now that we have our 
environment set up, the first step towards the implementation of SARSA algorithm is to discretize the observation 
space. This is needed because our Q-function is discrete, but the observation space is not since the temperature can 
be any float number from 0 to 100 degrees Celcius. """

discreteTemperatureSpace = np.linspace(0, 100, 100)


def getState(observation):
    discreteTemperature = int(np.digitize(observation, discreteTemperatureSpace))

    return discreteTemperature


numberActions = env.action_space.n


def maxAction(Q, state):
    values = np.array([Q[state, a] for a in range(numberActions)])
    action = np.argmax(values)
    return action


# model hyperparameters
# ALPHA = learning rate, controls how fast the algorithm learns
ALPHA = 0.1
# GAMMA = discount factor for future rewards
GAMMA = 0.9
# EPS = epsilon parameter for the epsilon-greedy methods such as SARSA
EPS = 1.0
# construct state space
states = []
for i in range(len(discreteTemperatureSpace) + 1):
    states.append((i))

Q = {}
for s in states:
    for a in range(numberActions):
        Q[s, a] = 0

totalRewards = []
avg_Rewards = []
numGames = 500

for i in range(numGames):
    if i % numGames == 0:
        print('starting game', i)
    observation = env.reset()
    s = getState(observation)
    rand = np.random.random()
    a = maxAction(Q, s) if rand < (1 - EPS) else env.action_space.sample()
    done = False
    epRewards = 0
    while not done:
        observation_, reward, done, info = env.step(a)
        s_ = getState(observation_)
        rand = np.random.random()
        a_ = maxAction(Q, s_) if rand < (1 - EPS) else env.action_space.sample()
        epRewards += reward
        Q[s, a] = Q[s, a] + ALPHA * (reward + GAMMA * Q[s_, a_] - Q[s, a])
        s, a = s_, a_
    EPS -= 2 / (numGames) if EPS > 0 else 0
    totalRewards.append(epRewards)
    avg_Rewards.append(np.mean(totalRewards[-10:]))
    print('episode ', i, 'Total Rewards', totalRewards[-1],
          'Average Rewards', avg_Rewards[-1],
          'epsilon %.2f' % EPS)
plt.plot(avg_Rewards, 'b--')
plt.title('Average Rewards vs. Episodes')
plt.ylabel('Average Rewards')
plt.xlabel('Episodes')
plt.show()

"""#Homework Assignment (40 points)

##Scenario

We are on the way to setting foot on Mars. 
To do so, we need to refuel our space capsule in a space station thousands of miles away from Earth. 

We need to perform a docking operation similarly as shown in [this video](https://www.youtube.com/watch?v=Jd_aIRkI5ws).

##Goal 
Build a reinforcement learning model to perfom the docking operation autonomously 
while respecting safety and time constraints.

##Time constraint
The docking operation must be done in 5 minutes (300 s)

##Safety constraint For the docking operation to be performed safely, the space capsule needs to be aligned +/- 15 cm 
from the docking center of the space station. 

##Actions
You only have 3 actions: go left, do nothing, go right.

##Observations Space capsule alignment with the space station. It is assumed the alignment can range from -2 m to 2 
m, where 0 m means perfect alignment. We also assume that negative values describe a misalignment to the left and 
positive values represent a misalignment to the right. 

##Tasks

Create a new .py file called SpaceCapsuleEnvironment.py and 
build a model that keeps the space capsule in the safety range for as long as possible using SARSA algorithm. 

* 1) [15 points] Create a `SpaceEnv()` class with the following function
  * [5 points] `__init__(self)` , an initilization function that defines the initial values for the environment. 
  * [5 points] `step(self, action)`, a step function that provides the next observation (state) of the system, a reward,
   and stopping criteria (done) given an action. 
  * [5 points] `reset(self)`, a function to reset the values of the environment after each epsiode.
* 2) [10 points] Discretize the alignment range in 100 evenly spaced numbers over the interval -200 cm and 200 cm. 
You should use the function `np.linspace`. Also, update the function `getState(observation)`.

* 3) [15 points] Run SARSA algorithm with the `SpaceEnv()` you've just created. You should save the plot containing the 
results.
"""
