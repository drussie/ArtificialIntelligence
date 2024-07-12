from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
import random

class SpaceEnv(Env):
    def __init__(self):
        # Actions we can take: go left, do nothing, go right
        self.action_space = Discrete(3)
        # Observation space: alignment of the spaceship must be from -200 to 200 cm
        self.observation_space = Box(low=np.array([-200]), high=np.array([200]))

        # Set start position with some random noise
        self.state = 0 + random.randint(-15, 15)
        # Set time constraint of 300 seconds
        self.time = 300

    def step(self, action):
        # Apply action: -1 = left, 0 = do nothing, 1 = right
        # 0 -1 = -1 position
        # 1 -1 = 0
        # 2 -1 = 1 position
        self.state += action - 1
        # Reduce docking time by 1 second
        self.time -= 1

        # Calculate reward. If position is within safety range, reward = 1.
        # Else, reward = -1
        if -15 <= self.state <= 15:
            reward = 1
        else:
            reward = -1

        # Check if docking is done
        done = self.time <= 0

        # Apply position noise
        self.state += random.randint(-1, 1)
        # Set placeholder for info. This is just an OpenAI Gym requirement
        info = {}

        # Return step information
        return np.array([self.state]), reward, done, info

    def reset(self):
        # Reset spaceship position
        self.state = 0 + random.randint(-15, 15)
        # Reset docking time
        self.time = 300
        return np.array([self.state])

# Discretize the alignment range in 100 evenly spaced numbers over the interval -200 cm and 200 cm
discreteAlignmentSpace = np.linspace(-200, 200, 100)

def getState(observation):
    # Convert continuous state to discrete state
    discreteAlignment = int(np.digitize(observation, discreteAlignmentSpace))
    return discreteAlignment

numberActions = 3

def maxAction(Q, state):
    # Determine the action with the highest Q-value for the given state
    values = np.array([Q[state, a] for a in range(numberActions)])
    action = np.argmax(values)
    return action

# Model hyperparameters
# Alpha = learning rate, controls how fast the algorithm learns
ALPHA = 0.1
# GAMMA = discount factor for future rewards
GAMMA = 0.9

# Construct state space
states = []
for i in range(len(discreteAlignmentSpace) + 1):
    states.append((i))

def run_sarsa(alpha, eps_start=1.0, eps_end=0.0, eps_decay=0.002):
    Q = {}
    for s in states:
        for a in range(numberActions):
            Q[s, a] = 0

    totalRewards = []  
    avg_Rewards = []  
    numGames = 500  

    env = SpaceEnv()  
    EPS = eps_start  # Initialize epsilon

    for i in range(numGames):
        if i % 100 == 0:
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
            Q[s, a] = Q[s, a] + alpha * (reward + GAMMA * Q[s_, a_] - Q[s, a])
            s, a = s_, a_  
        EPS = max(EPS - eps_decay, eps_end)  
        totalRewards.append(epRewards) 
        avg_Rewards.append(np.mean(totalRewards[-10:])) 
        print('episode ', i, 'Total Rewards', totalRewards[-1],
              'Average Rewards', avg_Rewards[-1],
              'epsilon %.2f' % EPS)

    # Plot the average rewards
    plt.plot(avg_Rewards, label=f'Alpha = {alpha}')
    plt.title('Average Rewards vs. Episodes')
    plt.ylabel('Average Rewards')
    plt.xlabel('Episodes')
    plt.legend()

# Run SARSA with different ALPHA values
plt.figure()
run_sarsa(0.1)
run_sarsa(0.2)
run_sarsa(0.5)
plt.show()
