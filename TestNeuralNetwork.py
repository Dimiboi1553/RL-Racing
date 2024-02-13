import os
import numpy as np
from GymEnvironment import RacingEnv
from stable_baselines3 import PPO

TRAINING_PATH = os.path.join('Training', 'Logs')
DQN_PATH = os.path.join('Training', 'Saved_Models')

# Define the model PPO in this instance
model = PPO.load(r'Training\V1_3.zip')

for i in range(10):
    env = RacingEnv()
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, _ = model.predict(np.array(observation))
        observation, reward, done, _ = env.step(action)
        env.render()
        score += reward
    print(score)

env.close()