import os
from GymEnvironment import RacingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

TRAINING_PATH = os.path.join('Training', 'Logs')
DQN_PATH = os.path.join('Training', 'Saved_Models')
#environment for training
env = RacingEnv()

total_timesteps = 400_000
save_interval = 50_000
#Define the model DQN in this instance
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=TRAINING_PATH,learning_rate=0.0003)#PPO.load(r"Training\DrivingModelV3.zip",env=env)#

# There are two modes to rendering in the racing sim Environment:
# 1) False, which is no render
# 2) True, which does render the scene
env.render_mode = False


model.learn(total_timesteps=total_timesteps)
    

# Save the trained model
#model.save(DQN_PATH)

# Close the environment
env.close()