import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from tensorflow.keras.callbacks import Callback
from keras.optimizers import Adam

from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

class MountainCarWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        position, velocity = self.state
        reward += (position + 0.6) / 1.1  # Normalize position value to range [0, 1] and add to reward
        return reward

class RewardLogger(Callback):
    def __init__(self):
        self.episode_rewards = []

    def on_episode_end(self, episode, logs={}):
        self.episode_rewards.append(logs.get('episode_reward'))

env = MountainCarWrapper(gym.make("MountainCarContinuous-v0"))  # environment for training
env_eval = gym.make("MountainCarContinuous-v0")  # environment for evaluation

states = env.observation_space.shape[0]  # amount of states
actions = env.action_space.shape[0]  # amount of actions

# Actor model
actor = Sequential()
actor.add(Flatten(input_shape=(1, states)))
actor.add(Dense(64, activation="relu"))
actor.add(Dense(64, activation="relu"))
actor.add(Dense(32, activation="relu"))
actor.add(Dense(actions, activation="tanh"))  # Continuous actions range from -1 to 1

# Critic model
action_input = Input(shape=(actions,), name="action_input")
observation_input = Input(shape=(1, states), name="observation_input")
flattened_observation = Flatten()(observation_input)
x = Dense(64, activation="relu")(flattened_observation)
x = Concatenate()([x, action_input])
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
x = Dense(1, activation="linear")(x)

critic = Model(inputs=[observation_input, action_input], outputs=x)

# DDPG Agent
memory = SequentialMemory(limit=50000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=actions, theta=0.15, mu=0.0, sigma=0.3)
agent = DDPGAgent(nb_actions=actions, actor=actor, critic=critic,
                  critic_action_input=action_input, memory=memory,
                  nb_steps_warmup_critic=100, nb_steps_warmup_actor=20000,
                  random_process=random_process, gamma=0.99, target_model_update=0.001)

# Compile the model with the Adam optimizer
agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=["mae"])

# Initialize reward logger
reward_logger = RewardLogger()

# Train the agent
agent.fit(env, nb_steps=100000, visualize=False, verbose=1, callbacks=[reward_logger])

# Calculate slope and intercept for the trend line
slope, intercept = np.polyfit(range(len(reward_logger.episode_rewards)), reward_logger.episode_rewards, 1)

# Plot rewards and trend line
plt.plot(reward_logger.episode_rewards)
plt.plot([0, len(reward_logger.episode_rewards)], [intercept, intercept + slope * len(reward_logger.episode_rewards)], 'r', label='Trendline')
plt.title('Training reward per episode')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend(['Reward', 'Trendline'], loc='upper left')
plt.show()

# Evaluate agent
episode_rewards = []
for episode in range(100):
    if episode < 10:
        result = agent.test(env_eval, nb_episodes=1, visualize=True)  # visualize the first 10 episodes
    else:
        result = agent.test(env_eval, nb_episodes=1, visualize=False)  # do not visualize the rest
    episode_rewards.append(result.history["episode_reward"][0])

print("Average reward over 100 test episodes: ", np.mean(episode_rewards))

env.close()
env_eval.close()
