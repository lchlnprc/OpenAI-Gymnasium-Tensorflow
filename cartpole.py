import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1")

states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01
)

# Compile the model with the Adam optimizer
agent.compile(Adam(lr=0.001), metrics=["mae"])
history = agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# Saving the rewards in a list
reward_history = history.history["episode_reward"]

# Select 10 random episodes to visualize
visualize_episodes = np.random.choice(100, 10, replace=False)

# Evaluate agent over 100 episodes
total_rewards = []
for i in range(100):
    visualize = i in visualize_episodes
    result = agent.test(env, nb_episodes=1, visualize=visualize)
    total_reward = result.history['episode_reward'][0]
    total_rewards.append(total_reward)

# Print the average reward for the evaluation episodes
avg_reward = np.mean(total_rewards)
print(f'Average reward: {avg_reward}')

# Plotting rewards
plt.figure(figsize=(12, 8))
plt.plot(reward_history, label='Rewards')
slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(reward_history)),reward_history)
plt.plot([0,len(reward_history)],[intercept,intercept+slope*len(reward_history)], 'r', label='Trendline') # Plotting trendline
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Rewards across training episodes')
plt.legend()
plt.show()

env.close()
