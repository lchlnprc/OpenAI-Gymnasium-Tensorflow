import gym
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

env = gym.make("MountainCarContinuous-v0")

states = env.observation_space.shape[0]  # amount of states
actions = env.action_space.shape[0]  # amount of actions

# Actor model
actor = Sequential()
actor.add(Flatten(input_shape=(1, states)))
actor.add(Dense(24, activation="relu"))
actor.add(Dense(24, activation="relu"))
actor.add(Dense(actions, activation="tanh"))  # Continuous actions range from -1 to 1

# Critic model
action_input = Input(shape=(actions,), name="action_input")
observation_input = Input(shape=(1, states), name="observation_input")
flattened_observation = Flatten()(observation_input)
x = Dense(24, activation="relu")(flattened_observation)
x = Concatenate()([x, action_input])
x = Dense(24, activation="relu")(x)
x = Dense(1, activation="linear")(x)

critic = Model(inputs=[observation_input, action_input], outputs=x)

# DDPG Agent
memory = SequentialMemory(limit=50000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=actions, theta=0.15, mu=0.0, sigma=0.3)
agent = DDPGAgent(nb_actions=actions, actor=actor, critic=critic,
                  critic_action_input=action_input, memory=memory,
                  nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=0.99, target_model_update=0.001)

# Compile the model with the Adam optimizer
agent.compile(Adam(lr=0.001, clipnorm=1.), metrics=["mae"])
agent.fit(env, nb_steps=1000000, visualize=False, verbose=1)

# Evaluate agent
results = agent.test(env, nb_episodes=1, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()
