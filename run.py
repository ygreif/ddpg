import gym
import nn
import tensorflow as tf
import memory
import agent

env = gym.make('Pendulum-v0')
actor = nn.Actor(nn.NeuralNetwork(3, 1, nonlinearity=tf.nn.relu), {
                 'learning_rate': .001, 'dim': 2})
critic = nn.Critic(nn.NeuralNetwork(
    4, 1, [100, 100], nonlinearity=tf.nn.relu), {'learning_rate': .001})
obs = env.reset()

a = agent.Agent(actor, critic)
#a.coldstart(env, 5000, 1000)
for _ in range(1000):
    a.run_epoch(env, agent.NoExploration(max_steps=200), learn=True)
#import numpy as np
# print np.concatenate(([env.observation_space.sample(),
# env.observation_space.sample()], [env.action_space.sample(),
# env.action_space.sample()]), axis=1)
