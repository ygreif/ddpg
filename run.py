import gym
import nn
import tensorflow as tf
import agent

env = gym.make('Pendulum-v0')
actor = nn.Actor(nn.ActorNeuralNetwork(3, 1, hidden_layers=[100], nonlinearity=tf.nn.relu), 2, {
                 'learning_rate': .001})
critic = nn.Critic(nn.CriticNeuralNetwork(
    3, 1, 1, [100, 100], nonlinearity=tf.nn.relu), {'learning_rate': .001})
obs = env.reset()

a = agent.Agent(actor, critic)
# a.coldstart(env, 5000, 1000)
for _ in range(1000):
    a.run_epoch(env, agent.NoExploration(max_steps=1000), learn=True)
# import numpy as np
# print np.concatenate(([env.observation_space.sample(),
# env.observation_space.sample()], [env.action_space.sample(),
# env.action_space.sample()]), axis=1)
