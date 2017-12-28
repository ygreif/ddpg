import random
import gym
import nn
import tensorflow as tf
import agent


def gen_parameters():
    hidden_layers = random.choice(([1000], [100, 100], [100],
                                   [200, 200, 200], [100, 100, 100], [150, 150]))
    learning_rate = random.choice(
        [.01, .001, .0005, .002, .005])
    nonlinearity = random.choice([tf.nn.relu, tf.nn.tanh])
    return {
        'network': {'hidden_layers': hidden_layers, 'nonlinearity': nonlinearity},
        'learning': {'learning_rate': learning_rate}}


def gen_exploration_params():
    scale = random.choice((0, .1, .5))
    decay = random.choice((0, 1))
    return {'scale': scale, 'decay': decay}


def gen_all_parameters():
    return {'critic_params': gen_parameters(), 'agent_params': gen_parameters(), 'explore_params': gen_exploration_params()}


def train(n_epochs, critic_params, agent_params, explore_params):
    env = gym.make('Pendulum-v0')
    actor = nn.Actor(nn.ActorNeuralNetwork(
        3, 1, **agent_params['network']), 2, agent_params['learning'])
    critic = nn.Critic(nn.CriticNeuralNetwork(
        3, 1, 1, **critic_params['network']), critic_params['learning'])
    rewards = []
    a = agent.Agent(actor, critic)
    for i in range(n_epochs):
        decay = 1.0 - i * explore_params['decay'] / (n_epochs + 0.0)
        strat = agent.UniformExploration(decay * explore_params['scale'])
        reward = a.run_epoch(env, strat, learn=True)
        rewards.append(reward)
        if i % 100 == 0:
            print "Epoch", i, "Reward", rewards[-1]
    reward = a.run_epoch(env, agent.NoExploration(), learn=False)
    return {'agent': a, 'reward': reward}
