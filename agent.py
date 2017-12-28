import memory
import numpy as np


class NoExploration(object):

    def __init__(self, max_steps=11000, target=11000):
        self.max_steps = max_steps
        self.target = target

    def explore(self, action):
        return action


class Agent(object):

    def __init__(self, actor, critic, memory_size=100000000, minibatch_size=300):
        self.actor = actor
        self.critic = critic
        self.memory = memory.Memory(memory_size)
        self.minibatch_size = minibatch_size

    def coldstart(self, env, rounds, batchsize):
        for i in range(rounds):
            state = [env.observation_space.sample() for _ in range(batchsize)]
            action = [env.action_space.sample() for _ in range(batchsize)]
            self.critic.coldstart(state, action)
            if i % 100 == 0:
                print self.critic.current_q(state, action)[0:4]

    def run_epoch(self, env, strat, learn=True):
        explore = strat.explore
        target = strat.target
        max_steps = strat.max_steps

        done = False
        cum_reward = False
        steps = 0
        state = env.reset()
        while not done and cum_reward < target and steps < max_steps:
            action = explore(self.action(state))
            action = np.minimum(np.maximum(
                action, env.action_space.low), env.action_space.high)
            if np.isnan(action).any():
                return -99999999
            next_state, reward, done, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            if learn:
                self.learn()
                self.sample()
            cum_reward += reward
            state = next_state
            steps += 1
            if steps > 11000 and steps % 100 == 0:
                print "On step", steps
            env.render()
        return cum_reward

    def action(self, state):
        return self.actor.action([state])

    def learn(self):
        minibatch = self.memory.minibatch(self.minibatch_size)
        state, actions, rewards, next_state, terminals = self.memory.minibatch(
            self.minibatch_size)
        next_action = self.actor.actions(next_state)
        self.critic.trainstep(state, actions, rewards,
                              next_state, next_action, terminals)
        gradients = self.critic.calcgradient(state, actions)
        self.actor.train(state, gradients)

    def sample(self):
        minibatch = self.memory.minibatch(1)
        state, actions, rewards, next_state, terminals = self.memory.minibatch(
            1)
        next_action = self.actor.actions(next_state)
        print "Action", actions, "Next action", next_action
        print "Current", self.critic.current_q(state, actions), "Predict", self.critic.current_q(next_state, next_action) * .95 + rewards[0][0], "Reward", rewards[0][0]
