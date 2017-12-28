import tensorflow as tf
import numpy as np


class FullyConnectedLayer(object):

    def __init__(self, inp, dim, nonlinearity=False):
        self.W = tf.Variable(tf.random_normal(dim))
        self.b = tf.Variable(tf.constant(1.0, shape=(1, dim[1])))
        if nonlinearity:
            self.out = nonlinearity(tf.matmul(inp, self.W) + self.b)
        else:
            self.out = tf.matmul(inp, self.W) + self.b


class NeuralNetwork(object):

    def __init__(self, inp, indim, enddim, hidden_layers=[10], nonlinearity=False, name='default'):
        self.layers = []
        self.x = inp
        self.indim = indim
        self.enddim = enddim

        inp = self.x
        prev_dim = indim
        for out_dim in hidden_layers:
            self.layers.append(
                FullyConnectedLayer(
                    inp, (prev_dim, out_dim), nonlinearity=nonlinearity))
            inp = self.layers[-1].out
            prev_dim = out_dim
        self.layers.append(FullyConnectedLayer(
            inp, (prev_dim, enddim), nonlinearity=False))
        self.out = self.layers[-1].out

    def variables(self):
        variables = []
        for layer in self.layers:
            variables.extend([layer.W, layer.b])
        return variables


class CriticNeuralNetwork(object):

    def __init__(self, statedim, actiondim, enddim, hidden_layers=[10], nonlinearity=False, name='default'):
        self.state = tf.placeholder(tf.float32, [None, statedim])
        self.action = tf.placeholder(tf.float32, [None, actiondim])
        # grr why this order?
        inp = tf.concat(1, [self.state, self.action])
        self.network = NeuralNetwork(
            inp, statedim + actiondim, enddim, hidden_layers, nonlinearity, name)
        self.out = self.network.out


class ActorNeuralNetwork(object):

    def __init__(self, indim, enddim, hidden_layers=[10], nonlinearity=False, name='default'):
        self.x = tf.placeholder(tf.float32, [None, indim])
        self.network = NeuralNetwork(
            self.x, indim, enddim, hidden_layers, nonlinearity, name)
        self.out = self.network.out

    def variables(self):
        return self.network.variables()


class Critic(object):

    def __init__(self, nn, parameters, discount=.95):
        self.discount = tf.constant(discount)

        self._setup_q_calculation(nn)
        self._setup_next_q_calulcation(nn)
        self._setup_train_step(parameters)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_q_calculation(self, nn):
        self.state = nn.state
        self.action = nn.action
        self.q = nn.out

    def _setup_next_q_calulcation(self, nn):
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.terminal = tf.placeholder(tf.float32, [None, 1])

        self.update = self.q * \
            (self.terminal * self.discount) + self.r

    def _setup_train_step(self, parameters):
        self.target = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.reduce_sum(tf.square(self.target - self.q))

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=parameters['learning_rate']).minimize(self.loss)
        self.gradients = tf.gradients(self.q, self.action)

    def next_q(self, rewards, next_state, next_action, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.state: next_state, self.action: next_action, self.terminal: terminals})

    def current_q(self, state, action):
        return self.session.run(self.q, feed_dict={self.state: state, self.action: action})

    def calcloss(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.loss, feed_dict={self.target: target, self.x: state})

    def calcgradient(self, state, action):
        return self.session.run(self.gradients, feed_dict={self.state: state, self.action: action})[0]

    def trainstep(self, state, action, rewards, next_state, next_action, terminals):
        target = self.next_q(rewards, next_state, next_action, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.state: state, self.action: action})

    def coldstart(self, state, action):
        target = [[-4] for _ in range(len(x))]
        return self.session.run(self.train_step, feed_dict={self.target: target, self.state: state, self.action: action})

    def __exit__(self):
        self.session.close()


class Actor(object):

    def __init__(self, nn, parameters, discount=.95):
        self.discount = tf.constant(discount)

        self._setup_q_calculation(nn, parameters)
        self._setup_train_step(nn, parameters)
        self.session = tf.Session()
        init = tf.global_variables_initializer()
        self.session.run(init)

    def _setup_train_step(self, nn, parameters):
        self.critic_gradients = tf.placeholder(
            tf.float32, [None, nn.out.get_shape()[1]])
        self.trainable_variables = nn.variables()

        self.gradients = tf.gradients(
            nn.out, self.trainable_variables, -self.critic_gradients)

        self.train_step = tf.train.AdamOptimizer(
            learning_rate=parameters['learning_rate']).apply_gradients(zip(self.gradients, self.trainable_variables))

    def _setup_q_calculation(self, nn, parameters):
        self.x = nn.x
        self.act = tf.nn.tanh(nn.out) * parameters['dim']

    def action(self, state):
        return self.session.run(self.act, feed_dict={self.x: state})[0]

    def actions(self, state):
        return self.session.run(self.act, feed_dict={self.x: state})

    def train(self, state, gradients):
        self.session.run(self.train_step, feed_dict={
                         self.x: state, self.critic_gradients: map(lambda x, d=len(state): x / d, gradients)})

    def __exit__(self):
        self.session.close()
