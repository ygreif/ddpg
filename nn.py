import tensorflow as tf


class FullyConnectedLayer(object):

    def __init__(self, inp, dim, nonlinearity=False):
        self.W = tf.Variable(tf.random_normal(dim))
        self.b = tf.Variable(tf.constant(1.0, shape=(1, dim[1])))
        if nonlinearity:
            self.out = nonlinearity(tf.matmul(inp, self.W) + self.b)
        else:
            self.out = tf.matmul(inp, self.W) + self.b


class NeuralNetwork(object):

    def __init__(self, indim, enddim, hidden_layers=[10], nonlinearity=False, name='default'):
        self.layers = []
        self.x = tf.placeholder(tf.float32, [None, indim], name=name)
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


class Critic(object):

    def __init__(self, nn, parameters, discount):
        self.discount = tf.constant(discount)
        self.x == nn.x


class QApproximation(object):

    def __init__(self, nn, parameters, discount=.95):
        self.discount = tf.constant(discount)

        self._setup_q_calculation(nn)
        self._setup_next_q_calulcation(nn)
        self._setup_train_step(parameters)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _setup_q_calculation(self, nn):
        self.x = nn.x
        self.qprobs = tf.nn.softmax(nn.out)
        self.act = tf.argmax(nn.out, axis=1)
        self.q = tf.transpose(tf.reduce_max(nn.out, axis=1))

    def _setup_next_q_calulcation(self, nn):
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.terminal = tf.placeholder(tf.float32, [None, 1])

        self.update = self.q * \
            tf.transpose(self.terminal * self.discount) + tf.transpose(self.r)

    def _setup_train_step(self, parameters):
        self.target = tf.transpose(tf.placeholder(tf.float32, [None, 1]))
        self.loss = tf.reduce_sum(tf.square(self.target - self.q))

        if parameters.learner == 'adam':
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)
        else:
            self.train_step = tf.train.GradientDescentOptimizer(
                learning_rate=parameters.learning_rate).minimize(self.loss)

    def action(self, state):
        return self.session.run(self.act, feed_dict={self.x: state})[0]

    def calcq(self, rewards, next_state, terminals):
        return self.session.run(self.update, feed_dict={self.r: rewards, self.x: next_state, self.terminal: terminals})

    def storedq(self, state):
        return self.session.run(self.q, feed_dict={self.x: state})

    def calcloss(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.loss, feed_dict={self.target: target, self.x: state})

    def trainstep(self, state, action, rewards, next_state, terminals):
        target = self.calcq(rewards, next_state, terminals)
        return self.session.run(self.train_step, feed_dict={self.target: target, self.x: state})

    def __exit__(self):
        self.session.close()
