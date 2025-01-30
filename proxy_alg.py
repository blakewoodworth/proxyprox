import jax
import jax.numpy as jnp
from optax.losses import sigmoid_binary_cross_entropy as optax_bce
# from optax.losses import softmax_cross_entropy_with_integer_labels as optax_sce
import scipy as sp
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from data import *

class GenericObjective:

    def __init__(self, train_data, test_data, args):
        '''
        train_data?
        test_data?
        '''
        self.function = self.get_loss()
        self.grad = jax.jit(jax.grad(self.function))
        self.value_and_grad = jax.jit(jax.value_and_grad(self.function))
        # self.value_and_grad = jax.value_and_grad(self.function)

        self.train_data = train_data
        self.test_data = test_data
        self.train_n = self.train_data[0].shape[0]
        self.test_n = self.test_data[0].shape[0]
        self.d = self.train_data[0].shape[1]

        self.args = args
        self.batch_size = args['batch_size']

        np.random.seed(1)
        self.reset_batch()

    def reset_batch(self):
        self.batch_order = np.random.permutation(self.train_n)
        self.batch_idx = 0

    def get_loss(self):

        def function(w, X, Y):
            return 0

        return function

    def get_batch(self):
        if self.batch_idx + self.batch_size > self.train_n:
            self.reset_batch()

        x = self.train_data[0][self.batch_order[self.batch_idx:self.batch_idx + self.batch_size]]
        y = self.train_data[1][self.batch_order[self.batch_idx:self.batch_idx + self.batch_size]]
        return x, y

    def stoch_value_and_grad(self, w):
        x, y = self.get_batch()
        return self.value_and_grad(w, x, y)

    def stoch_grad(self, w):
        x, y = self.get_batch()
        return self.grad(w, x, y)

    def evaluate(self, w):
        batch_size = min(10*self.batch_size, self.test_n)

        acc = 0.
        loss = 0.
        n_batches = 0
        for i in range(0, self.test_n, batch_size):
            next_i = min((i+1)*batch_size, self.test_n)
            x = self.test_data[0][i:next_i]
            y = self.test_data[1][i:next_i]

            loss += self.function(w, x, y)
            preds = 1*(jnp.dot(x, w) > 0)
            acc += jnp.mean(preds == y)
            n_batches += 1
        return loss / n_batches, acc / n_batches


class LeastSquares(GenericObjective):
    '''
    X batches: [b, d]
    Y batches: [b]
    w:         [d]
    '''
    def get_loss(self):

        def function(w, X, Y):
            return 0.5 * jnp.mean(jnp.power(jnp.dot(X, w) - Y, 2))

        return function


class LogisticRegression(GenericObjective):
    '''
    X batches: [b, d]
    Y batches: [b] (Integer labels)
    w:         [d]
    '''
    def get_loss(self):

        def function(w, X, Y):
            return jnp.mean(optax_bce(jnp.dot(X,w), Y))

        return function


class ProxyProx:

    def __init__(self, target, proxy, args):
        self.target = target
        self.proxy = proxy
        self.args = args

    def train(self):
        inv_eta = 1./self.args['eta']
        lr = self.args['lr']
        # momentum = self.args['momentum']
        w = jnp.zeros(self.target.d)

        loss, acc = self.target.evaluate(w)
        test_losses = [loss]
        test_accs = [acc]

        # wt_wtm1 = jnp.zeros(self.target.d)

        for k in range(self.args['n_outer']):

            w_k = w
            bias_correction = self.target.stoch_grad(w_k) - self.proxy.stoch_grad(w_k)

            for t in range(self.args['n_inner']):
                new_dir = self.proxy.stoch_grad(w) + bias_correction + inv_eta * (w - w_k)
                w_new = w - lr * new_dir # + momentum * wt_wtm1 
                # wt_wtm1 = w_new - w
                w = w_new           

            loss, acc = self.target.evaluate(w)
            if loss > 1000:
                return None, None
            test_losses.append(loss)
            test_accs.append(acc)

        return test_losses, test_accs



