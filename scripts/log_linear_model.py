import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


def full_states(d_j):

	initial_states = tf.expand_dims(tf.cast(tf.linspace(0, d_j[0]-1, d_j[0]), dtype = tf.int32), axis = -1)

	for k in range(1, tf.shape(d_j)[0]):
		replicate_initial_states = tf.concat([initial_states for i in range(d_j[k])], axis = 0)

		new_states = tf.concat([i*tf.ones(tf.shape(initial_states)[0], dtype = tf.int32) for i in tf.cast(tf.linspace(0, d_j[k]-1, d_j[k]), dtype = tf.int32)], axis = 0)
		new_states = tf.expand_dims(new_states, axis = -1)

		initial_states = tf.concat((replicate_initial_states, new_states), axis = -1)

	return initial_states

def poisson_loss(x, y, pi, beta):

	log_lambda_k = tf.einsum("ki,i->k", x, beta)

	return -tf.reduce_sum(tfp.distributions.Poisson(rate = pi*tf.math.exp(log_lambda_k)).log_prob(y))

tf.function(jit_compile=True)
def grad_poisson_loss(x, y, pi, beta):

	with tf.GradientTape() as g:
		loss = poisson_loss(x, y, pi, beta)

	return loss, g.gradient(loss, [beta])

def log_linear_disclosure_risk(beta_0, pi, x, y, optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), n_iteration = 2000):

	loss_history = []
	for i in range(n_iteration):

		# print(i)
		loss, gradients = grad_poisson_loss(x, y, pi, beta_0)

		loss_history.append(loss)

		optimizer.apply_gradients(zip(gradients, [beta_0]))

	log_lambda_k = tf.einsum("ki,i->k", tf.gather(x, tf.where(y==1)[:,0], axis = 0), beta_0)

	return loss_history, beta_0, tf.reduce_sum(tfp.distributions.Poisson(rate = (1-pi)*tf.math.exp(log_lambda_k)).prob(0))

def penalized_poisson_loss(x, y, lambda_0, pi, beta):

	log_lambda_k = tf.einsum("ki,i->k", x, beta)

	return -tf.reduce_sum(tfp.distributions.Poisson(rate = pi*tf.math.exp(log_lambda_k)).log_prob(y)) + lambda_0*tf.reduce_sum(tf.math.abs(beta))

tf.function(jit_compile=True)
def penalized_grad_poisson_loss(x, y, lambda_0, pi, beta):

	with tf.GradientTape() as g:
		loss = penalized_poisson_loss(x, y, lambda_0, pi, beta)

	return loss, g.gradient(loss, [beta])

def penalized_log_linear_disclosure_risk(beta_0, pi, x, y, lambda_0, optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), n_iteration = 4000):

	loss_history = []
	for i in range(n_iteration):

		# print(i)
		loss, gradients = penalized_grad_poisson_loss(x, y, lambda_0, pi, beta_0)

		loss_history.append(loss)

		optimizer.apply_gradients(zip(gradients, [beta_0]))

	log_lambda_k = tf.einsum("ki,i->k", tf.gather(x, tf.where(y==1)[:,0], axis = 0), beta_0)

	return loss_history, beta_0, tf.reduce_sum(tfp.distributions.Poisson(rate = (1-pi)*tf.math.exp(log_lambda_k)).prob(0))