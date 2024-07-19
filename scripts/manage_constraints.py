import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

def nonempy_intersection_index(S_d, mu):
    expanded_mu = tf.expand_dims(mu, axis = 0)*tf.ones(tf.shape(S_d), dtype = tf.int32)

    S_d_compare_mu = tf.where(S_d==0,         expanded_mu, S_d)
    mu_compare_S_d = tf.where(expanded_mu==0, S_d,         expanded_mu)

    return tf.where(tf.reduce_all(S_d_compare_mu==mu_compare_S_d, axis = 1))[:,0]

def what_to_expand(mu, domain_j, ComparList):
	expanded_mu               = tf.expand_dims(mu, axis = 0)*tf.ones(tf.shape(ComparList), dtype = tf.int32)
	bool_components_to_expand = tf.reduce_all(tf.stack((ComparList!=0, expanded_mu==0), axis =-1), axis = -1)
	bool_components_to_expand = tf.reduce_any(bool_components_to_expand, axis =0)

	components_to_expand      = tf.where(bool_components_to_expand)[:,0]

	domain_to_expand          = tf.gather(domain_j, components_to_expand)

	return components_to_expand, domain_to_expand


def create_mu_to_add(mu, components_to_expand, domain_to_expand):

	total_n_states = tf.reduce_prod(domain_to_expand)
	states         = tf.cast(tf.expand_dims(tf.linspace(1, total_n_states, total_n_states), axis = 1)-1, dtype = tf.float32)

	mu_to_add = tf.zeros((total_n_states, tf.shape(mu)[0]))
	domain_to_expand = tf.cast(domain_to_expand, dtype = tf.float32)
	total_n_states   = tf.cast(total_n_states, dtype = tf.float32)

	for i in range(tf.shape(domain_to_expand)[0]):
		denominator_state = tf.cast(total_n_states/domain_to_expand[i], dtype = tf.float32)
		total_n_states    = denominator_state

		state_current = tf.floor(states/denominator_state)
		mu_to_add     = mu_to_add + (1+state_current)*tf.one_hot(components_to_expand[i], tf.shape(mu)[0])

		states = states - state_current*denominator_state

	return tf.cast(mu_to_add, dtype = tf.int32)+tf.expand_dims(mu, axis = 0)

def check_disjoint(S_d, mu_to_add):

	accepted_mu_to_add = []
	for n in range(tf.shape(mu_to_add)[0]):

		ComparList = tf.gather(S_d, nonempy_intersection_index(S_d, mu_to_add[n,:]), axis = 0)
		if tf.shape(ComparList)[0]==0:
			accepted_mu_to_add.append(mu_to_add[n,:])

	return accepted_mu_to_add



def iteration_disjoint_constraint(domain_j, current_constraint):

	# sort by number of explained cells
	free_cells_dim = tf.cast(current_constraint==0, dtype = tf.int32)*tf.expand_dims(domain_j, axis = 0)
	free_cells = tf.reduce_prod(tf.where(free_cells_dim==0, tf.ones(tf.shape(free_cells_dim), dtype = tf.int32), free_cells_dim), axis =1)

	sorting_index = tf.argsort(free_cells)[::-1]
	current_constraint_sorted = tf.gather(current_constraint, sorting_index, axis = 0)

	# initialize everything
	S_d = current_constraint_sorted[0:1,:]
	Pending = current_constraint_sorted[1:,:]

	i = 0
	while tf.shape(Pending)[0]!=0:

		i = i +1

		print("Left in pending ", tf.shape(Pending)[0].numpy())

		mu = Pending[0,:]
		Pending = Pending[1:,:]

		ComparList = tf.gather(S_d, nonempy_intersection_index(S_d, mu), axis = 0)

		if tf.shape(ComparList)[0]==0:

			S_d     = tf.concat((S_d, tf.expand_dims(mu, axis = 0)), axis = 0)

		else:

			components_to_expand, domain_to_expand = what_to_expand(mu, domain_j, ComparList)

			mu_to_add = create_mu_to_add(mu, components_to_expand, domain_to_expand)

			mu_to_add = check_disjoint(ComparList, mu_to_add)

			if len(mu_to_add)!=0:
				S_d = tf.concat((S_d, tf.stack(mu_to_add, axis = 0)), axis = 0)

	return S_d

# if __name__ == "__main__":
      
# 	from manage_constraints import *

# 	domain_j = tf.convert_to_tensor([2, 10, 3, 5, 2], dtype = tf.int32)

# 	current_constraint = tf.convert_to_tensor([[2, 1, 0, 0, 0],
# 						[2, 0, 2, 0, 0],
# 						[2, 0, 0, 3, 0],
# 						[0, 0, 0, 3, 1]], dtype = tf.int32)
      

# 	test = iteration_disjoint_constraint(domain_j, current_constraint)

