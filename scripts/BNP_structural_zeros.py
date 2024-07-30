import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from BNP import BNP_MCMC_initialization
from mixed_membership_model import *

@tf.function(jit_compile=True)
def middle_sampling(prob, prev_0_count, n_max, seed_to_use = None):

	sample = tfp.distributions.Categorical(probs = prob).sample(n_max, seed = seed_to_use)
	sample_0 = tf.where(sample==0, tf.ones(tf.shape(sample)), tf.zeros(tf.shape(sample)))
	current_0_count  = prev_0_count + tf.math.cumsum(sample_0)
	prev_0_count     = current_0_count[-1]

	return sample, sample_0, current_0_count, prev_0_count

# sample from Negative BInomial and then from Multinomial
# @tf.function
# def NegativeMultinomial(n, prob_constr, seed_sim = None, max_val = 10):

# 	C = tf.shape(prob_constr)[0]

# 	p_0 = 1 - tf.reduce_sum(prob_constr, keepdims=True)
# 	prob = tf.concat((p_0, prob_constr), axis  = 0)

# 	n_max = n*max_val

# 	seed_to_use, seed_carry = tfp.random.split_seed( seed_sim, n=2, salt='seed_negative_multinomial')

# 	prev_sample_freq = tf.zeros(C+1)
# 	prev_0_count     = tf.zeros(n_max)

# 	sample, sample_0, current_0_count, prev_0_count = middle_sampling(prob, prev_0_count, n_max)
# 	while tf.shape(tf.where(current_0_count==tf.cast(n, dtype = tf.float32)))[0]==0:

# 		seed_to_use, seed_carry = tfp.random.split_seed( seed_carry, n=2, salt='seed_negative_multinomial')
		
# 		prev_sample_freq = prev_sample_freq + tf.reduce_sum(tf.one_hot(sample, C+1), axis = 0)

# 		sample, sample_0, current_0_count, prev_0_count = middle_sampling(prob, prev_0_count, n_max)

# 	index = tf.where(current_0_count==tf.cast(n, dtype = tf.float32))[0,0]
# 	prev_sample_freq = prev_sample_freq + tf.reduce_sum(tf.one_hot(sample[:(index+1)], C+1), axis = 0)

# 	return prev_sample_freq[:]

@tf.function
def NegativeMultinomial(n, prob_constr, seed_sim = None, max_val = 10):

	C = tf.shape(prob_constr)[0]

	p_0 = 1 - tf.reduce_sum(prob_constr, keepdims=True)
	prob = tf.concat((p_0, prob_constr), axis  = 0)

	n_max = n*max_val

	seed_to_use, seed_carry = tfp.random.split_seed( seed_sim, n=2, salt='seed_negative_multinomial')

	prev_sample_freq = tf.zeros(C+1)
	prev_0_count     = tf.zeros(n_max)

	sample, sample_0, current_0_count, prev_0_count = middle_sampling(prob, prev_0_count, n_max)
	counter = n*max_val
	while tf.shape(tf.where(current_0_count==tf.cast(n, dtype = tf.float32)))[0]==0 and counter<(100*n):

		seed_to_use, seed_carry = tfp.random.split_seed( seed_carry, n=2, salt='seed_negative_multinomial')
		
		prev_sample_freq = prev_sample_freq + tf.reduce_sum(tf.one_hot(sample, C+1), axis = 0)

		sample, sample_0, current_0_count, prev_0_count = middle_sampling(prob, prev_0_count, n_max)

		counter = counter + n*max_val

	if len(tf.where(current_0_count==tf.cast(n, dtype = tf.float32)))!=0:
		index = tf.where(current_0_count==tf.cast(n, dtype = tf.float32))[0,0]
		prev_sample_freq = prev_sample_freq + tf.reduce_sum(tf.one_hot(sample[:(index+1)], C+1), axis = 0)
	else:
		prev_sample_freq = prev_sample_freq + tf.reduce_sum(tf.one_hot(sample[:-1], C+1), axis = 0)

	return prev_sample_freq[:]

	# p_c = prob_constr

	# seed_to_use = tfp.random.split_seed( seed_sim, n=2, salt='seed_negative_multinomial')

	# n0 = tfp.distributions.NegativeBinomial(total_count = tf.cast(n, tf.float32), probs = tf.reduce_sum(p_c)).sample(seed = seed_to_use[0])
	# n_SZ_no0 = tfp.distributions.Multinomial(total_count = n0, probs = p_c/tf.reduce_sum(p_c)).sample(seed = seed_to_use[1])
	# n_SZ = tf.concat((tf.cast(tf.expand_dims(n, axis = 0), tf.float32), n_SZ_no0), axis = 0)

	# return n_SZ


# Step 1 on old mixtures
def unobserved_step_1_old_mixtures(one_n_j, unobserved_X, pi_i_0k_full, theta_k_full, seed_s1_1):
    
    one_hot_X = tf.one_hot(unobserved_X, tf.shape(one_n_j)[1])    

    theta_jxk = tf.einsum("ijn,kjn->ijk", one_hot_X, theta_k_full)
    
    prob_ijnk = tf.einsum("ik,ijk->ijk", pi_i_0k_full[:,1:], theta_jxk)
    prob_ijn0 = pi_i_0k_full[:,0:1]/tf.expand_dims(tf.reduce_sum(one_n_j, axis = 1), axis = 0)

    # TODO we could use only pi for the placeholder and do everything in here
    # if a placeholder is assigned to a new mixture then sample from a uniform with probabilitites 1/n_j
    prob_ijn0k = tf.concat((tf.expand_dims(prob_ijn0, axis = -1), prob_ijnk), axis = -1)
    prob_ijn0k = prob_ijn0k/tf.reduce_sum(prob_ijn0k, axis = -1, keepdims = True)

    prob_ijn0k = tf.where((tf.expand_dims(unobserved_X, axis = -1)*tf.ones(tf.shape(prob_ijn0k), dtype = tf.int32))==-1, tf.expand_dims(pi_i_0k_full, axis = 1)*tf.ones(tf.shape(prob_ijn0k)), prob_ijn0k)

    Z_ij = tfp.distributions.Categorical(probs = prob_ijn0k).sample(seed = seed_s1_1)

    return Z_ij

# Step 1
def unobserved_step_1(one_n_j, K_current, unobserved_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i, seed_s1):

	seed_s1_1, seed_s1_2 = tfp.random.split_seed( seed_s1, n=2, salt='step_1_split')

	Z_ij = unobserved_step_1_old_mixtures(one_n_j, unobserved_X, pi_i_0k_full, theta_k_full, seed_s1_1)

	indexes_new_assignment = tf.where(Z_ij==0)

	if tf.reduce_any(Z_ij==0):

		# states = tf.zeros(1, dtype = tf.int32)

		seed_s1_2_to_use, seed_s1_2_to_carry = tfp.random.split_seed( seed_s1_2, n=2, salt='seed_s1_2_to_carry_split')

		beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, 
											beta_k_00_full, pi_i_0k_full, Z_ij, seed_s1_2_to_use)

		K_current = K_current+1
		states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

		counter = 0
		while tf.reduce_any(Z_ij==0):

			counter = counter + 1
			if counter>1000:
				raise ValueError("Something wrong")

			seed_s1_2_to_use_first, seed_s1_2_to_use_second, seed_s1_2_to_carry = tfp.random.split_seed( seed_s1_2_to_carry, n=3, salt='seed_s1_2_to_carry_split')

			indexes_new_assignment = tf.where(Z_ij==0)
			index_rows    = indexes_new_assignment[:,0]

			probability_assign_same_mixture = tf.gather(tf.gather(pi_i_0k_full, index_rows, axis =0), states, axis =1)
			probability_assign_same_mixture = probability_assign_same_mixture/tf.reduce_sum(probability_assign_same_mixture, axis = 1, keepdims = True)

			assignment = tfp.distributions.Categorical(probs = probability_assign_same_mixture).sample(seed = seed_s1_2_to_use_first)
			state_assignment = tf.gather(states, assignment)

			Z_ij = tf.tensor_scatter_nd_update(Z_ij, indexes_new_assignment, state_assignment)

			indexes_new_assignment = tf.where(Z_ij==0)

			if tf.reduce_any(Z_ij==0):

				beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, 
													beta_k_00_full, pi_i_0k_full, Z_ij, seed_s1_2_to_use_second)

				K_current = K_current+1
				states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

	return beta_k_00_full, pi_i_0k_full, Z_ij-1, K_current

def SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size, seed_SZ):   

	seed_step_tau_1, seed_step_tau_2_carry = tfp.random.split_seed( seed_SZ, n=2, salt='seed_split_for_tau_mc')

	sub_alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(mc_sample_size, seed = seed_step_tau_1)

	sub_dirichlet_concentration = tf.expand_dims(sub_alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0)

	seed_step_tau_2, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau_2_carry, n=2, salt='seed_split_for_tau_mc_carry')
	sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample(seed = seed_step_tau_2)
	while tf.reduce_any(tf.math.is_nan(sub_pi_0k)):
		seed_step_tau_2, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau_2_carry, n=2, salt='seed_split_for_tau_mc_carry')
		sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample(seed = seed_step_tau_2)

	one_hot_disjoint_constraints = tf.one_hot(disjoint_constraints, tf.shape(one_n_j)[1])
	sub_theta = tf.einsum("kjn,cjn->ckj", theta_k_full, one_hot_disjoint_constraints)

	sum_pi_theta = tf.einsum("mk,ckj->mcj", sub_pi_0k[...,1:], sub_theta)
	pi_n     = tf.expand_dims(sub_pi_0k[...,0:1], axis = -1)*(1/tf.expand_dims(tf.expand_dims((tf.reduce_sum(one_n_j, axis = 1)), axis = 0), axis = 0))

	pi_prob = sum_pi_theta + pi_n
	pi_prob_mask = tf.where(tf.expand_dims(disjoint_constraints, 0)==-1, tf.ones(tf.shape(pi_prob)), pi_prob)
	mc_sample_prob = tf.reduce_prod(pi_prob_mask, axis = -1) 

	return tf.reduce_mean(mc_sample_prob, axis = 0), sub_pi_0k

@tf.function(jit_compile=True)
def SZ_XZ_counting_cheap(one_hot_disjoint_constraints_with_placeholders, n_SZ, theta_k_full, sub_pi_0k):
	n_SZ_disjoint_constraints_with_placeholders = tf.einsum("cjn,c->cjn", one_hot_disjoint_constraints_with_placeholders, n_SZ[1:])

	constrained_theta = tf.einsum("cjn,kjn->kjn", n_SZ_disjoint_constraints_with_placeholders, theta_k_full)
	
	unobserved_XZ_counting_cheap = tf.math.round(tf.reduce_mean(tf.einsum("mk,kjn->mkjn", sub_pi_0k[:,1:], constrained_theta), axis = 0))

	return unobserved_XZ_counting_cheap

@tf.function(jit_compile=True)
def SZ_m_ik_cheap(one_hot_disjoint_constraints, theta_k_full, disjoint_constraints, J, alpha_0, beta_k_00_full, sub_pi_0k, seed_s789_splitted, n_SZ):
	theta_c = tf.einsum("cjn,kjn->ckj", one_hot_disjoint_constraints, theta_k_full)
	theta_c_with_ones = tf.where(tf.expand_dims(disjoint_constraints, axis = 1)==-1, tf.ones(tf.shape(theta_c)), theta_c)

	grid_j = tf.cast(tf.linspace(1, J, J), tf.float32)
	dir_process_prob = tf.expand_dims(alpha_0*beta_k_00_full[1:], axis = -1)/(tf.expand_dims(alpha_0*beta_k_00_full[1:], axis = -1) + tf.expand_dims(grid_j, axis = 0) - 1)
	dir_process_prob = tf.where(tf.expand_dims(grid_j, axis = 0)==1, tf.ones(tf.shape(dir_process_prob)), dir_process_prob)
	cumsum_dir_process_prob = tf.math.cumsum(dir_process_prob, axis = -1)

	# iterations = tf.cast(tf.shape(sub_pi_0k)[0]/100, tf.int32)
	# seed_for_loop = tfp.random.split_seed( seed_s789_splitted[2], n=iterations, salt='step_789_split_within_for_loops')
	# def body(input, iteration):
	# 	print("here")

	# 	bernoulli_trial = tf.einsum("mk,ckj->mckj", sub_pi_0k[iteration*100:(iteration+1)*100,1:], theta_c_with_ones)

	# 	print("here MF1")

	# 	sample_n_idotk_curr_sample = tf.reduce_sum(tfp.distributions.Bernoulli(probs = bernoulli_trial).sample(seed = seed_for_loop[iteration]), axis = -1)

	# 	print("here MF2")

	# 	output = tf.einsum("kj,mckj->mck", cumsum_dir_process_prob, tf.one_hot(sample_n_idotk_curr_sample-1, J))

	# 	print("here MF3")

	# 	return output
	
	# sample_sum_to_n_idotk = tf.scan(body, tf.range(iterations), initializer=tf.zeros((100, tf.shape(one_hot_disjoint_constraints)[0], tf.shape(theta_k_full)[0])))

	# reshaped_sample_sum_to_n_idotk = tf.reshape(sample_sum_to_n_idotk, (-1, tf.shape(sample_sum_to_n_idotk)[2], tf.shape(sample_sum_to_n_idotk)[3]))

	# unobserved_m_ik_cheap = tf.math.round(tf.reduce_sum(tf.expand_dims(n_SZ[1:], axis = -1)*tf.reduce_mean(reshaped_sample_sum_to_n_idotk, axis = 0), axis = 0))

	# sample_sum_to_n_idotk = []
	# for iteration in range(iterations):
	# 	bernoulli_trial = tf.einsum("mk,ckj->mckj", sub_pi_0k[iteration*100:(iteration+1)*100,1:], theta_c_with_ones)

	# 	sample_n_idotk_curr_sample = tf.reduce_sum(tfp.distributions.Bernoulli(probs = bernoulli_trial).sample(seed = seed_for_loop[iteration]), axis = -1)
	# 	sample_sum_to_n_idotk.append(tf.einsum("kj,mckj->mck", cumsum_dir_process_prob, tf.one_hot(sample_n_idotk_curr_sample-1, J)))

	# unobserved_m_ik_cheap = tf.math.round(tf.reduce_sum(tf.expand_dims(n_SZ[1:], axis = -1)*tf.reduce_mean(tf.concat(sample_sum_to_n_idotk, axis = 0), axis = 0), axis = 0))


	bernoulli_trial = tf.einsum("mk,ckj->mckj", sub_pi_0k[:,1:], theta_c_with_ones)

	sample_n_idotk_curr_sample = tf.reduce_sum(tfp.distributions.Bernoulli(probs = bernoulli_trial).sample(seed = seed_s789_splitted[2]), axis = -1)
	sample_sum_to_n_idotk = tf.einsum("kj,mckj->mck", cumsum_dir_process_prob, tf.one_hot(sample_n_idotk_curr_sample-1, J))

	unobserved_m_ik_cheap = tf.math.round(tf.reduce_sum(tf.expand_dims(n_SZ[1:], axis = -1)*tf.reduce_mean(sample_sum_to_n_idotk, axis = 0), axis = 0))

	return unobserved_m_ik_cheap

def SZ_step_789(disjoint_constraints, one_n_j, alpha_0, a, b, K_current, beta_k_00_full, theta_k_full, n, seed_s789):

	nJ = tf.shape(one_n_j)[1]
	one_hot_disjoint_constraints = tf.one_hot(disjoint_constraints, nJ)
	one_hot_disjoint_constraints_with_placeholders = tf.where(tf.expand_dims(disjoint_constraints, axis = -1)==-1, tf.ones(tf.shape(one_hot_disjoint_constraints)), one_hot_disjoint_constraints)

	n_mc_samples = 500
	n_max = 10
	J = tf.shape(one_n_j)[0]
     
	seed_s789_splitted = tfp.random.split_seed( seed_s789, n=3, salt='step_789_split')
 
	p_c, sub_pi_0k = SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, n_mc_samples, seed_s789_splitted[0])

	n_0 = tf.cast(n, tf.float32)*tf.reduce_sum(p_c)/(1-tf.reduce_sum(p_c))

	# if n_0>500*tf.cast(n, tf.float32):
	# 	n_SZ_no_zero = tf.math.round(n_0*p_c)
	# 	n_SZ = tf.concat((tf.expand_dims(tf.cast(n, tf.float32), axis = 0), n_SZ_no_zero), axis = 0)

	# else:
	n_SZ = NegativeMultinomial(n, p_c, seed_s789_splitted[1], n_max)

	unobserved_XZ_counting_cheap = SZ_XZ_counting_cheap(one_hot_disjoint_constraints_with_placeholders, n_SZ, theta_k_full, sub_pi_0k)

	unobserved_m_ik_cheap = SZ_m_ik_cheap(one_hot_disjoint_constraints, theta_k_full, disjoint_constraints, J, alpha_0, beta_k_00_full, sub_pi_0k, seed_s789_splitted, n_SZ)

	return tf.cast(unobserved_m_ik_cheap, tf.int32), unobserved_XZ_counting_cheap, beta_k_00_full, K_current


# # Mixture cutter
def SZ_mixture_cutter(n_i_dot_k, one_hot_Z, beta_k_00_full, pi_i_0k_full, unobserved_m_dotk, unobserved_XZ_counting):

    K_max = tf.shape(one_hot_Z)[-1]

    condition_1 = tf.reduce_sum(n_i_dot_k, axis =0)!=0
    condition_2 = unobserved_m_dotk!=0
    condition = tf.reduce_any(tf.stack((condition_1, condition_2)), axis = 0)

    non_zero_index = tf.cast(tf.where(condition)[:,0], dtype = tf.int32)

    new_one_hot_Z = tf.gather(one_hot_Z, non_zero_index, axis = -1)
    new_n_i_dot_k = tf.gather(n_i_dot_k, non_zero_index, axis = -1)

    unobserved_m_dotk      = tf.gather(unobserved_m_dotk,       non_zero_index, axis = -1)
    unobserved_XZ_counting = tf.gather(unobserved_XZ_counting,  non_zero_index, axis = 0)

    non_zero_index_plus_0 = tf.concat((tf.zeros(1, dtype = tf.int32), 1+non_zero_index), axis = 0)

    new_beta_k_00_full = tf.gather(beta_k_00_full, non_zero_index_plus_0, axis = -1)
    new_pi_i_0k_full   = tf.gather(pi_i_0k_full,   non_zero_index_plus_0, axis = -1)

    new_K_current = tf.shape(new_n_i_dot_k)[1]

    new_one_hot_Z_zeros      = tf.zeros((tf.shape(new_one_hot_Z)[0], tf.shape(new_one_hot_Z)[1], K_max- tf.shape(new_one_hot_Z)[2]))
    new_beta_k_00_full_zeros = tf.zeros((K_max- tf.shape(new_one_hot_Z)[2]))
    new_pi_i_0k_full_zeros   = tf.zeros((tf.shape(pi_i_0k_full)[0], K_max- tf.shape(new_one_hot_Z)[2]))

    unobserved_m_dotk_zeros = tf.zeros((K_max - tf.shape(unobserved_m_dotk)), dtype = tf.int32)
    unobserved_XZ_counting_zeros = tf.zeros((K_max - tf.shape(unobserved_XZ_counting)[0], tf.shape(unobserved_XZ_counting)[1], tf.shape(unobserved_XZ_counting)[2]), dtype = tf.float32)

    return tf.concat((new_one_hot_Z, new_one_hot_Z_zeros), axis = -1), tf.concat((new_beta_k_00_full, new_beta_k_00_full_zeros), axis = -1), tf.concat((new_pi_i_0k_full, new_pi_i_0k_full_zeros), axis = -1), tf.cast(new_K_current+1, dtype = tf.int32), tf.concat((unobserved_m_dotk, unobserved_m_dotk_zeros), axis = -1), tf.concat((unobserved_XZ_counting, unobserved_XZ_counting_zeros), axis = 0)

# Step 1 on old mixtures
@tf.function(jit_compile=True)
def SZ_step_1_old_mixtures(one_n_j, one_hot_X, pi_i_0k_full, theta_k_full, seed_s1_1):

    theta_jxk = tf.einsum("ijn,kjn->ijk", one_hot_X, theta_k_full)
    
    prob_ijnk = tf.einsum("ik,ijk->ijk", pi_i_0k_full[:,1:], theta_jxk)
    prob_ijn0 = pi_i_0k_full[:,0:1]/tf.expand_dims(tf.reduce_sum(one_n_j, axis = 1), axis = 0)

    prob_ijn0k = tf.concat((tf.expand_dims(prob_ijn0, axis = -1), prob_ijnk), axis = -1)
    prob_ijn0k = prob_ijn0k/tf.reduce_sum(prob_ijn0k, axis = -1, keepdims = True)

    Z_ij = tfp.distributions.Categorical(probs = prob_ijn0k).sample(seed = seed_s1_1)

    return Z_ij

# Step 1 on old mixtures
@tf.function(jit_compile = True)
def SZ_step_1_new_mixture_assignment(index_new_assignment_current, K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij, seed_new_mixture):

    seed_new_mixture_split = tfp.random.split_seed( seed_new_mixture, n=2, salt='step_new_mixture')

    new_K_current = K_current + 1

    nu_0 = tfp.distributions.Beta(concentration1=alpha_0, concentration0=1.).sample(seed = seed_new_mixture_split[0])
    nu_i = tfp.distributions.Beta(concentration1=alpha_i*beta_k_00_full[0]*nu_0, concentration0=alpha_i*beta_k_00_full[0]*(1-nu_0)).sample(seed = seed_new_mixture_split[1])

    new_beta_k_00_0   = beta_k_00_full[0]*nu_0
    new_beta_k_00_Kp1 = beta_k_00_full[0]*(1-nu_0)

    index_to_update = tf.expand_dims(tf.convert_to_tensor([0, new_K_current], dtype = tf.int32), axis = 1)
    elements_to_use = tf.expand_dims(tf.stack((new_beta_k_00_0, new_beta_k_00_Kp1)), axis = 1)

    new_beta_k_00_full = tf.squeeze(tf.tensor_scatter_nd_update(tf.expand_dims(beta_k_00_full, axis = 1), index_to_update, elements_to_use))

    one_hot_0   = tf.one_hot(0,               tf.shape(pi_i_0k_full)[1])
    one_hot_Kp1 = tf.one_hot(new_K_current, tf.shape(pi_i_0k_full)[1])
    new_pi_i_0k_0   = tf.expand_dims(pi_i_0k_full[:,0]*nu_i,     axis = 1)*one_hot_0
    new_pi_i_0k_Kp1 = tf.expand_dims(pi_i_0k_full[:,0]*(1-nu_i), axis = 1)*one_hot_Kp1

    new_pi_i_0k_full = (pi_i_0k_full*(1- (one_hot_0 + one_hot_Kp1))) + new_pi_i_0k_0 + new_pi_i_0k_Kp1

    new_Z_ij = tf.tensor_scatter_nd_update(Z_ij, tf.expand_dims(index_new_assignment_current, axis = 0), (new_K_current)*tf.ones((1), dtype = tf.int32))

    return new_beta_k_00_full, new_pi_i_0k_full, new_Z_ij

# Step 1
def SZ_step_1(one_n_j, K_current, one_hot_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i, seed_s1):

    seed_s1_1, seed_s1_2 = tfp.random.split_seed( seed_s1, n=2, salt='step_1_split')

    Z_ij = SZ_step_1_old_mixtures(one_n_j, one_hot_X, pi_i_0k_full, theta_k_full, seed_s1_1)

    indexes_new_assignment = tf.where(Z_ij==0)

    if tf.reduce_any(Z_ij==0):

        # states = tf.zeros(1, dtype = tf.int32)

        seed_s1_2_to_use, seed_s1_2_to_carry = tfp.random.split_seed( seed_s1_2, n=2, salt='seed_s1_2_to_carry_split')

        beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij, seed_s1_2_to_use)

        K_current = K_current+1
        states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

        counter = 0
        while tf.reduce_any(Z_ij==0):

            counter = counter + 1
            if counter>1000:
                raise ValueError("Something wrong")

            seed_s1_2_to_use_first, seed_s1_2_to_use_second, seed_s1_2_to_carry = tfp.random.split_seed( seed_s1_2_to_carry, n=3, salt='seed_s1_2_to_carry_split')

            indexes_new_assignment = tf.where(Z_ij==0)
            index_rows    = indexes_new_assignment[:,0]

            probability_assign_same_mixture = tf.gather(tf.gather(pi_i_0k_full, index_rows, axis =0), states, axis =1)
            probability_assign_same_mixture = probability_assign_same_mixture/tf.reduce_sum(probability_assign_same_mixture, axis = 1, keepdims = True)

            assignment = tfp.distributions.Categorical(probs = probability_assign_same_mixture).sample(seed = seed_s1_2_to_use_first)
            state_assignment = tf.gather(states, assignment)

            Z_ij = tf.tensor_scatter_nd_update(Z_ij, indexes_new_assignment, state_assignment)

            indexes_new_assignment = tf.where(Z_ij==0)

            if tf.reduce_any(Z_ij==0):

                beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij, seed_s1_2_to_use_second)

                K_current = K_current+1
                states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

    return beta_k_00_full, pi_i_0k_full, Z_ij-1, K_current

# Step 2
def SZ_step_2( n_i_dot_k, alpha_0, beta_k_00_full, J, seed_s2):

	# divide in batches to avoid OOM (structural zeros)
	n_batch    = tf.cast(tf.shape(n_i_dot_k)[0]/10000, dtype = tf.int32)+1
	batch_size = tf.cast(tf.shape(n_i_dot_k)[0]/n_batch, dtype = tf.int32)
	seed_s2_splitted = tfp.random.split_seed( seed_s2, n=n_batch, salt='table_batching')

	m_list = []

	for i in range(n_batch):
        
		if i<n_batch-1:
			n_i_dot_k_batch = n_i_dot_k[i*batch_size:(i+1)*batch_size,...]
		else:
			n_i_dot_k_batch = n_i_dot_k[i*batch_size:,...]

		concentration_chinese_restaurant = tf.expand_dims(alpha_0*beta_k_00_full[1:], axis = 0)*tf.ones(tf.shape(n_i_dot_k_batch))

		one_hot_n_i_dot_k = tf.one_hot(tf.cast(n_i_dot_k_batch, dtype = tf.int32), tf.cast(J+1, dtype = tf.int32))

		sequential_custumers_keep_0 = tf.math.cumsum(one_hot_n_i_dot_k[..., ::-1], axis = -1)[...,::-1]
		sequential_custumers = tf.math.cumsum(sequential_custumers_keep_0, axis =-1)-1
		sequential_custumers = sequential_custumers*sequential_custumers_keep_0
		sequential_custumers_no_zeros = sequential_custumers[...,1:]
		sequential_custumers_keep_no_zeros = sequential_custumers_keep_0[...,1:]

		prob_new_table = sequential_custumers_keep_no_zeros*(tf.expand_dims(concentration_chinese_restaurant, axis = -1)/(sequential_custumers_no_zeros - 1 + tf.expand_dims(concentration_chinese_restaurant, axis = -1)))
		prob_new_table = tf.stack((1- prob_new_table, prob_new_table), axis = -1)
		table_or_not_i = tfp.distributions.Categorical(probs=prob_new_table).sample(seed = seed_s2_splitted[i])
		m_ik_i = tf.reduce_sum(table_or_not_i, axis = -1)

		m_list.append(m_ik_i)
        
	return tf.concat((m_list), axis = 0)

# Step 3
@tf.function(jit_compile=True)
def SZ_step_3(m_ik, unobserved_m_dotk, alpha_0, seed_s3):

    concentration_beta_1k = tf.cast(tf.reduce_sum(m_ik, axis = 0), dtype = tf.float32) + tf.cast(unobserved_m_dotk, dtype = tf.float32)
    concentration_beta_0k = tf.concat((tf.expand_dims(alpha_0, axis = 0), concentration_beta_1k), axis = 0)

    beta_k_00_full = tfp.distributions.Dirichlet(concentration = concentration_beta_0k).sample(seed = seed_s3)

    return beta_k_00_full

# Step 4
@tf.function(jit_compile=True)
def SZ_step_4(n_i_dot_k, alpha_i, beta_k_00_full, seed_s4):

    n_i_dot_0k = tf.concat((tf.zeros((tf.shape(n_i_dot_k)[0], 1)), n_i_dot_k), axis = -1)
    concentration_pi_i = tf.expand_dims(alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0) + n_i_dot_0k

    pi_ik = tfp.distributions.Dirichlet(concentration = concentration_pi_i).sample(seed = seed_s4)

    return pi_ik

# Step 5
@tf.function(jit_compile=True)
def SZ_step_5(one_hot_X, one_n_j, one_hot_Z, unobserved_XZ_counting, seed_s5):

    dirichlet_lambda = tf.expand_dims(one_n_j, axis = 0) + tf.einsum("ijk,ijn->kjn", one_hot_Z, one_hot_X) + unobserved_XZ_counting

    theta_j_k = tfp.distributions.Dirichlet(concentration = dirichlet_lambda).sample(seed = seed_s5)

    return theta_j_k

# Step 6
def SZ_step_6_multiple(a_0, b_0, a, b, K_current, J, m_ik, unobserved_m_dotk, alpha_0, alpha_i, seed_s6):

    seed_s6 = tfp.random.split_seed( seed_s6, n=6, salt='step_6_seed_split')

    m_dot_dot = tf.reduce_sum(tf.cast(m_ik, dtype = tf.float32) + tf.cast(unobserved_m_dotk, dtype = tf.float32))
    eta_0     = tfp.distributions.Beta(concentration1 = alpha_0 + 1, concentration0 = m_dot_dot).sample(seed = seed_s6[0])
    bern_p_0  = m_dot_dot*(b_0 - tf.math.log(eta_0))/(tf.cast(K_current, dtype = tf.float32) + a_0 - 1 + m_dot_dot*(b_0 - tf.math.log(eta_0))) 
    s_0       = tf.cast(tfp.distributions.Bernoulli(probs = bern_p_0).sample(seed = seed_s6[1]), dtype = tf.float32)
    alpha_0  = tfp.distributions.Gamma( concentration = a_0 + tf.cast(K_current, dtype = tf.float32) - s_0, rate = b_0 - tf.math.log(eta_0)).sample(seed = seed_s6[2])

    m_i_dot   = tf.cast(tf.reduce_sum(m_ik, axis = -1), dtype = tf.float32)
    eta_i     = tfp.distributions.Beta( concentration1 = alpha_i+1, concentration0 = J ).sample(seed = seed_s6[3])
    bern_p_i  = J*(b - tf.math.log(eta_i))/(m_i_dot + a -1 + J*(b - tf.math.log(eta_i))) 
    s_i       = tf.cast(tfp.distributions.Bernoulli( probs = bern_p_i ).sample(seed = seed_s6[4]), dtype = tf.float32)
    alpha_i   = tfp.distributions.Gamma(concentration = a + m_i_dot - s_i, rate = b - tf.math.log(eta_i)).sample(seed = seed_s6[5])

    return alpha_0, alpha_i

# Estimate tau
@tf.function(jit_compile = True)
def noSZ_prob_estimator(X_ij, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size, seed_step_tau):

    freq_0 = fast_frequency(X_ij, one_n_j)

    subX_ij       = tf.gather(X_ij,       tf.where(freq_0==1)[:,0], axis =0)       

    seed_step_tau_1, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau, n=2, salt='seed_split_for_tau_mc')

    sub_alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(mc_sample_size, seed = seed_step_tau_1)

    sub_dirichlet_concentration = tf.expand_dims(sub_alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0)

    seed_step_tau_2, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau_2_carry, n=2, salt='seed_split_for_tau_mc_carry')
    sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample(seed = seed_step_tau_2)
    while tf.reduce_any(tf.math.is_nan(sub_pi_0k)):
        seed_step_tau_2, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau_2_carry, n=2, salt='seed_split_for_tau_mc_carry')
        sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample(seed = seed_step_tau_2)

    one_hot_X = tf.one_hot(subX_ij, tf.shape(one_n_j)[1])
    sub_theta = tf.einsum("kjn,ijn->ikj", theta_k_full, one_hot_X)

    pi_theta = tf.einsum("mk,ikj->mikj", sub_pi_0k[...,1:], sub_theta)
    pi_n     = tf.expand_dims(sub_pi_0k[...,0:1], axis = -1)*(1/tf.expand_dims(tf.expand_dims((tf.reduce_sum(one_n_j, axis = 1)), axis = 0), axis = 0))

    mc_sample_prob = tf.reduce_prod(tf.reduce_sum(pi_theta, axis = 2) + pi_n, axis = -1) 

    return mc_sample_prob

def SZ_step_mc_tau(MCMC_output, X_ij, disjoint_constraints, one_n_j, N, mc_sample_size, seed_step_tau):

    a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = MCMC_output

    n = tf.shape(X_ij)[0]

    seed_step_tau_to_split = tfp.random.split_seed( seed_step_tau, n=6, salt='seed_split_for_tau_mc_carry')

    mc_sample_prob_list = [noSZ_prob_estimator(X_ij, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size, seed_step_tau_to_split[i]) for i in range(5)]
    mc_sample_prob   = tf.concat(mc_sample_prob_list, axis = 0)

    prob_approx = tf.reduce_mean(mc_sample_prob, axis = 0)  

    p_c, _ = SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size, seed_step_tau_to_split[5])

    log_prob = tf.math.log(1-prob_approx)
    exponent = tf.cast((N-n), dtype = tf.float32)/(1 - tf.reduce_sum(p_c))
    M = tf.reduce_max(log_prob, axis = 0, keepdims = True)

    mc_tau = tf.math.exp(M[0]*exponent + tf.math.log(tf.reduce_sum(tf.math.exp(exponent*(log_prob - M)))))

    return mc_tau

def SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, initialization_MCMC, unobserved_m_dotk, unobserved_XZ_counting, seed_step, type_1, type_2):
     
	n = tf.shape(one_hot_X)[0]

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

	a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = initialization_MCMC

	seed_1, seed_2, seed_3, seed_4, seed_5, seed_6, seed_7 = tfp.random.split_seed( seed_step, n=7, salt='seed_step_split')
		
	J = tf.cast(tf.shape(one_hot_X)[1], dtype = tf.float32)

	beta_k_00_full, pi_i_0k_full, Z_ij, K_current = SZ_step_1(one_n_j, K_current, one_hot_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i, seed_1)

	one_hot_Z = tf.one_hot(Z_ij, K_max)
	n_i_dot_k = tf.reduce_sum(one_hot_Z, axis = 1)

	m_ik = SZ_step_2(n_i_dot_k, alpha_0, beta_k_00_full, J, seed_2)

	one_hot_Z, beta_k_00_full, pi_i_0k_full, K_current, unobserved_m_dotk, unobserved_XZ_counting = SZ_mixture_cutter(n_i_dot_k, one_hot_Z, beta_k_00_full, pi_i_0k_full, unobserved_m_dotk, unobserved_XZ_counting)
	n_i_dot_k = tf.reduce_sum(one_hot_Z, axis = 1)

	seed_s3, seed_s3_carry  = tfp.random.split_seed( seed_3, n=2, salt='seed_s3_carry_'+str(K_current))
	beta_k_00_full = SZ_step_3(m_ik, unobserved_m_dotk, alpha_0, seed_s3)

	counter = 0
	while tf.reduce_any(tf.math.is_nan(beta_k_00_full)) and counter<100:

		seed_s3, seed_s3_carry  = tfp.random.split_seed( seed_s3_carry, n=2, salt='seed_s3_carry_'+str(K_current))
		beta_k_00_full = SZ_step_3(m_ik, unobserved_m_dotk, alpha_0, seed_s3)

		counter = counter +1

	seed_s4, seed_s4_carry  = tfp.random.split_seed( seed_4, n=2, salt='seed_s4_carry_'+str(K_current))
	pi_i_0k_full = SZ_step_4(n_i_dot_k[:n,:], alpha_i, beta_k_00_full, seed_s4)

	counter = 0
	while tf.reduce_any(tf.math.is_nan(pi_i_0k_full)) and counter<100:

		seed_s4, seed_s4_carry  = tfp.random.split_seed( seed_s4_carry, n=2, salt='seed_s4_carry_'+str(K_current))
		pi_i_0k_full = SZ_step_4(n_i_dot_k[:n,:], alpha_i, beta_k_00_full, seed_s4)

		counter = counter +1

	seed_s5, seed_s5_carry  = tfp.random.split_seed( seed_5, n=2, salt='seed_s5_carry_'+str(K_current))
	theta_k_full = SZ_step_5(one_hot_X, one_n_j, one_hot_Z, unobserved_XZ_counting, seed_s5)

	counter = 0
	while tf.reduce_any(tf.math.is_nan(theta_k_full)) and counter<100:

		seed_s5, seed_s5_carry  = tfp.random.split_seed( seed_s5_carry, n=2, salt='seed_s5_carry_'+str(K_current))
		theta_k_full = SZ_step_5(one_hot_X, one_n_j, one_hot_Z, unobserved_XZ_counting, seed_s5)

		counter = counter +1

	if type_1 == "multiple":
		alpha_0, alpha_i = SZ_step_6_multiple(a_0, b_0, a, b, K_current, J, m_ik, unobserved_m_dotk, alpha_0, alpha_i, seed_6)
                  
	if type_2 == "fixed a,b":
		a, b = a_1, b_1
        
	return a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current

def SZ_BNP_MCMC_from_start(X_ij, disjoint_constraints, one_n_j, K_current, K_max, prior_parameters, MCMC_iterations, seed_MCMC, type_1, type_2):

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

	seed_initialization_1, seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=2, salt='seed_MCMC')

	n = tf.shape(X_ij)[0]
	one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

	# initialization 
	a, b = a_1, b_1
	alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = BNP_MCMC_initialization(a_0, b_0, a, b, one_n_j, n, K_max, K_current, seed_initialization_1)

	# alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = BNP_MCMC_initialization(a_0, b_0, a, b, one_n_j, n, K_max, K_current, None)
	# tf.reduce_sum(SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, 500, None)[0])

	seed_step_to_split  = tfp.random.split_seed( seed_step_to_split, n=MCMC_iterations, salt='seed_MCMC_step_per_iter')

	def body(input, t):

		(a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current) = input

		seed_step_inside  = tfp.random.split_seed( seed_step_to_split[t], n=2, salt='seed_MCMC_step_inside')

		mu  = tf.reduce_mean(alpha_i)
		mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
		hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
		hat_b = mu/(mu2 - tf.math.pow(mu, 2))
                  
		unobserved_m_dotk, unobserved_XZ_counting, beta_k_00_full, K_current = SZ_step_789(disjoint_constraints, one_n_j, alpha_0, hat_a, hat_b, K_current, beta_k_00_full, theta_k_full, n, seed_step_inside[0])

		input = (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current)

		output_t = SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, unobserved_m_dotk, unobserved_XZ_counting, seed_step_inside[1], type_1, type_2)

		return output_t
    
	output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current))

	return output

def SZ_BNP_MCMC_initialized(X_ij, disjoint_constraints, one_n_j, K_max, prior_parameters, initialization_MCMC, MCMC_iterations, seed_MCMC, type_1, type_2):

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

	n = tf.shape(X_ij)[0]
	one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

	seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=MCMC_iterations, salt='seed_MCMC_step_per_iter_not_init')

	def body(input, t):

		(a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current) = input

		seed_step_inside  = tfp.random.split_seed( seed_step_to_split[t], n=2, salt='seed_MCMC_step_inside')

		mu  = tf.reduce_mean(alpha_i)
		mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
		hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
		hat_b = mu/(mu2 - tf.math.pow(mu, 2))
                  
		unobserved_m_dotk, unobserved_XZ_counting, beta_k_00_full, K_current = SZ_step_789(disjoint_constraints, one_n_j, alpha_0, hat_a, hat_b, K_current, beta_k_00_full, theta_k_full, n, seed_step_inside[0])

		input = (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current)

		output_t = SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, unobserved_m_dotk, unobserved_XZ_counting, seed_step_inside[1], type_1, type_2)

		return output_t
    
	output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  initialization_MCMC)

	return output

def SZ_BNP_MCMC_initialized_tau(X_ij, disjoint_constraints, one_n_j, K_max, prior_parameters, 
				initialization_MCMC, MCMC_iterations, N, batch_size, seed_MCMC, type_1, type_2, type_tau):

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters
	
	n = tf.shape(X_ij)[0]
	J = tf.shape(one_n_j)[0]
	one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

	seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=MCMC_iterations, salt='seed_MCMC_step_per_iter_not_init_with_tau')
	tau = 0.

	def body(input, t):

		feed_MCMC, _ = input

		seed_step_without_tau_1, seed_step_without_tau_2, seed_step_with_tau = tfp.random.split_seed( seed_step_to_split[t], n=3, salt='without_and_with_tau')

		(a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current) = feed_MCMC

		mu  = tf.reduce_mean(alpha_i)
		mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
		hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
		hat_b = mu/(mu2 - tf.math.pow(mu, 2))
                  
		unobserved_m_dotk, unobserved_XZ_counting, beta_k_00_full, K_current = SZ_step_789(disjoint_constraints, one_n_j, alpha_0, hat_a, hat_b, K_current, beta_k_00_full, theta_k_full, n, seed_step_without_tau_1)

		input = (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current)

		output_t = SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, unobserved_m_dotk, unobserved_XZ_counting, seed_step_without_tau_2, type_1, type_2)

		if type_1=="single":
			pass
		else:
			if type_2 == "fixed a,b":
				alpha_i = output_t[3]
				mu  = tf.reduce_mean(alpha_i)
				mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
				hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
				hat_b = mu/(mu2 - tf.math.pow(mu, 2))

				output_t_tau = hat_a, hat_b, output_t[2], output_t[3], output_t[4], output_t[5], output_t[6], output_t[7]

			else:
				output_t_tau = output_t

			if type_tau=="Monte Carlo":
				tau = SZ_step_mc_tau(output_t_tau, X_ij, disjoint_constraints, one_n_j, N, batch_size, seed_step_with_tau)

		return output_t, tau
    
	MCMC_output, Tau = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (initialization_MCMC, tau))

	return MCMC_output, Tau
