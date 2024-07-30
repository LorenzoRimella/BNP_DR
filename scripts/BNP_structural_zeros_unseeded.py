import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from mixed_membership_model import *

# MCMC
def BNP_MCMC_initialization(a_0, b_0, a, b, one_n_j, n_indiv, K_max, K_initial):

    # alpha_0
    alpha_0 = tfp.distributions.Gamma( concentration = a_0, rate = b_0).sample()

    # theta
    theta_k_full  = tfp.distributions.Dirichlet(concentration = one_n_j).sample(K_max)

    # alpha_i
    alpha_i       = tfp.distributions.Gamma( concentration = a, rate = b).sample(n_indiv)

    # beta
    beta_k_00_full= tfp.distributions.Dirichlet(concentration = tf.concat((tf.ones((K_initial+1)), 
                                                                           tf.zeros((K_max - K_initial))), axis = 0)).sample()

    # pi
    pi_i_0k_full  = tfp.distributions.Dirichlet(concentration = tf.concat((tf.ones((K_initial+1)), 
                                                                            tf.zeros((K_max - K_initial))), axis = 0)).sample(n_indiv)

    return alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, tf.cast(K_initial, dtype = tf.int32)

@tf.function(jit_compile=True)
def middle_sampling(prob, prev_0_count, n_max):

	sample = tfp.distributions.Categorical(probs = prob).sample(n_max)
	sample_0 = tf.where(sample==0, tf.ones(tf.shape(sample)), tf.zeros(tf.shape(sample)))
	current_0_count  = prev_0_count + tf.math.cumsum(sample_0)
	prev_0_count     = current_0_count[-1]

	return sample, sample_0, current_0_count, prev_0_count

# sample from Negative BInomial and then from Multinomial
@tf.function
def NegativeMultinomial(n, prob_constr, max_val = 10):

	C = tf.shape(prob_constr)[0]

	p_0 = 1 - tf.reduce_sum(prob_constr, keepdims=True)
	prob = tf.concat((p_0, prob_constr), axis  = 0)

	n_max = n*max_val

	prev_sample_freq = tf.zeros(C+1)
	prev_0_count     = tf.zeros(n_max)

	sample, sample_0, current_0_count, prev_0_count = middle_sampling(prob, prev_0_count, n_max)
	while tf.shape(tf.where(current_0_count==tf.cast(n, dtype = tf.float32)))[0]==0:
		
		prev_sample_freq = prev_sample_freq + tf.reduce_sum(tf.one_hot(sample, C+1), axis = 0)

		sample, sample_0, current_0_count, prev_0_count = middle_sampling(prob, prev_0_count, n_max)

	index = tf.where(current_0_count==tf.cast(n, dtype = tf.float32))[0,0]
	prev_sample_freq = prev_sample_freq + tf.reduce_sum(tf.one_hot(sample[:(index+1)], C+1), axis = 0)

	return prev_sample_freq[:]

# Step 1 on old mixtures
def unobserved_step_1_old_mixtures(one_n_j, unobserved_X, pi_i_0k_full, theta_k_full):
    
    one_hot_X = tf.one_hot(unobserved_X, tf.shape(one_n_j)[1])    

    theta_jxk = tf.einsum("ijn,kjn->ijk", one_hot_X, theta_k_full)
    
    prob_ijnk = tf.einsum("ik,ijk->ijk", pi_i_0k_full[:,1:], theta_jxk)
    prob_ijn0 = pi_i_0k_full[:,0:1]/tf.expand_dims(tf.reduce_sum(one_n_j, axis = 1), axis = 0)

    prob_ijn0k = tf.concat((tf.expand_dims(prob_ijn0, axis = -1), prob_ijnk), axis = -1)
    prob_ijn0k = prob_ijn0k/tf.reduce_sum(prob_ijn0k, axis = -1, keepdims = True)

    prob_ijn0k = tf.where((tf.expand_dims(unobserved_X, axis = -1)*tf.ones(tf.shape(prob_ijn0k), dtype = tf.int32))==-1, tf.expand_dims(pi_i_0k_full, axis = 1)*tf.ones(tf.shape(prob_ijn0k)), prob_ijn0k)

    Z_ij = tfp.distributions.Categorical(probs = prob_ijn0k).sample()

    return Z_ij

# Step 1
def unobserved_step_1(one_n_j, K_current, unobserved_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i):

	Z_ij = unobserved_step_1_old_mixtures(one_n_j, unobserved_X, pi_i_0k_full, theta_k_full)

	indexes_new_assignment = tf.where(Z_ij==0)

	if tf.reduce_any(Z_ij==0):

		beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, 
											beta_k_00_full, pi_i_0k_full, Z_ij)

		K_current = K_current+1
		states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

		counter = 0
		while tf.reduce_any(Z_ij==0):

			counter = counter + 1
			if counter>1000:
				raise ValueError("Something wrong")

			indexes_new_assignment = tf.where(Z_ij==0)
			index_rows    = indexes_new_assignment[:,0]

			probability_assign_same_mixture = tf.gather(tf.gather(pi_i_0k_full, index_rows, axis =0), states, axis =1)
			probability_assign_same_mixture = probability_assign_same_mixture/tf.reduce_sum(probability_assign_same_mixture, axis = 1, keepdims = True)

			assignment = tfp.distributions.Categorical(probs = probability_assign_same_mixture).sample()
			state_assignment = tf.gather(states, assignment)

			Z_ij = tf.tensor_scatter_nd_update(Z_ij, indexes_new_assignment, state_assignment)

			indexes_new_assignment = tf.where(Z_ij==0)

			if tf.reduce_any(Z_ij==0):

				beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, 
													beta_k_00_full, pi_i_0k_full, Z_ij)

				K_current = K_current+1
				states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

	return beta_k_00_full, pi_i_0k_full, Z_ij-1, K_current

def SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size):   

	sub_alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(mc_sample_size)

	sub_dirichlet_concentration = tf.expand_dims(sub_alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0)

	sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample()

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
def SZ_m_ik_cheap(one_hot_disjoint_constraints, theta_k_full, disjoint_constraints, J, alpha_0, beta_k_00_full, sub_pi_0k, n_SZ):
	theta_c = tf.einsum("cjn,kjn->ckj", one_hot_disjoint_constraints, theta_k_full)
	theta_c_with_ones = tf.where(tf.expand_dims(disjoint_constraints, axis = 1)==-1, tf.ones(tf.shape(theta_c)), theta_c)

	grid_j = tf.cast(tf.linspace(1, J, J), tf.float32)
	dir_process_prob = tf.expand_dims(alpha_0*beta_k_00_full[1:], axis = -1)/(tf.expand_dims(alpha_0*beta_k_00_full[1:], axis = -1) + tf.expand_dims(grid_j, axis = 0) - 1)
	dir_process_prob = tf.where(tf.expand_dims(grid_j, axis = 0)==1, tf.ones(tf.shape(dir_process_prob)), dir_process_prob)
	cumsum_dir_process_prob = tf.math.cumsum(dir_process_prob, axis = -1)

	bernoulli_trial = tf.einsum("mk,ckj->mckj", sub_pi_0k[:,1:], theta_c_with_ones)

	sample_n_idotk_curr_sample = tf.reduce_sum(tfp.distributions.Bernoulli(probs = bernoulli_trial).sample(), axis = -1)
	sample_sum_to_n_idotk = tf.einsum("kj,mckj->mck", cumsum_dir_process_prob, tf.one_hot(sample_n_idotk_curr_sample-1, J))

	unobserved_m_ik_cheap = tf.math.round(tf.reduce_sum(tf.expand_dims(n_SZ[1:], axis = -1)*tf.reduce_mean(sample_sum_to_n_idotk, axis = 0), axis = 0))

	return unobserved_m_ik_cheap

def SZ_step_789(disjoint_constraints, one_n_j, alpha_0, a, b, K_current, beta_k_00_full, theta_k_full, n):

	nJ = tf.shape(one_n_j)[1]
	one_hot_disjoint_constraints = tf.one_hot(disjoint_constraints, nJ)
	one_hot_disjoint_constraints_with_placeholders = tf.where(tf.expand_dims(disjoint_constraints, axis = -1)==-1, tf.ones(tf.shape(one_hot_disjoint_constraints)), one_hot_disjoint_constraints)

	n_mc_samples = 500
	n_max = 10
	J = tf.shape(one_n_j)[0]
 
	p_c, sub_pi_0k = SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, n_mc_samples)

	counter = 1
	while tf.reduce_sum(p_c)>1 and counter<100:
		p_c, sub_pi_0k = SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, n_mc_samples)

	if tf.reduce_sum(p_c)>1:
		raise ValueError("Something wrong with pc")

	n_0 = tf.cast(n, tf.float32)*tf.reduce_sum(p_c)/(1-tf.reduce_sum(p_c))

	if n_0>500*tf.cast(n, tf.float32):
		n_SZ_no_zero = tf.math.round(n_0*p_c)
		n_SZ = tf.concat((tf.expand_dims(tf.cast(n, tf.float32), axis = 0), n_SZ_no_zero), axis = 0)

	else:
		n_SZ = NegativeMultinomial(n, p_c, n_max)

	unobserved_XZ_counting_cheap = SZ_XZ_counting_cheap(one_hot_disjoint_constraints_with_placeholders, n_SZ, theta_k_full, sub_pi_0k)

	unobserved_m_ik_cheap = SZ_m_ik_cheap(one_hot_disjoint_constraints, theta_k_full, disjoint_constraints, J, alpha_0, beta_k_00_full, sub_pi_0k, n_SZ)

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
def SZ_step_1_old_mixtures(one_n_j, one_hot_X, pi_i_0k_full, theta_k_full):

    theta_jxk = tf.einsum("ijn,kjn->ijk", one_hot_X, theta_k_full)
    
    prob_ijnk = tf.einsum("ik,ijk->ijk", pi_i_0k_full[:,1:], theta_jxk)
    prob_ijn0 = pi_i_0k_full[:,0:1]/tf.expand_dims(tf.reduce_sum(one_n_j, axis = 1), axis = 0)

    prob_ijn0k = tf.concat((tf.expand_dims(prob_ijn0, axis = -1), prob_ijnk), axis = -1)
    prob_ijn0k = prob_ijn0k/tf.reduce_sum(prob_ijn0k, axis = -1, keepdims = True)

    Z_ij = tfp.distributions.Categorical(probs = prob_ijn0k).sample()

    return Z_ij

# Step 1 on old mixtures
@tf.function(jit_compile = True)
def SZ_step_1_new_mixture_assignment(index_new_assignment_current, K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij):

    new_K_current = K_current + 1

    nu_0 = tfp.distributions.Beta(concentration1=alpha_0, concentration0=1.).sample()
    nu_i = tfp.distributions.Beta(concentration1=alpha_i*beta_k_00_full[0]*nu_0, concentration0=alpha_i*beta_k_00_full[0]*(1-nu_0)).sample()

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
def SZ_step_1(one_n_j, K_current, one_hot_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i):

    Z_ij = SZ_step_1_old_mixtures(one_n_j, one_hot_X, pi_i_0k_full, theta_k_full)

    indexes_new_assignment = tf.where(Z_ij==0)

    if tf.reduce_any(Z_ij==0):

        beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij)

        K_current = K_current+1
        states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

        counter = 0
        while tf.reduce_any(Z_ij==0):

            counter = counter + 1
            if counter>1000:
                raise ValueError("Something wrong")

            indexes_new_assignment = tf.where(Z_ij==0)
            index_rows    = indexes_new_assignment[:,0]

            probability_assign_same_mixture = tf.gather(tf.gather(pi_i_0k_full, index_rows, axis =0), states, axis =1)
            probability_assign_same_mixture = probability_assign_same_mixture/tf.reduce_sum(probability_assign_same_mixture, axis = 1, keepdims = True)

            assignment = tfp.distributions.Categorical(probs = probability_assign_same_mixture).sample()
            state_assignment = tf.gather(states, assignment)

            Z_ij = tf.tensor_scatter_nd_update(Z_ij, indexes_new_assignment, state_assignment)

            indexes_new_assignment = tf.where(Z_ij==0)

            if tf.reduce_any(Z_ij==0):

                beta_k_00_full, pi_i_0k_full, Z_ij = SZ_step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij)

                K_current = K_current+1
                states = tf.concat((tf.zeros(1, dtype = tf.int32), tf.expand_dims(K_current, axis = 0)), axis = 0)

    return beta_k_00_full, pi_i_0k_full, Z_ij-1, K_current

# Step 2
def SZ_step_2( n_i_dot_k, alpha_0, beta_k_00_full, J):

	# divide in batches to avoid OOM (structural zeros)
	n_batch    = tf.cast(tf.shape(n_i_dot_k)[0]/10000, dtype = tf.int32)+1
	batch_size = tf.cast(tf.shape(n_i_dot_k)[0]/n_batch, dtype = tf.int32)

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
		table_or_not_i = tfp.distributions.Categorical(probs=prob_new_table).sample()
		m_ik_i = tf.reduce_sum(table_or_not_i, axis = -1)

		m_list.append(m_ik_i)
        
	return tf.concat((m_list), axis = 0)

# Step 3
@tf.function(jit_compile=True)
def SZ_step_3(m_ik, unobserved_m_dotk, alpha_0):

    concentration_beta_1k = tf.cast(tf.reduce_sum(m_ik, axis = 0), dtype = tf.float32) + tf.cast(unobserved_m_dotk, dtype = tf.float32)
    concentration_beta_0k = tf.concat((tf.expand_dims(alpha_0, axis = 0), concentration_beta_1k), axis = 0)

    beta_k_00_full = tfp.distributions.Dirichlet(concentration = concentration_beta_0k).sample()

    return beta_k_00_full

# Step 4
@tf.function(jit_compile=True)
def SZ_step_4(n_i_dot_k, alpha_i, beta_k_00_full):

    n_i_dot_0k = tf.concat((tf.zeros((tf.shape(n_i_dot_k)[0], 1)), n_i_dot_k), axis = -1)
    concentration_pi_i = tf.expand_dims(alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0) + n_i_dot_0k

    pi_ik = tfp.distributions.Dirichlet(concentration = concentration_pi_i).sample()

    return pi_ik

# Step 5
@tf.function(jit_compile=True)
def SZ_step_5(one_hot_X, one_n_j, one_hot_Z, unobserved_XZ_counting):

    dirichlet_lambda = tf.expand_dims(one_n_j, axis = 0) + tf.einsum("ijk,ijn->kjn", one_hot_Z, one_hot_X) + unobserved_XZ_counting

    theta_j_k = tfp.distributions.Dirichlet(concentration = dirichlet_lambda).sample()

    return theta_j_k

# Step 6
def SZ_step_6_multiple(a_0, b_0, a, b, K_current, J, m_ik, unobserved_m_dotk, alpha_0, alpha_i):

    m_dot_dot = tf.reduce_sum(tf.cast(m_ik, dtype = tf.float32) + tf.cast(unobserved_m_dotk, dtype = tf.float32))
    eta_0     = tfp.distributions.Beta(concentration1 = alpha_0 + 1, concentration0 = m_dot_dot).sample()
    bern_p_0  = m_dot_dot*(b_0 - tf.math.log(eta_0))/(tf.cast(K_current, dtype = tf.float32) + a_0 - 1 + m_dot_dot*(b_0 - tf.math.log(eta_0))) 
    s_0       = tf.cast(tfp.distributions.Bernoulli(probs = bern_p_0).sample(), dtype = tf.float32)
    alpha_0  = tfp.distributions.Gamma( concentration = a_0 + tf.cast(K_current, dtype = tf.float32) - s_0, rate = b_0 - tf.math.log(eta_0)).sample()

    m_i_dot   = tf.cast(tf.reduce_sum(m_ik, axis = -1), dtype = tf.float32)
    eta_i     = tfp.distributions.Beta( concentration1 = alpha_i+1, concentration0 = J ).sample()
    bern_p_i  = J*(b - tf.math.log(eta_i))/(m_i_dot + a -1 + J*(b - tf.math.log(eta_i))) 
    s_i       = tf.cast(tfp.distributions.Bernoulli( probs = bern_p_i ).sample(), dtype = tf.float32)
    alpha_i   = tfp.distributions.Gamma(concentration = a + m_i_dot - s_i, rate = b - tf.math.log(eta_i)).sample()

    return alpha_0, alpha_i

# Estimate tau
@tf.function(jit_compile = True)
def noSZ_prob_estimator(X_ij, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size):

    freq_0 = fast_frequency(X_ij, one_n_j)

    subX_ij       = tf.gather(X_ij,       tf.where(freq_0==1)[:,0], axis =0)       

    sub_alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(mc_sample_size)

    sub_dirichlet_concentration = tf.expand_dims(sub_alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0)

    sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample()

    one_hot_X = tf.one_hot(subX_ij, tf.shape(one_n_j)[1])
    sub_theta = tf.einsum("kjn,ijn->ikj", theta_k_full, one_hot_X)

    pi_theta = tf.einsum("mk,ikj->mikj", sub_pi_0k[...,1:], sub_theta)
    pi_n     = tf.expand_dims(sub_pi_0k[...,0:1], axis = -1)*(1/tf.expand_dims(tf.expand_dims((tf.reduce_sum(one_n_j, axis = 1)), axis = 0), axis = 0))

    mc_sample_prob = tf.reduce_prod(tf.reduce_sum(pi_theta, axis = 2) + pi_n, axis = -1) 

    return mc_sample_prob

def SZ_step_mc_tau(MCMC_output, X_ij, disjoint_constraints, one_n_j, N, mc_sample_size):

    a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = MCMC_output

    n = tf.shape(X_ij)[0]

    mc_sample_prob_list = [noSZ_prob_estimator(X_ij, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size) for i in range(5)]
    mc_sample_prob   = tf.concat(mc_sample_prob_list, axis = 0)

    prob_approx = tf.reduce_mean(mc_sample_prob, axis = 0)  

    p_c, _ = SZ_prob_estimator(disjoint_constraints, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size)

    log_prob = tf.math.log(1-prob_approx)
    exponent = tf.cast((N-n), dtype = tf.float32)/(1 - tf.reduce_sum(p_c))
    M = tf.reduce_max(log_prob, axis = 0, keepdims = True)

    mc_tau = tf.math.exp(M[0]*exponent + tf.math.log(tf.reduce_sum(tf.math.exp(exponent*(log_prob - M)))))

    return mc_tau

def SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, initialization_MCMC, unobserved_m_dotk, unobserved_XZ_counting, type_1, type_2):
     
	n = tf.shape(one_hot_X)[0]

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

	a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = initialization_MCMC
		
	J = tf.cast(tf.shape(one_hot_X)[1], dtype = tf.float32)

	beta_k_00_full, pi_i_0k_full, Z_ij, K_current = SZ_step_1(one_n_j, K_current, one_hot_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i)

	one_hot_Z = tf.one_hot(Z_ij, K_max)
	n_i_dot_k = tf.reduce_sum(one_hot_Z, axis = 1)

	m_ik = SZ_step_2(n_i_dot_k, alpha_0, beta_k_00_full, J)

	one_hot_Z, beta_k_00_full, pi_i_0k_full, K_current, unobserved_m_dotk, unobserved_XZ_counting = SZ_mixture_cutter(n_i_dot_k, one_hot_Z, beta_k_00_full, pi_i_0k_full, unobserved_m_dotk, unobserved_XZ_counting)
	n_i_dot_k = tf.reduce_sum(one_hot_Z, axis = 1)

	beta_k_00_full = SZ_step_3(m_ik, unobserved_m_dotk, alpha_0)

	pi_i_0k_full = SZ_step_4(n_i_dot_k[:n,:], alpha_i, beta_k_00_full)

	theta_k_full = SZ_step_5(one_hot_X, one_n_j, one_hot_Z, unobserved_XZ_counting)

	if type_1 == "multiple":
		alpha_0, alpha_i = SZ_step_6_multiple(a_0, b_0, a, b, K_current, J, m_ik, unobserved_m_dotk, alpha_0, alpha_i)
                  
	if type_2 == "fixed a,b":
		a, b = a_1, b_1
        
	return a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current

def SZ_BNP_MCMC_from_start(X_ij, disjoint_constraints, one_n_j, K_current, K_max, prior_parameters, MCMC_iterations, type_1, type_2):

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

	n = tf.shape(X_ij)[0]
	one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

	# initialization 
	a, b = a_1, b_1
	alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = BNP_MCMC_initialization(a_0, b_0, a, b, one_n_j, n, K_max, K_current)

	def body(input, t):

		(a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current) = input

		mu  = tf.reduce_mean(alpha_i)
		mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
		hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
		hat_b = mu/(mu2 - tf.math.pow(mu, 2))
                  
		unobserved_m_dotk, unobserved_XZ_counting, beta_k_00_full, K_current = SZ_step_789(disjoint_constraints, one_n_j, alpha_0, hat_a, hat_b, K_current, beta_k_00_full, theta_k_full, n)

		input = (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current)

		output_t = SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, unobserved_m_dotk, unobserved_XZ_counting, type_1, type_2)

		return output_t
    
	output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current))

	return output

def SZ_BNP_MCMC_initialized(X_ij, disjoint_constraints, one_n_j, K_max, prior_parameters, initialization_MCMC, MCMC_iterations, type_1, type_2):

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

	n = tf.shape(X_ij)[0]
	one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

	def body(input, t):

		(a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current) = input

		mu  = tf.reduce_mean(alpha_i)
		mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
		hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
		hat_b = mu/(mu2 - tf.math.pow(mu, 2))
                  
		unobserved_m_dotk, unobserved_XZ_counting, beta_k_00_full, K_current = SZ_step_789(disjoint_constraints, one_n_j, alpha_0, hat_a, hat_b, K_current, beta_k_00_full, theta_k_full, n)

		input = (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current)

		output_t = SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, unobserved_m_dotk, unobserved_XZ_counting, type_1, type_2)

		return output_t
    
	output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  initialization_MCMC)

	return output

def SZ_BNP_MCMC_initialized_tau(X_ij, disjoint_constraints, one_n_j, K_max, prior_parameters, 
				initialization_MCMC, MCMC_iterations, N, batch_size, type_1, type_2, type_tau):

	a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters
	
	n = tf.shape(X_ij)[0]
	J = tf.shape(one_n_j)[0]
	one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

	tau = 0.

	def body(input, t):

		feed_MCMC, _ = input

		(a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current) = feed_MCMC

		mu  = tf.reduce_mean(alpha_i)
		mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
		hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
		hat_b = mu/(mu2 - tf.math.pow(mu, 2))
                  
		unobserved_m_dotk, unobserved_XZ_counting, beta_k_00_full, K_current = SZ_step_789(disjoint_constraints, one_n_j, alpha_0, hat_a, hat_b, K_current, beta_k_00_full, theta_k_full, n)

		input = (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current)

		output_t = SZ_BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, unobserved_m_dotk, unobserved_XZ_counting, type_1, type_2)

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
				tau = SZ_step_mc_tau(output_t_tau, X_ij, disjoint_constraints, one_n_j, N, batch_size)

		return output_t, tau
    
	MCMC_output, Tau = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (initialization_MCMC, tau))

	return MCMC_output, Tau
