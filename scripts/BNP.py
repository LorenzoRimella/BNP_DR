import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from mixed_membership_model import *

# Initializer
# initialize the parameters using a hierarchical Dirichlet process with the stick breaking construction
def initialize_param_single_step(a_0, b_0, a, b, one_n_j, n_indiv, K_max, K_initial, seed_initialization_single_step):
    
    seed_sample_initialize = tfp.random.split_seed( seed_initialization_single_step, n=5, salt='seed_initialization_single_step')

    alpha_0 = tfp.distributions.Gamma( concentration = a_0, rate = b_0).sample(seed = seed_sample_initialize[0])

    theta_k_full  = tfp.distributions.Dirichlet(concentration = one_n_j).sample(K_max, seed = seed_sample_initialize[1])

    V_k_0      = tfp.distributions.Beta(concentration1 = 1, concentration0 = alpha_0).sample(K_initial, seed = seed_sample_initialize[2])
    log_cumprod_0 = tf.math.cumsum(tf.math.log(1-V_k_0)) - tf.math.log(1-V_k_0)

    beta_k_0 = tf.math.exp(tf.math.log(V_k_0) + log_cumprod_0)
    beta_k_00 = tf.concat((1-tf.reduce_sum(beta_k_0, axis = 0, keepdims= True), beta_k_0), axis =0)
    beta_k_00 = tf.math.abs(beta_k_00)/tf.reduce_sum(tf.math.abs(beta_k_00), axis = -1, keepdims= True)

    cumsum_pi_k_0 = tf.math.cumsum(beta_k_0)
    cumsum_pi_k_0 = tf.reduce_min(tf.stack((cumsum_pi_k_0, tf.ones(tf.shape(cumsum_pi_k_0))), axis = 1), axis = -1)

    alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(n_indiv, seed = seed_sample_initialize[3])
    concentration1_i_k = tf.expand_dims(alpha_i, axis = -1)*tf.expand_dims(beta_k_0, axis = 0)
    concentration0_i_k = tf.expand_dims(alpha_i, axis = -1)*(1 - tf.expand_dims(cumsum_pi_k_0, axis = 0))

    V_i_k = tfp.distributions.Beta(concentration1 = concentration1_i_k, concentration0 = concentration0_i_k+1e-35).sample(seed = seed_sample_initialize[4], )
    log_cumprod_i = tf.math.cumsum(tf.math.log(1-V_i_k+1e-35), axis = 1) - tf.math.log(1-V_i_k+1e-30)

    pi_i_k = tf.math.exp(tf.math.log(V_i_k+1e-35) + log_cumprod_i)
    pi_i_k = tf.math.abs(pi_i_k)/tf.reduce_sum(tf.math.abs(pi_i_k), axis = -1, keepdims= True)
    pi_i_0k = tf.concat((1-tf.reduce_sum(pi_i_k, axis =1, keepdims=True), pi_i_k), axis = 1)
    pi_i_0k = tf.math.abs(pi_i_0k)/tf.reduce_sum(tf.math.abs(pi_i_0k), axis = -1, keepdims= True)

    beta_k_00_full = tf.concat((beta_k_00, tf.zeros((tf.shape(beta_k_0)))), axis =-1)
    pi_i_0k_full = tf.concat((pi_i_0k, tf.zeros((tf.shape(pi_i_k)))), axis =-1)

    return alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full

def initialize_param(a_0, b_0, a, b, one_n_j, n_indiv, K_max, K_initial, seed_initialization):

    seed_initialization_single_step, seed_initialization_carry = tfp.random.split_seed( seed_initialization, n=2, salt='seed_initialization')

    alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full = initialize_param_single_step(a_0, b_0, a, b, one_n_j, n_indiv, K_max, K_initial, seed_initialization_single_step)

    counter = 0 
    while tf.reduce_any(tf.math.is_nan(alpha_0)) or tf.reduce_any(tf.math.is_nan(alpha_i)) or tf.reduce_any(tf.math.is_nan(theta_k_full)) or tf.reduce_any(tf.math.is_nan(beta_k_00_full)) or tf.reduce_any(tf.math.is_nan(pi_i_0k_full)):

        seed_initialization_single_step, seed_initialization_carry = tfp.random.split_seed( seed_initialization_carry, n=2, salt='seed_initialization_next'+str(counter))

        alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full = initialize_param_single_step(a_0, b_0, a, b, one_n_j, n_indiv, K_max, K_initial, seed_initialization_single_step)

        counter = counter + 1

    return alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full

# Mixture cutter
# cut the unused mixtures and keep some extra for safety
# the cutting is based on pi_i_0k_full which will not change
def initilizer_mixture_cutter(pi_i_0k_full, beta_k_00_full, theta_k_full):

    new_K_current = tf.cast(tf.where(tf.reduce_sum(pi_i_0k_full, axis = 0)!=0)[-1] + 10, dtype = tf.int32)[0]

    mass_to_move = tf.reduce_sum(beta_k_00_full[new_K_current:])
    mask_cutter  = tf.concat((tf.ones(tf.shape(beta_k_00_full[:new_K_current])), tf.zeros(tf.shape(beta_k_00_full[new_K_current:]))), axis = 0)
    new_beta_k_00_full = beta_k_00_full*mask_cutter
    new_beta_k_00_full = tf.squeeze(tf.tensor_scatter_nd_update(tf.expand_dims(new_beta_k_00_full, axis = 1), [[0]], [[mass_to_move]]))

    new_theta_k_full = tf.expand_dims(tf.expand_dims(mask_cutter[1:], axis = 1), axis = 1)*theta_k_full

    return new_K_current-1, new_beta_k_00_full, new_theta_k_full

# Mixture cutter
def mixture_cutter(n_i_dot_k, one_hot_Z, beta_k_00_full, pi_i_0k_full):

    K_max = tf.shape(one_hot_Z)[-1]

    non_zero_index = tf.cast(tf.where(tf.reduce_sum(n_i_dot_k, axis =0)!=0)[:,0], dtype = tf.int32)

    new_one_hot_Z = tf.gather(one_hot_Z, non_zero_index, axis = -1)
    new_n_i_dot_k = tf.gather(n_i_dot_k, non_zero_index, axis = -1)

    non_zero_index_plus_0 = tf.concat((tf.zeros(1, dtype = tf.int32), 1+non_zero_index), axis = 0)

    new_beta_k_00_full = tf.gather(beta_k_00_full, non_zero_index_plus_0, axis = -1)
    new_pi_i_0k_full   = tf.gather(pi_i_0k_full,   non_zero_index_plus_0, axis = -1)

    new_K_current = tf.shape(new_n_i_dot_k)[1]

    new_one_hot_Z_zeros      = tf.zeros((tf.shape(new_one_hot_Z)[0], tf.shape(new_one_hot_Z)[1], K_max- tf.shape(new_one_hot_Z)[2]))
    new_beta_k_00_full_zeros = tf.zeros((K_max- tf.shape(new_one_hot_Z)[2]))
    new_pi_i_0k_full_zeros   = tf.zeros((tf.shape(new_one_hot_Z)[0], K_max- tf.shape(new_one_hot_Z)[2]))

    return tf.concat((new_one_hot_Z, new_one_hot_Z_zeros), axis = -1), tf.concat((new_beta_k_00_full, new_beta_k_00_full_zeros), axis = -1), tf.concat((new_pi_i_0k_full, new_pi_i_0k_full_zeros), axis = -1), tf.cast(new_K_current+1, dtype = tf.int32)

# Step 1 on old mixtures
@tf.function(jit_compile = True)
def step_1_old_mixtures(one_hot_X, pi_i_0k_full, theta_k_full, seed_s1_1):
    
    J = tf.cast(tf.shape(one_hot_X)[1], dtype = tf.float32)

    theta_jxk = tf.einsum("ijn,kjn->ijk", one_hot_X, theta_k_full)
    
    prob_ijnk = tf.einsum("ik,ijk->ijk", pi_i_0k_full[:,1:], theta_jxk)
    prob_ijn0 = (pi_i_0k_full[:,0:1]/J)*tf.ones(tf.shape(prob_ijnk)[:-1])

    prob_ijn0k = tf.concat((tf.expand_dims(prob_ijn0, axis = -1), prob_ijnk), axis = -1)
    prob_ijn0k = prob_ijn0k/tf.reduce_sum(prob_ijn0k, axis = -1, keepdims = True)

    Z_ij = tfp.distributions.Categorical(probs = prob_ijn0k).sample(seed = seed_s1_1)

    return Z_ij

# Step 1 on old mixtures
@tf.function(jit_compile = True)
def step_1_new_mixture_assignment(index_new_assignment_current, K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij, seed_new_mixture):

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
    one_hot_Kp1 = tf.one_hot(new_K_current+1, tf.shape(pi_i_0k_full)[1])
    new_pi_i_0k_0   = tf.expand_dims(pi_i_0k_full[:,0]*nu_i,     axis = 1)*one_hot_0
    new_pi_i_0k_Kp1 = tf.expand_dims(pi_i_0k_full[:,0]*(1-nu_i), axis = 1)*one_hot_Kp1

    new_pi_i_0k_full = (pi_i_0k_full*(1- (one_hot_0 + one_hot_Kp1))) + new_pi_i_0k_0 + new_pi_i_0k_Kp1

    new_Z_ij = tf.tensor_scatter_nd_update(Z_ij, tf.expand_dims(index_new_assignment_current, axis = 0), (new_K_current)*tf.ones((1), dtype = tf.int32))

    return new_beta_k_00_full, new_pi_i_0k_full, new_Z_ij

# Step 1
def step_1(K_current, one_hot_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i, seed_s1):

    seed_s1_1, seed_s1_2 = tfp.random.split_seed( seed_s1, n=2, salt='step_1_split')

    Z_ij = step_1_old_mixtures(one_hot_X, pi_i_0k_full, theta_k_full, seed_s1_1)

    indexes_new_assignment = tf.where(Z_ij==0)

    if tf.reduce_any(Z_ij==0):

        states = tf.zeros(1, dtype = tf.int32)

        seed_s1_2_to_use, seed_s1_2_to_carry = tfp.random.split_seed( seed_s1_2, n=2, salt='seed_s1_2_to_carry_split')

        beta_k_00_full, pi_i_0k_full, Z_ij = step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij, seed_s1_2_to_use)

        K_current = K_current+1
        states = tf.concat((states, tf.expand_dims(K_current, axis = 0)), axis = 0)

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

                beta_k_00_full, pi_i_0k_full, Z_ij = step_1_new_mixture_assignment(indexes_new_assignment[0], K_current, alpha_0, alpha_i, beta_k_00_full, pi_i_0k_full, Z_ij, seed_s1_2_to_use_second)

                K_current = K_current+1
                states = tf.concat((states, tf.expand_dims(K_current, axis = 0)), axis = 0)

    return beta_k_00_full, pi_i_0k_full, Z_ij-1, K_current

# Step 2
@tf.function(jit_compile = True)
def step_2( n_i_dot_k, alpha_0, beta_k_00_full, J, seed_s2):

    # do the sampling only on custumers greater than 1

    concentration_chinese_restaurant = tf.expand_dims(alpha_0*beta_k_00_full[1:], axis = 0)*tf.ones(tf.shape(n_i_dot_k))

    one_hot_n_i_dot_k = tf.one_hot(tf.cast(n_i_dot_k, dtype = tf.int32), tf.cast(J+1, dtype = tf.int32))

    sequential_custumers_keep_0 = tf.math.cumsum(one_hot_n_i_dot_k[..., ::-1], axis = -1)[...,::-1]
    sequential_custumers = tf.math.cumsum(sequential_custumers_keep_0, axis =-1)-1
    sequential_custumers = sequential_custumers*sequential_custumers_keep_0
    sequential_custumers_no_zeros = sequential_custumers[...,1:]
    sequential_custumers_keep_no_zeros = sequential_custumers_keep_0[...,1:]

    prob_new_table = sequential_custumers_keep_no_zeros*(tf.expand_dims(concentration_chinese_restaurant, axis = -1)/(sequential_custumers_no_zeros - 1 + tf.expand_dims(concentration_chinese_restaurant, axis = -1)))
    prob_new_table = tf.stack((1- prob_new_table, prob_new_table), axis = -1)
    table_or_not = tfp.distributions.Categorical(probs=prob_new_table).sample(seed = seed_s2)

    m_ik = tf.reduce_sum(table_or_not, axis = -1)

    return m_ik

# Step 3
@tf.function(jit_compile = True)
def step_3(m_ik, alpha_0, seed_s3):

    concentration_beta_1k = tf.cast(tf.reduce_sum(m_ik, axis = 0), dtype = tf.float32)
    concentration_beta_0k = tf.concat((tf.expand_dims(alpha_0, axis = 0), concentration_beta_1k), axis = 0)

    beta_k_00_full = tfp.distributions.Dirichlet(concentration = concentration_beta_0k).sample(seed = seed_s3)

    return beta_k_00_full

# Step 4
@tf.function(jit_compile = True)
def step_4(n_i_dot_k, alpha_i, beta_k_00_full, seed_s4):

    n_i_dot_0k = tf.concat((tf.zeros((tf.shape(n_i_dot_k)[0], 1)), n_i_dot_k), axis = -1)
    concentration_pi_i = tf.expand_dims(alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0) + n_i_dot_0k

    pi_ik = tfp.distributions.Dirichlet(concentration = concentration_pi_i).sample(seed = seed_s4)

    return pi_ik

# Step 5
@tf.function(jit_compile = True)
def step_5(one_hot_X, one_n_j, one_hot_Z, seed_s5):

    dirichlet_lambda = tf.expand_dims(one_n_j, axis = 0) + tf.einsum("ijk,ijn->kjn", one_hot_Z, one_hot_X)

    theta_j_k = tfp.distributions.Dirichlet(concentration = dirichlet_lambda).sample(seed = seed_s5)

    return theta_j_k

# Step 6
@tf.function(jit_compile = True)
def step_6_multiple(a_0, b_0, a, b, K_current, J, m_ik, alpha_0, alpha_i, seed_s6):

    seed_s6 = tfp.random.split_seed( seed_s6, n=6, salt='step_6_seed_split')

    m_dot_dot = tf.cast(tf.reduce_sum(m_ik), dtype = tf.float32)
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

# Step 6
@tf.function(jit_compile = True)
def step_6_single(a_0, b_0, a, b, K_current, J, m_ik, alpha_0, alpha_i, seed_s6):

    seed_s6 = tfp.random.split_seed( seed_s6, n=6, salt='step_6_seed_split')

    m_dot_dot = tf.cast(tf.reduce_sum(m_ik), dtype = tf.float32)
    eta_0     = tfp.distributions.Beta(concentration1 = alpha_0 + 1, concentration0 = m_dot_dot).sample(seed = seed_s6[0])
    bern_p_0  = m_dot_dot*(b_0 - tf.math.log(eta_0))/(tf.cast(K_current, dtype = tf.float32) + a_0 - 1 + m_dot_dot*(b_0 - tf.math.log(eta_0))) 
    s_0       = tf.cast(tfp.distributions.Bernoulli(probs = bern_p_0).sample(seed = seed_s6[1]), dtype = tf.float32)
    alpha_0  = tfp.distributions.Gamma( concentration = a_0 + tf.cast(K_current, dtype = tf.float32) - s_0, rate = b_0 - tf.math.log(eta_0)).sample(seed = seed_s6[2])

    n = tf.cast(tf.shape(alpha_i)[0], dtype = tf.float32)
    eta_i     = tfp.distributions.Beta( concentration1 = alpha_i, concentration0 = J ).sample(seed = seed_s6[3])
    bern_p    = J/(tf.reduce_mean(alpha_i) + J) 
    s         = tf.cast(tfp.distributions.Binomial( total_count = n, probs = bern_p ).sample(seed = seed_s6[4]), dtype = tf.float32)
    alpha_i   = tfp.distributions.Gamma(concentration = a + m_dot_dot - s, rate = b - tf.reduce_sum(tf.math.log(eta_i))).sample(seed = seed_s6[5])*tf.ones(tf.cast(n, dtype = tf.int32), dtype = tf.float32)

    return alpha_0, alpha_i

# Step 6
@tf.function(jit_compile = True)
def step_7(alpha_i, a_prev, b_prev, a_1, b_1, a_2, b_2, sigma, seed_7 ):

    seed_s7 = tfp.random.split_seed( seed_7, n=3, salt='step_7_seed_split')

    n = tf.cast(tf.shape(alpha_i)[0], dtype = tf.float32)

    b_new  = tfp.distributions.Gamma(concentration = a_1 + n*a_prev, rate = b_1 + tf.reduce_sum(alpha_i)).sample(seed = seed_s7[0])

    # a_proposed = tfp.distributions.Gamma(concentration = a_2, rate= b_2).sample(seed = seed_s7[1])
    a_proposed = tf.math.exp(tfp.distributions.Normal(loc = tf.math.log(a_prev), scale = sigma).sample(seed=seed_s7[1]))

    log_like_diff = tf.reduce_sum(tfp.distributions.Gamma(concentration = a_proposed, rate = b_new).log_prob(alpha_i)) - tf.reduce_sum(tfp.distributions.Gamma(concentration = a_prev, rate = b_new).log_prob(alpha_i))
    prior_diff    = tf.reduce_sum(tfp.distributions.Gamma(concentration = a_2, rate = b_2).log_prob(a_proposed)) - tf.reduce_sum(tfp.distributions.Gamma(concentration = a_2, rate = b_2).log_prob(a_prev))
    logp          = log_like_diff + prior_diff

    boolean = tf.cast(tfp.distributions.Uniform(low=0.0, high=1.0).sample(seed = seed_s7[2])< tf.math.exp(logp), dtype = tf.float32)
    a_new   = a_proposed*boolean + a_prev*(1-boolean)
    
    return a_new, b_new

# MCMC
def BNP_MCMC_initialization(a_0, b_0, a, b, one_n_j, n, K_max, K_current, seed_init):

    alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full = initialize_param(a_0, b_0, a, b, one_n_j, n, K_max, K_current, seed_init)

    K_current, beta_k_00_full, theta_k_full = initilizer_mixture_cutter(pi_i_0k_full, beta_k_00_full, theta_k_full)

    return alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current

def BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, initialization_MCMC, seed_step, type_1, type_2):

    a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

    a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = initialization_MCMC
    
    seed_1, seed_2, seed_3, seed_4, seed_5, seed_6, seed_7 = tfp.random.split_seed( seed_step, n=7, salt='seed_step_split')
        
    J = tf.cast(tf.shape(one_hot_X)[1], dtype = tf.float32)

    beta_k_00_full, pi_i_0k_full, Z_ij, K_current = step_1(K_current, one_hot_X, pi_i_0k_full, theta_k_full, beta_k_00_full, alpha_0, alpha_i, seed_1)

    one_hot_Z = tf.one_hot(Z_ij, K_max)
    n_i_dot_k = tf.reduce_sum(one_hot_Z, axis = 1)

    m_ik = step_2(n_i_dot_k, alpha_0, beta_k_00_full, J, seed_2)
    
    one_hot_Z, beta_k_00_full, pi_i_0k_full, K_current = mixture_cutter(n_i_dot_k, one_hot_Z, beta_k_00_full, pi_i_0k_full)
    n_i_dot_k = tf.reduce_sum(one_hot_Z, axis = 1)


    seed_s3, seed_s3_carry  = tfp.random.split_seed( seed_3, n=2, salt='seed_s3_carry_'+str(K_current))
    beta_k_00_full = step_3(m_ik, alpha_0, seed_s3)

    counter = 0
    while tf.reduce_any(tf.math.is_nan(beta_k_00_full)) and counter<100:

        seed_s3, seed_s3_carry  = tfp.random.split_seed( seed_s3_carry, n=2, salt='seed_s3_carry_'+str(K_current))
        beta_k_00_full = step_3(m_ik, alpha_0, seed_s3)

        counter = counter +1

    seed_s4, seed_s4_carry  = tfp.random.split_seed( seed_4, n=2, salt='seed_s4_carry_'+str(K_current))
    pi_i_0k_full = step_4(n_i_dot_k, alpha_i, beta_k_00_full, seed_s4)

    counter = 0
    while tf.reduce_any(tf.math.is_nan(pi_i_0k_full)) and counter<100:

        seed_s4, seed_s4_carry  = tfp.random.split_seed( seed_s4_carry, n=2, salt='seed_s4_carry_'+str(K_current))
        pi_i_0k_full = step_4(n_i_dot_k, alpha_i, beta_k_00_full, seed_s4)

        counter = counter +1

    seed_s5, seed_s5_carry  = tfp.random.split_seed( seed_5, n=2, salt='seed_s5_carry_'+str(K_current))
    theta_k_full = step_5(one_hot_X, one_n_j, one_hot_Z, seed_s5)

    counter = 0
    while tf.reduce_any(tf.math.is_nan(theta_k_full)) and counter<100:

        seed_s5, seed_s5_carry  = tfp.random.split_seed( seed_s5_carry, n=2, salt='seed_s5_carry_'+str(K_current))
        theta_k_full = step_5(one_hot_X, one_n_j, one_hot_Z, seed_s5)

        counter = counter +1

    if type_1 == "single":
        alpha_0, alpha_i = step_6_single(a_0, b_0, a, b, K_current, J, m_ik, alpha_0, alpha_i, seed_6)

    if type_1 == "multiple":
        alpha_0, alpha_i = step_6_multiple(a_0, b_0, a, b, K_current, J, m_ik, alpha_0, alpha_i, seed_6)

    if type_2 == "random a,b":

        if type_1 == "single":
            a, b = step_7(alpha_i[0:1,...], a, b, a_1, b_1, a_2, b_2, sigma, seed_7 )

        else:
            a, b = step_7(alpha_i, a, b, a_1, b_1, a_2, b_2, sigma, seed_7 )
    
    if type_2 == "fixed a,b":
        a, b = a_1, b_1

    return a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current

def BNP_MCMC_from_start(X_ij, one_n_j, K_current, K_max, prior_parameters, MCMC_iterations, seed_MCMC, type_1, type_2):

    a_0, b_0, a_1, b_1, a_2, b_2, sigma = prior_parameters

    seed_initialization, seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=2, salt='seed_MCMC')

    n = tf.shape(X_ij)[0]
    one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    # initialization 
    a, b = a_2/b_2, a_1/b_1
    alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = BNP_MCMC_initialization(a_0, b_0, a, b, one_n_j, n, K_max, K_current, seed_initialization)

    seed_step_to_split  = tfp.random.split_seed( seed_step_to_split, n=MCMC_iterations, salt='seed_MCMC_step_per_iter')

    def body(input, t):

        output_t = BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, seed_step_to_split[t], type_1, type_2)

        return output_t
    
    output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current))

    return output

def BNP_MCMC_initialized(X_ij, one_n_j, K_max, prior_parameters, initialization_MCMC, MCMC_iterations, seed_MCMC, type_1, type_2):

    one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=MCMC_iterations, salt='seed_MCMC_step_per_iter_not_init')

    def body(input, t):

        output_t = BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, input, seed_step_to_split[t], type_1, type_2)

        return output_t
    
    output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  initialization_MCMC)

    return output

# Estimate tau
@tf.function(jit_compile = True)
def BNP_population_sampler(MCMC_output, one_n_j, J, batch_size, seed_pop_sampler):

    a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = MCMC_output
    
    seed_pop_sampler_splitted  = tfp.random.split_seed( seed_pop_sampler, n=4, salt='seed_pop_sampler_split')

    new_alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(batch_size, seed = seed_pop_sampler_splitted[0])

    new_dirichlet_concentration = tf.expand_dims(new_alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0)

    new_pi_0k = tfp.distributions.Dirichlet(concentration=new_dirichlet_concentration).sample(seed = seed_pop_sampler_splitted[1])

    choose_profile_unobserved = tfp.distributions.Categorical(probs = new_pi_0k).sample(J, seed = seed_pop_sampler_splitted[2])
    choose_profile_unobserved = tf.transpose(choose_profile_unobserved)

    one_hot = tf.one_hot(choose_profile_unobserved, tf.shape(new_pi_0k)[1])

    theta_0k_full = tf.concat((tf.expand_dims(one_n_j, axis = 0), theta_k_full), axis = 0)
    individual_profile = tf.einsum("kjn,ijk->ijn", theta_0k_full, one_hot)

    X_ij_unobserved = tfp.distributions.Categorical(probs = individual_profile).sample(seed = seed_pop_sampler_splitted[3])

    return X_ij_unobserved


@tf.function(jit_compile = True)
def BNP_final_population_sampler(MCMC_output, one_n_j, J, batch_size, final_size, seed_pop_sampler):

    a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = MCMC_output
    
    seed_pop_sampler_splitted  = tfp.random.split_seed( seed_pop_sampler, n=4, salt='seed_pop_sampler_split')

    new_alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(batch_size, seed = seed_pop_sampler_splitted[0])

    new_dirichlet_concentration = tf.expand_dims(new_alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0)

    new_pi_0k = tfp.distributions.Dirichlet(concentration=new_dirichlet_concentration).sample(seed = seed_pop_sampler_splitted[1])

    choose_profile_unobserved = tfp.distributions.Categorical(probs = new_pi_0k).sample(J, seed = seed_pop_sampler_splitted[2])
    choose_profile_unobserved = tf.transpose(choose_profile_unobserved)

    one_hot = tf.one_hot(choose_profile_unobserved, tf.shape(new_pi_0k)[1])

    theta_0k_full = tf.concat((tf.expand_dims(one_n_j, axis = 0), theta_k_full), axis = 0)
    individual_profile = tf.einsum("kjn,ijk->ijn", theta_0k_full, one_hot)

    X_ij_unobserved = tfp.distributions.Categorical(probs = individual_profile).sample(seed = seed_pop_sampler_splitted[3])

    return X_ij_unobserved[:final_size,...]

def step_tau(MCMC_output, X_ij, one_n_j, J, batch_size, N, seed_step_tau):
    
    n = tf.shape(X_ij)[0]

    iterations = tf.math.floordiv(N - n, batch_size)
    final_size = N - n - iterations*batch_size

    freq_0 = fast_frequency(X_ij, one_n_j)

    subX_ij    = tf.gather(X_ij,   tf.where(freq_0==1)[:,0], axis =0)
    subfreq_0  = tf.gather(freq_0, tf.where(freq_0==1)[:,0], axis =0)

    seed_step_tau_splitted  = tfp.random.split_seed( seed_step_tau, n=iterations+1, salt='seed_split_for_tau_batches')

    def cond(iteration, input):

        return iteration<iterations

    def body(iteration, input):

        freq_prev = input

        X_ij_unobserved = BNP_population_sampler(MCMC_output, one_n_j, J, batch_size, seed_step_tau_splitted[iteration])

        freq_updated = fast_frequency_batched(subX_ij, X_ij_unobserved, one_n_j)

        freq_join = tf.stack((freq_prev, freq_updated), axis = 1)

        new_freq = tf.reduce_max(freq_join, axis = 1)

        return iteration+1, new_freq

    _, freq_1 = tf.while_loop(cond, body, loop_vars = (0, subfreq_0))

    X_ij_unobserved_final = BNP_final_population_sampler(MCMC_output, one_n_j, J, batch_size, final_size, seed_step_tau_splitted[-1])
    freq_final = fast_frequency_batched_final_size(subX_ij, X_ij_unobserved_final, one_n_j)

    freq_join_final = tf.stack((freq_1, freq_final), axis = 1)

    freq_2 = tf.reduce_max(freq_join_final, axis = 1)

    tau = tf.reduce_sum(tf.cast(freq_2==1, dtype = tf.float32))

    return tau


# def step_mc_tau(X_ij, one_n_j, theta_k_full, a, b, beta_k_00_full, N, mc_sample_size, seed_step_tau):

#     n = tf.shape(X_ij)[0]

#     freq_0 = fast_frequency(X_ij, one_n_j)

#     subX_ij       = tf.gather(X_ij,       tf.where(freq_0==1)[:,0], axis =0)        

#     seed_step_tau_1, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau, n=2, salt='seed_split_for_tau_mc')

#     sub_alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(mc_sample_size, seed = seed_step_tau_1)

#     sub_dirichlet_concentration = tf.expand_dims(sub_alpha_i, axis = 1)*tf.expand_dims(beta_k_00_full, axis = 0)

#     seed_step_tau_2, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau_2_carry, n=2, salt='seed_split_for_tau_mc_carry')
#     sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample(seed = seed_step_tau_2)
#     while tf.reduce_any(tf.math.is_nan(sub_pi_0k)):
#         seed_step_tau_2, seed_step_tau_2_carry = tfp.random.split_seed( seed_step_tau_2_carry, n=2, salt='seed_split_for_tau_mc_carry')
#         sub_pi_0k = tfp.distributions.Dirichlet(concentration=sub_dirichlet_concentration).sample(seed = seed_step_tau_2)

#     one_hot_X = tf.one_hot(subX_ij, tf.shape(one_n_j)[1])
#     sub_theta = tf.einsum("kjn,ijn->ikj", theta_k_full, one_hot_X)

#     pi_theta = tf.einsum("mk,ikj->mikj", sub_pi_0k[...,1:], sub_theta)
#     pi_n     = tf.expand_dims(sub_pi_0k[...,0:1], axis = -1)*(1/tf.expand_dims(tf.expand_dims((tf.reduce_sum(one_n_j, axis = 1)), axis = 0), axis = 0))

#     mc_sample_prob = tf.reduce_prod(tf.reduce_sum(pi_theta, axis = 2) + pi_n, axis = -1) 

#     prob_approx = tf.reduce_mean(mc_sample_prob, axis = 0)  

#     log_prob = tf.math.log(1-prob_approx)
#     exponent = tf.cast(N-n, dtype = tf.float32)
#     # M = tf.reduce_max(log_prob, axis = 0, keepdims = True)

#     # mc_tau = tf.math.exp(M[0]*exponent + tf.math.log(tf.reduce_sum(tf.math.exp(exponent*(log_prob - M)))))
#     mc_tau = tf.reduce_sum(tf.math.exp(exponent*(log_prob)))

#     return mc_tau


@tf.function(jit_compile = True)
def prob_estimator(X_ij, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size, seed_step_tau):

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


def step_mc_tau(MCMC_output, X_ij, one_n_j, N, mc_sample_size, seed_step_tau):

    a, b, alpha_0, alpha_i, theta_k_full, beta_k_00_full, pi_i_0k_full, K_current = MCMC_output

    n = tf.shape(X_ij)[0]

    seed_step_tau_to_split = tfp.random.split_seed( seed_step_tau, n=5, salt='seed_split_for_tau_mc_carry')

    mc_sample_prob_list = [prob_estimator(X_ij, one_n_j, a, b, theta_k_full, beta_k_00_full, mc_sample_size, seed_step_tau_to_split[i]) for i in range(5)]
    mc_sample_prob   = tf.concat(mc_sample_prob_list, axis = 0)

    prob_approx = tf.reduce_mean(mc_sample_prob, axis = 0)  

    log_prob = tf.math.log(1-prob_approx)
    exponent = tf.cast(N-n, dtype = tf.float32)
    M = tf.reduce_max(log_prob, axis = 0, keepdims = True)

    mc_tau = tf.math.exp(M[0]*exponent + tf.math.log(tf.reduce_sum(tf.math.exp(exponent*(log_prob - M)))))

    return mc_tau


def BNP_MCMC_initialized_tau(X_ij, one_n_j, K_max, prior_parameters, initialization_MCMC, N, MCMC_iterations, batch_size, seed_MCMC, type_1, type_2, type_tau):

    J = tf.shape(one_n_j)[0]

    seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=MCMC_iterations, salt='seed_MCMC_step_per_iter_not_init_with_tau')
    tau = 0.

    one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    def body(input, t):

        feed_MCMC, _ = input

        seed_step_without_tau, seed_step_with_tau = tfp.random.split_seed( seed_step_to_split[t], n=2, salt='without_and_with_tau')

        output_t = BNP_MCMC_step(one_hot_X, one_n_j, K_max, prior_parameters, feed_MCMC, seed_step_without_tau, type_1, type_2)

        if type_2 == "fixed a,b":
            alpha_i = output_t[3]
            mu  = tf.reduce_mean(alpha_i)
            mu2 = tf.reduce_mean(tf.math.pow(alpha_i, 2))
            hat_a = tf.math.pow(mu, 2)/(mu2 - tf.math.pow(mu, 2))
            hat_b = mu/(mu2 - tf.math.pow(mu, 2))

            output_t_tau = hat_a, hat_b, output_t[2], output_t[3], output_t[4], output_t[5], output_t[6], output_t[7]

        else:
            output_t_tau = output_t

        if type_tau=="sampling":
            tau = step_tau(output_t_tau, X_ij, one_n_j, J, batch_size, N, seed_step_with_tau)

        if type_tau=="Monte Carlo":
            tau = step_mc_tau(output_t_tau, X_ij, one_n_j, N, batch_size, seed_step_with_tau)

        return output_t, tau
    
    MCMC_output, Tau = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (initialization_MCMC, tau))

    return MCMC_output, Tau
