import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from mixed_membership_model import *

# Step 1
@tf.function(jit_compile = True)
def step_1(one_hot_X, g_i, theta_j_k, seed_s1):

    g_lambda = tf.einsum("ik,kjn->ikjn", g_i, theta_j_k)
    p_ij = tf.einsum("ikjn,ijn->ijk", g_lambda, one_hot_X) # better do theta and onehot first, or use tf.gather
    p_ij = p_ij/tf.reduce_sum(p_ij, axis = -1, keepdims = True)

    Z_ij = tfp.distributions.Categorical(probs = p_ij).sample(seed = seed_s1)

    return Z_ij

# Step 2
@tf.function(jit_compile = True)
def step_2(one_hot_X, one_n_j, one_hot_Z, seed_s2):

    dirichlet_lambda = tf.expand_dims(one_n_j, axis = 0) + tf.einsum("ijk,ijn->kjn", one_hot_Z, one_hot_X)

    theta_j_k = tfp.distributions.Dirichlet(concentration = dirichlet_lambda).sample(seed = seed_s2)

    return theta_j_k

# Step 3
@tf.function(jit_compile = True)
def step_3(one_hot_Z, g_0, seed_s3):

    dirichlet_g_i = tf.expand_dims(g_0, axis = 0 ) + tf.reduce_sum(one_hot_Z, axis = 1)

    g_i = tfp.distributions.Dirichlet(concentration = dirichlet_g_i).sample(seed = seed_s3)

    return g_i

# Step 4
@tf.function(jit_compile = True)
def lratioGamma(x, y):

    return tf.cast(tf.math.lgamma(x) - tf.math.lgamma(y), dtype = tf.float32)
    
@tf.function(jit_compile = True)
def step_4(a_0, b_0, g_0, g_i, n, sigma, seed_s4):

    seed_s4_1, seed_s4_2 = tfp.random.split_seed( seed_s4, n=2, salt='step_4_seed_split')
    
    alpha_star = tf.math.exp(tfp.distributions.Normal(loc = tf.math.log(g_0), scale = sigma).sample(seed=seed_s4_1))
    alpha_0_star = tf.reduce_sum(alpha_star)

    alpha_0 = tf.reduce_sum(g_0)

    log_exp_0 = -b_0*(alpha_0_star - alpha_0)
    log_prod_alpha_ratio = tf.reduce_sum(tf.math.log(alpha_star) - tf.math.log(g_0)) + (a_0 -1)*(tf.math.log(alpha_0_star) - tf.math.log(alpha_0))
    log_prod_gamma_alpha_ratio = n*(lratioGamma(alpha_0_star, alpha_0) + tf.reduce_sum(lratioGamma(g_0, alpha_star)))
    log_g_power = tf.reduce_sum((alpha_star - g_0)*tf.reduce_sum(tf.math.log(1e-30 + g_i), axis = 0))

    r = tf.math.exp(log_exp_0 + log_prod_alpha_ratio + log_prod_gamma_alpha_ratio + log_g_power)

    bool_alpha = tf.cast(tfp.distributions.Uniform(low = 0., high = 1.).sample(seed=seed_s4_2)<r, dtype = tf.float32)
    g_0 = (1-bool_alpha)*g_0 + bool_alpha*alpha_star

    return g_0
    
# @tf.function(jit_compile = True)
# def step_4_adaptive(a_0, b_0, g_0, g_i, n, Sigma):
    
#     alpha_star = tf.math.exp(tfp.distributions.MultivariateNormalTriL( loc = tf.math.log(g_0), scale_tril = tf.linalg.cholesky(Sigma)).sample())
#     alpha_0_star = tf.reduce_sum(alpha_star)

#     alpha_0 = tf.reduce_sum(g_0)

#     log_exp_0 = -b_0*(alpha_0_star - alpha_0)
#     log_prod_alpha_ratio = tf.reduce_sum(tf.math.log(alpha_star) - tf.math.log(g_0)) + (a_0 -1)*(tf.math.log(alpha_0_star) - tf.math.log(alpha_0))
#     log_prod_gamma_alpha_ratio = n*(lratioGamma(alpha_0_star, alpha_0) + tf.reduce_sum(lratioGamma(g_0, alpha_star)))
#     log_g_power = tf.reduce_sum((alpha_star - g_0)*tf.reduce_sum(tf.math.log(1e-30 + g_i), axis = 0))

#     r = tf.math.exp(log_exp_0 + log_prod_alpha_ratio + log_prod_gamma_alpha_ratio + log_g_power)

#     bool_alpha = tf.cast(tfp.distributions.Uniform(low = 0., high = 1.).sample()<r, dtype = tf.float32)
#     g_0 = (1-bool_alpha)*g_0 + bool_alpha*alpha_star

#     return g_0

def MV_MCMC_initialization(X_ij, one_n_j, K, a_0, b_0, seed_initialization):

    seed_initialization_splitted = tfp.random.split_seed( seed_initialization, n=5, salt='seed_initialization_split')

    n = tf.shape(X_ij)[0]
    one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    # initialization 
    theta_j_k = tfp.distributions.Dirichlet(concentration = one_n_j).sample(K, seed = seed_initialization_splitted[0])

    alpha_0 = tfp.distributions.Gamma( concentration = a_0, rate = b_0).sample(seed = seed_initialization_splitted[1])
    g_0 = alpha_0*tfp.distributions.Dirichlet(concentration = tf.ones(int(K))).sample(seed = seed_initialization_splitted[2])

    g_i = tfp.distributions.Dirichlet(concentration = g_0).sample(n, seed = seed_initialization_splitted[3])

    Z_ij = step_1(one_hot_X, g_i, theta_j_k, seed_initialization_splitted[4])

    nr_accepted = 0.0

    return Z_ij, theta_j_k, g_i, g_0, nr_accepted

def MV_MCMC_step(one_hot_X, one_n_j, K, a_0, b_0, Z_ij, theta_j_k, g_i, g_0, sigma, seed_step):
    
    seed_step_splitted = tfp.random.split_seed( seed_step, n=4, salt='seed_step_split')

    n = tf.cast(tf.shape(g_i)[0], dtype = tf.float32)
        
    Z_ij = step_1(one_hot_X, g_i, theta_j_k, seed_step_splitted[0])
    one_hot_Z = tf.one_hot(Z_ij, K)

    seed_s2, seed_s2_carry  = tfp.random.split_seed( seed_step_splitted[1], n=2, salt='seed_s2_carry_'+str(n))
    theta_j_k = step_2(one_hot_X, one_n_j, one_hot_Z, seed_s2)

    counter = 0
    while tf.reduce_any(tf.math.is_nan(theta_j_k)):

        seed_s2, seed_s2_carry  = tfp.random.split_seed( seed_s2_carry, n=2, salt='seed_s2_carry_'+str(n)+str(counter))

        theta_j_k = step_2(one_hot_X, one_n_j, one_hot_Z, seed_s2)

        counter = counter +1

    g_i = step_3(one_hot_Z, g_0, seed_step_splitted[2])

    g_0_new = step_4(a_0, b_0, g_0, g_i, n, sigma, seed_step_splitted[3])

    return Z_ij, theta_j_k, g_i, g_0_new

# def MV_MCMC_step_adaptive(one_hot_X, one_n_j, K, a_0, b_0, Z_ij, theta_j_k, g_i, g_0, Sigma):

#     n = tf.cast(tf.shape(g_i)[0], dtype = tf.float32)
        
#     Z_ij = step_1(one_hot_X, g_i, theta_j_k)
#     one_hot_Z = tf.one_hot(Z_ij, K)

#     theta_j_k = step_2(one_hot_X, one_n_j, one_hot_Z)

#     g_i = step_3(one_hot_Z, g_0)

#     g_0_new = step_4_adaptive(a_0, b_0, g_0, g_i, n, Sigma)

#     return Z_ij, theta_j_k, g_i, g_0_new


def MV_MCMC_from_start(X_ij, one_n_j, K, a_0, b_0, sigma, MCMC_iterations, seed_MCMC):

    seed_initialization, seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=2, salt='seed_MCMC')

    n = tf.shape(X_ij)[0]
    one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    # initialization 
    Z_ij, theta_j_k, g_i, g_0, nr_accepted = MV_MCMC_initialization(X_ij, one_n_j, K, a_0, b_0, seed_initialization)

    seed_step_to_split  = tfp.random.split_seed( seed_step_to_split, n=MCMC_iterations, salt='seed_MCMC_step_per_iter')

    def body(input, t):

        Z_ij, theta_j_k, g_i, g_0, nr_accepted = input

        Z_ij, theta_j_k, g_i, g_0_new = MV_MCMC_step(one_hot_X, one_n_j, K, a_0, b_0, Z_ij, theta_j_k, g_i, g_0, sigma, seed_step_to_split[t])

        nr_accepted = nr_accepted + tf.cast(tf.reduce_all(g_0_new!=g_0), dtype = tf.float32)

        return Z_ij, theta_j_k, g_i, g_0_new, nr_accepted
    
    output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (Z_ij, theta_j_k, g_i, g_0, nr_accepted))

    return output

def MV_MCMC_initialized(X_ij, one_n_j, K, a_0, b_0, sigma, initialization_MCMC, MCMC_iterations, seed_MCMC):

    one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=MCMC_iterations, salt='seed_MCMC_step_per_iter_not_init')

    def body(input, t):

        Z_ij, theta_j_k, g_i, g_0, nr_accepted = input

        Z_ij, theta_j_k, g_i, g_0_new = MV_MCMC_step(one_hot_X, one_n_j, K, a_0, b_0, Z_ij, theta_j_k, g_i, g_0, sigma, seed_step_to_split[t])

        nr_accepted = nr_accepted + tf.cast(tf.reduce_all(g_0_new!=g_0), dtype = tf.float32)

        return Z_ij, theta_j_k, g_i, g_0_new, nr_accepted
    
    output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  initialization_MCMC)

    return output


# def MV_MCMC_initialized_adaptive(X_ij, one_n_j, K, a_0, b_0, sigma, initialization_MCMC, MCMC_iterations = 100):

#     _, _, _, MCMC_g_0, _  = initialization_MCMC
#     initialization_MCMC_2 = initialization_MCMC[0], initialization_MCMC[1], initialization_MCMC[2], MCMC_g_0[-1,...], initialization_MCMC[4]

#     hat_mu = tf.reduce_mean(tf.math.log(MCMC_g_0), axis = 0, keepdims=True)
#     MCMC_size = tf.cast(tf.shape(MCMC_g_0)[0], dtype = tf.float32)
#     Sigma = sigma*tf.eye(tf.shape(MCMC_g_0)[1])/10 + sigma*(tf.einsum("ni,nj->ij", tf.math.log(MCMC_g_0) - hat_mu, tf.math.log(MCMC_g_0) - hat_mu))/MCMC_size

#     one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

#     def body(input, t):

#         Z_ij, theta_j_k, g_i, g_0, nr_accepted = input

#         Z_ij, theta_j_k, g_i, g_0_new = MV_MCMC_step_adaptive(one_hot_X, one_n_j, K, a_0, b_0, Z_ij, theta_j_k, g_i, g_0, Sigma)

#         nr_accepted = nr_accepted + tf.cast(tf.reduce_all(g_0_new!=g_0), dtype = tf.float32)

#         return Z_ij, theta_j_k, g_i, g_0_new, nr_accepted
    
#     output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  initialization_MCMC_2)

#     return output

def step_tau_with_data(X_ij, one_n_j, X_ij_unobserved, batch_size, N):

    n = tf.shape(X_ij)[0]

    iterations = tf.math.floordiv(N, batch_size)
    final_size = N - iterations*batch_size

    freq_0 = fast_frequency(X_ij, one_n_j)

    subX_ij    = tf.gather(X_ij,   tf.where(freq_0==1)[:,0], axis =0)
    subfreq_0  = tf.gather(freq_0, tf.where(freq_0==1)[:,0], axis =0)

    def cond(iteration, input):

        return iteration<iterations

    def body(iteration, input):

        freq_prev = input

        freq_updated = fast_frequency_batched(subX_ij, X_ij_unobserved[iteration*batch_size:(iteration*batch_size + batch_size),:], one_n_j)

        freq_join = tf.stack((freq_prev, freq_updated), axis = 1)

        new_freq = tf.reduce_max(freq_join, axis = 1)

        return iteration+1, new_freq

    _, freq_1 = tf.while_loop(cond, body, loop_vars = (0, subfreq_0))

    freq_final = fast_frequency_batched_final_size(subX_ij, X_ij_unobserved[iterations*batch_size:(iterations*batch_size+final_size),:], one_n_j)

    freq_join_final = tf.stack((freq_1, freq_final), axis = 1)

    freq_2 = tf.reduce_max(freq_join_final, axis = 1)

    tau = tf.reduce_sum(tf.cast(freq_2==1, dtype = tf.float32))

    return tau

@tf.function(jit_compile = True)
def population_sampler(g_0, theta_j_k, J, batch_size, seed_pop_sampler):
    
    seed_pop_sampler_splitted  = tfp.random.split_seed( seed_pop_sampler, n=3, salt='seed_pop_sampler_split')
    
    g_i_unobserved = tfp.distributions.Dirichlet(concentration = g_0).sample(batch_size, seed = seed_pop_sampler_splitted[0])

    choose_profile_unobserved = tfp.distributions.Categorical(probs = g_i_unobserved).sample(J, seed = seed_pop_sampler_splitted[1])
    choose_profile_unobserved = tf.transpose(choose_profile_unobserved)

    one_hot = tf.one_hot(choose_profile_unobserved, tf.shape(theta_j_k)[0])

    individual_profile = tf.einsum("kjn,ijk->ijn", theta_j_k, one_hot)

    X_ij_unobserved = tfp.distributions.Categorical(probs = individual_profile).sample(seed = seed_pop_sampler_splitted[2])

    return X_ij_unobserved

@tf.function(jit_compile = True)
def population_sampler_final(g_0, theta_j_k, J, batch_size, final_size, seed_pop_sampler):

    seed_pop_sampler_splitted  = tfp.random.split_seed( seed_pop_sampler, n=3, salt='seed_pop_sampler_split')
    
    g_i_unobserved = tfp.distributions.Dirichlet(concentration = g_0).sample(batch_size, seed = seed_pop_sampler_splitted[0])

    choose_profile_unobserved = tfp.distributions.Categorical(probs = g_i_unobserved).sample(J, seed = seed_pop_sampler_splitted[1])
    choose_profile_unobserved = tf.transpose(choose_profile_unobserved)

    one_hot = tf.one_hot(choose_profile_unobserved, tf.shape(theta_j_k)[0])

    individual_profile = tf.einsum("kjn,ijk->ijn", theta_j_k, one_hot)

    X_ij_unobserved = tfp.distributions.Categorical(probs = individual_profile).sample(seed = seed_pop_sampler_splitted[2])

    return X_ij_unobserved[:final_size,...]

def step_tau(g_0, theta_j_k, X_ij, one_n_j, J, batch_size, N, seed_step_tau):

    n = tf.shape(X_ij)[0]

    iterations = tf.math.floordiv(N-n, batch_size)
    final_size = N-n - iterations*batch_size

    freq_0 = fast_frequency(X_ij, one_n_j)

    subX_ij    = tf.gather(X_ij,   tf.where(freq_0==1)[:,0], axis =0)
    subfreq_0  = tf.gather(freq_0, tf.where(freq_0==1)[:,0], axis =0)

    seed_step_tau_splitted  = tfp.random.split_seed( seed_step_tau, n=iterations+1, salt='seed_split_for_tau_batches')

    def cond(iteration, input):

        return iteration<iterations

    def body(iteration, input):

        freq_prev = input

        X_ij_unobserved = population_sampler(g_0, theta_j_k, J, batch_size, seed_step_tau_splitted[iteration])

        freq_updated = fast_frequency_batched(subX_ij, X_ij_unobserved, one_n_j)

        freq_join = tf.stack((freq_prev, freq_updated), axis = 1)

        new_freq = tf.reduce_max(freq_join, axis = 1)

        return iteration+1, new_freq

    _, freq_1 = tf.while_loop(cond, body, loop_vars = (0, subfreq_0))

    X_ij_unobserved_final = population_sampler_final(g_0, theta_j_k, J, batch_size, final_size, seed_step_tau_splitted[-1])
    freq_final = fast_frequency_batched_final_size(subX_ij, X_ij_unobserved_final, one_n_j)

    freq_join_final = tf.stack((freq_1, freq_final), axis = 1)

    freq_2 = tf.reduce_max(freq_join_final, axis = 1)

    tau = tf.reduce_sum(tf.cast(freq_2==1, dtype = tf.float32))

    return tau

def MV_MCMC_initialized_tau(X_ij, one_n_j, K, a_0, b_0, sigma, initialization_MCMC, N, MCMC_iterations, batch_size, seed_MCMC):

    J = tf.shape(one_n_j)[0]

    seed_step_to_split  = tfp.random.split_seed( seed_MCMC, n=MCMC_iterations, salt='seed_MCMC_step_per_iter_not_init_with_tau')
    tau = 0.

    one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    def body(input, t):

        (Z_ij, theta_j_k, g_i, g_0, nr_accepted), _ = input

        seed_step_without_tau, seed_step_with_tau = tfp.random.split_seed( seed_step_to_split[t], n=2, salt='without_and_with_tau')

        Z_ij, theta_j_k, g_i, g_0_new = MV_MCMC_step(one_hot_X, one_n_j, K, a_0, b_0, Z_ij, theta_j_k, g_i, g_0, sigma, seed_step_without_tau)

        nr_accepted = nr_accepted + tf.cast(tf.reduce_all(g_0_new!=g_0), dtype = tf.float32)

        tau = step_tau(g_0, theta_j_k, X_ij, one_n_j, J, batch_size, N, seed_step_with_tau)

        return (Z_ij, theta_j_k, g_i, g_0_new, nr_accepted), tau
    
    MCMC_output, Tau = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  (initialization_MCMC, tau))

    return MCMC_output, Tau

# def MV_MCMC_initialized_adaptive_tau(X_ij, unique_X_ij, one_n_j, K, a_0, b_0, sigma, initialization_MCMC, N, MCMC_iterations = 100, batch_size = 1000):

#     J = tf.shape(one_n_j)[0]

#     _, _, _, MCMC_g_0, _  = initialization_MCMC

#     tau = 0.

#     initialization_MCMC_2 = initialization_MCMC[0], initialization_MCMC[1], initialization_MCMC[2], MCMC_g_0[-1,...], initialization_MCMC[4], tau

#     hat_mu = tf.reduce_mean(tf.math.log(MCMC_g_0), axis = 0, keepdims=True)
#     MCMC_size = tf.cast(tf.shape(MCMC_g_0)[0], dtype = tf.float32)
#     Sigma = sigma*tf.eye(tf.shape(MCMC_g_0)[1])/10 + sigma*(tf.einsum("ni,nj->ij", tf.math.log(MCMC_g_0) - hat_mu, tf.math.log(MCMC_g_0) - hat_mu))/MCMC_size

#     one_hot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

#     def body(input, t):

#         Z_ij, theta_j_k, g_i, g_0, nr_accepted, _ = input

#         Z_ij, theta_j_k, g_i, g_0_new = MV_MCMC_step_adaptive(one_hot_X, one_n_j, K, a_0, b_0, Z_ij, theta_j_k, g_i, g_0, Sigma)

#         nr_accepted = nr_accepted + tf.cast(tf.reduce_all(g_0_new!=g_0), dtype = tf.float32)

#         tau = step_tau(g_0_new, theta_j_k, unique_X_ij, J, batch_size, N)

#         return Z_ij, theta_j_k, g_i, g_0_new, nr_accepted, tau
    
#     output = tf.scan(body, tf.range(0, MCMC_iterations), initializer =  initialization_MCMC_2)

#     return output