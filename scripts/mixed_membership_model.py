import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp


def param_mixed_membership_model(K, one_n_j, a_0, b_0, n):

    J = tf.shape(one_n_j)[0]

    enhance = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor([2.]*2, dtype = tf.float32), axis = 0), axis = 1)
    mixture_expansion = tf.expand_dims(tf.expand_dims((tf.cast(tf.linspace(K-5, K, K), dtype = tf.float32)/2), axis = 1), axis = 1)

    prior_to_use =  tf.concat((enhance*mixture_expansion, tf.ones((K, 1, tf.shape(one_n_j)[1]- tf.shape(enhance)[2]))), axis = -1)*tf.expand_dims(one_n_j, axis = 0)
    
    theta_j_k = tfp.distributions.Dirichlet(concentration = prior_to_use).sample() # tfp.distributions.Dirichlet(concentration = one_n_j).sample(K) 

    alpha_0 = tfp.distributions.Gamma( concentration = a_0, rate = b_0).sample()

    g_0 = alpha_0*tfp.distributions.Dirichlet(concentration = tf.ones(int(K))).sample()

    g_i = tfp.distributions.Dirichlet(concentration = g_0).sample(n)

    choose_profile = tfp.distributions.Categorical(probs = g_i).sample(J)
    choose_profile = tf.transpose(choose_profile)

    # indeces = tf.cast(tf.linspace(0, J-1, J), dtype = tf.int32)
    # indeces = tf.expand_dims(indeces, axis = 0)*tf.ones(tf.shape(choose_profile), dtype = tf.int32)
    # choose_profile_stack = tf.stack((choose_profile, indeces), axis = 2)

    individual_profile = tf.einsum("kjn,ijk->ijn", theta_j_k, tf.one_hot(choose_profile, K))

    X_ij = tfp.distributions.Categorical(probs = individual_profile).sample()

    return X_ij, theta_j_k, g_i, g_0

def param_mixed_membership_model_given(K, one_n_j, g_0, theta_j_k, n):

    J = tf.shape(one_n_j)[0]

    g_i = tfp.distributions.Dirichlet(concentration = g_0).sample(n)

    choose_profile = tfp.distributions.Categorical(probs = g_i).sample(J)
    choose_profile = tf.transpose(choose_profile)

    individual_profile = tf.einsum("kjn,ijk->ijn", theta_j_k, tf.one_hot(choose_profile, K))

    X_ij = tfp.distributions.Categorical(probs = individual_profile).sample()

    return X_ij, g_i

@tf.function(jit_compile = True)
def param_mixed_membership_model_given_theta_g(K, one_n_j, theta_j_k, g_0, n):

    J = tf.shape(one_n_j)[0]

    g_i = tfp.distributions.Dirichlet(concentration = g_0).sample(n)

    choose_profile = tfp.distributions.Categorical(probs = g_i).sample(J)
    choose_profile = tf.transpose(choose_profile)

    individual_profile = tf.einsum("kjn,ijk->ijn", theta_j_k, tf.one_hot(choose_profile, K))

    X_ij = tfp.distributions.Categorical(probs = individual_profile).sample()

    return X_ij

@tf.function(jit_compile = True)
def batch_nonparam_mixed_membership_model(one_n_j, a_0, b_0, a, b, batch_size, nr_simulation_DP = 200):

    J = tf.shape(one_n_j)[0]

    alpha_0 = tfp.distributions.Gamma( concentration = a_0, rate = b_0).sample()

    theta_k_G_0  = tfp.distributions.Dirichlet(concentration = one_n_j).sample(nr_simulation_DP)

    V_k_0      = tfp.distributions.Beta(concentration1 = 1, concentration0 = alpha_0).sample(nr_simulation_DP)
    log_cumprod_0 = tf.math.cumsum(tf.math.log(1-V_k_0+1e-35)) - tf.math.log(1-V_k_0+1e-35)
    weight_k_G_0 = tf.math.exp(tf.math.log(V_k_0+1e-35) + log_cumprod_0)

    alpha_i = tfp.distributions.Gamma( concentration = a, rate = b).sample(batch_size)

    weight_k_cumsum_G_0 = tf.math.cumsum(weight_k_G_0)

    concentration1_k_i = tf.expand_dims(alpha_i, axis = -1)*tf.expand_dims(weight_k_G_0, axis = 0)
    concentration0_k_i = tf.expand_dims(alpha_i, axis = -1)*(1 - tf.expand_dims(weight_k_cumsum_G_0, axis = 0))

    V_k_i = tfp.distributions.Beta(concentration1 = concentration1_k_i, concentration0 = concentration0_k_i).sample()
    log_cumprod_i = tf.math.cumsum(tf.math.log(1-V_k_i+1e-35), axis = 1) - tf.math.log(1-V_k_i+1e-35)
    weight_k_G_i = tf.math.exp(tf.math.log(V_k_i+1e-35) + log_cumprod_i)

    from_G_i = tfp.distributions.Categorical(probs = weight_k_G_i).sample(J)
    from_G_i = tf.transpose(from_G_i)

    indeces = tf.cast(tf.linspace(0, J-1, J), dtype = tf.int32)
    indeces = tf.expand_dims(indeces, axis = 0)*tf.ones(tf.shape(from_G_i), dtype = tf.int32)
    choose_profile_stack = tf.stack((from_G_i, indeces), axis = 2)

    theta_i  = tf.gather_nd(theta_k_G_0, choose_profile_stack, batch_dims=0)

    X_ij_batch = tfp.distributions.Categorical(probs = theta_i).sample()

    return X_ij_batch

def nonparam_mixed_membership_model(one_n_j, a_0, b_0, a, b, n_batch, batch_size, nr_simulation_DP):

    X_ij_0 = batch_nonparam_mixed_membership_model(one_n_j, a_0, b_0, a, b, batch_size, nr_simulation_DP)

    def body(input, batch):

        X_ij_batch = batch_nonparam_mixed_membership_model(one_n_j, a_0, b_0, a, b, batch_size, nr_simulation_DP)

        return X_ij_batch
    
    X_ij_batched = tf.scan(body, tf.range(0, n_batch), initializer = X_ij_0)

    X_ij = tf.reshape(X_ij_batched, (tf.reduce_prod(tf.shape(X_ij_batched)[:-1]), tf.shape(X_ij_batched)[-1]))

    return X_ij

# def assign_state(n_j, X_ij):
    
#     cumprod_n = tf.cast(tf.concat((tf.math.cumprod(n_j[1:][::-1])[::-1], [1.], ), axis = 0), dtype = tf.int32)
#     states = tf.reduce_sum(tf.expand_dims(cumprod_n, axis = 0)*X_ij, axis = 1)

#     return states

# def encode_state(states, n_j):

#     reminder = tf.cast(states, dtype = tf.float32)
#     cumprod_n = tf.cast(tf.concat((tf.math.cumprod(n_j[1:][::-1])[::-1], [1.], ), axis = 0), dtype = tf.float32)

#     def body(input, t):

#         reminder, _ = input

#         division = tf.math.floordiv(reminder, cumprod_n[t])
#         reminder = reminder - division*cumprod_n[t]

#         return reminder, division

#     _, encoded_state = tf.scan(body, tf.range(0, tf.shape(cumprod_n)[0]), initializer = (reminder, tf.zeros(tf.shape(reminder))))

#     return tf.cast(tf.transpose(encoded_state), dtype = tf.int32)

# def frequency_1(X_ij):

#     equal_control = tf.cast(tf.reduce_all(tf.expand_dims(X_ij, axis = 0) == tf.expand_dims(X_ij, axis = 1), axis = -1), tf.float32)
#     index = tf.where(tf.reduce_sum(equal_control, axis = 1)==1)

#     return index, tf.gather(X_ij, index)

# def frequency(X_ij, one_n_j):

#     equal_control = tf.cast(tf.reduce_all(tf.expand_dims(X_ij, axis = 0) == tf.expand_dims(X_ij, axis = 1), axis = -1), tf.float32)
#     freq = tf.reduce_sum(equal_control, axis = 1)

#     return freq

@tf.function(jit_compile = True)
def fast_frequency(X_ij, one_n_j):
    
    max_nj = tf.cast(tf.shape(one_n_j)[0], dtype = tf.float32)
    onehot_X = tf.one_hot(X_ij, tf.shape(one_n_j)[1])

    square_matrix = tf.einsum("ijn, Ijn->iI", onehot_X, onehot_X)
    square_matrix = tf.where(square_matrix==max_nj, tf.ones(tf.shape(square_matrix)), tf.zeros(tf.shape(square_matrix)))
    frequencies = tf.reduce_sum(square_matrix, axis = -1)

    return frequencies

@tf.function(jit_compile = True)
def fast_frequency_batched(X_ij, X_ij_batch, one_n_j):

    max_nj = tf.cast(tf.shape(one_n_j)[0], dtype = tf.float32)
    onehot_X         = tf.one_hot(X_ij, tf.shape(one_n_j)[1])
    onehot_X_batched = tf.one_hot(X_ij_batch, tf.shape(one_n_j)[1])

    square_matrix = tf.einsum("ijn, Ijn->iI", onehot_X, onehot_X_batched)
    square_matrix = tf.where(square_matrix==max_nj, tf.ones(tf.shape(square_matrix)), tf.zeros(tf.shape(square_matrix)))
    frequencies = 1 + tf.reduce_sum(square_matrix, axis = -1)

    return frequencies

@tf.function(jit_compile = True)
def fast_frequency_batched_final_size(X_ij, X_ij_batch, one_n_j):

    max_nj = tf.cast(tf.shape(one_n_j)[0], dtype = tf.float32)
    onehot_X         = tf.one_hot(X_ij, tf.shape(one_n_j)[1])
    onehot_X_batched = tf.one_hot(X_ij_batch, tf.shape(one_n_j)[1])

    square_matrix = tf.einsum("ijn, Ijn->iI", onehot_X, onehot_X_batched)
    square_matrix = tf.where(square_matrix==max_nj, tf.ones(tf.shape(square_matrix)), tf.zeros(tf.shape(square_matrix)))
    frequencies = 1 + tf.reduce_sum(square_matrix, axis = -1)

    return frequencies