import os
import argparse

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import time

import sys
sys.path.append('scripts/')
from mixed_membership_model import *
from Manrique_Vallier import *

n = 10000

input_path  = "data/synthetic/"
output_path = "data/synthetic/MV/"+str(n)+"/"
if not os.path.exists(output_path):

    os.makedirs(output_path)
    os.makedirs(output_path+"Check/")


name_sim = "synthdata_tau_"+str(n)+"_"+str(task_id)+"_"

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

# Get the list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Enable memory growth for each GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


a_0, b_0 = 2, 1

K_list = [20, 30, 40, 60, 80, 100, 120, 140, 160, 180]
K = K_list[task_id]

X_ij    = tf.convert_to_tensor(np.load(input_path+"synth_X_ij_"+str(n)+".npy"), dtype = tf.int32)
one_n_j = tf.convert_to_tensor(np.load(input_path+"NY_one_n_j.npy"), dtype = tf.float32)

batch_size = 2**13
N = 1150934

MCMC_iterations = 100

seed_to_use = 1234+n

tf.random.set_seed((seed_to_use+n))
np.random.seed((seed_to_use+n))

seed_MCMC_start, seed_MCMC_after_start  = tfp.random.split_seed( seed_to_use, n=2, salt='seed_MCMC_start_'+str(n))

sigma_list = [0.005, 0.0045, 0.0035, 0.003, 0.0025, 0.0025, 0.002, 0.002, 0.002, 0.0015] 
sigma = sigma_list[task_id]

nr_accepted_list = []
tau_list = []

MCMC_Z_ij_init, MCMC_theta_j_k_init, MCMC_g_i_init, MCMC_g_0_init, nr_accepted_init = MV_MCMC_from_start(X_ij, one_n_j, K, a_0, b_0, sigma, MCMC_iterations, seed_MCMC_start)

nr_accepted_list.append(nr_accepted_init)

string = ["Start MCMC with K "+str(K), "\n"]
f= open(output_path+"Check/"+name_sim+".txt", "a")
f.writelines(string)
f.close()

initialization_MCMC = MCMC_Z_ij_init[-1,...], MCMC_theta_j_k_init[-1,...], MCMC_g_i_init[-1,...], MCMC_g_0_init[-1,...], nr_accepted_init[-1,...]

seed_MCMC_before_tau, seed_MCMC_after_tau  = tfp.random.split_seed( seed_MCMC_after_start, n=2, salt='seed_MCMC_before_after_tau_'+str(n)+str(task_id))

before_tau_iterations = 3000
seed_MCMC_before_tau_splitted  = tfp.random.split_seed( seed_MCMC_before_tau, n=before_tau_iterations, salt='seed_MCMC_before_tau_'+str(n)+str(task_id))
for i in range(before_tau_iterations):

    string = ["Batch number "+str(i), "\n"]
    f= open(output_path+"Check/"+name_sim+".txt", "a")
    f.writelines(string)
    f.close()

    MCMC_Z_ij, MCMC_theta_j_k, MCMC_g_i, MCMC_g_0, nr_accepted = MV_MCMC_initialized(X_ij, one_n_j, K, a_0, b_0, sigma, initialization_MCMC, MCMC_iterations, seed_MCMC_before_tau_splitted[i])
    initialization_MCMC = MCMC_Z_ij[-1,...], MCMC_theta_j_k[-1,...], MCMC_g_i[-1,...], MCMC_g_0[-1,...], nr_accepted[-1,...]

    MCMC_nr_accepted = tf.concat(nr_accepted_list, axis = 0)
    acceptance_rate = MCMC_nr_accepted[-1]/tf.cast(tf.shape(MCMC_nr_accepted)[0], dtype = tf.float32)

    string = ["Acceptance rate "+str(np.round(acceptance_rate.numpy(), 3)), "\n"]
    f= open(output_path+"Check/"+name_sim+".txt", "a")
    f.writelines(string)
    f.close()
    
    nr_accepted_list.append(nr_accepted)

after_tau_iterations = 1000
seed_MCMC_after_tau_splitted  = tfp.random.split_seed( seed_MCMC_after_tau, n=after_tau_iterations, salt='seed_MCMC_after_tau_'+str(n)+str(task_id))
for i in range(after_tau_iterations):
    
    string = ["Tau Batch number "+str(i), "\n"]
    f= open(output_path+"Check/"+name_sim+".txt", "a")
    f.writelines(string)
    f.close()

    start = time.time()
    (MCMC_Z_ij, MCMC_theta_j_k, MCMC_g_i, MCMC_g_0, nr_accepted), tau = MV_MCMC_initialized_tau(X_ij, one_n_j, K, a_0, b_0, sigma, initialization_MCMC, N, MCMC_iterations, batch_size, seed_MCMC_after_tau_splitted[i])
    initialization_MCMC = MCMC_Z_ij[-1,...], MCMC_theta_j_k[-1,...], MCMC_g_i[-1,...], MCMC_g_0[-1,...], nr_accepted[-1,...]
    end = time.time()- start

    MCMC_nr_accepted = tf.concat(nr_accepted_list, axis = 0)
    acceptance_rate = MCMC_nr_accepted[-1]/tf.cast(tf.shape(MCMC_nr_accepted)[0], dtype = tf.float32)

    string = ["Acceptance rate "+str(np.round(acceptance_rate.numpy(), 3)), "\n"]
    string2 = ["with time "+str(end), "\n"]
    f= open(output_path+"Check/"+name_sim+".txt", "a")
    f.writelines(string)
    f.writelines(string2)
    f.close()
    
    nr_accepted_list.append(nr_accepted)

    tau_list.append(tau)
    MCMC_tau = tf.concat(tau_list, axis = 0)
    np.save(output_path+name_sim+"MCMC_tau.npy", np.array(MCMC_tau))

