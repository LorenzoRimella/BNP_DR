import os
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import time

import sys
sys.path.append('scripts/')
from mixed_membership_model import *
from BNP import *

task_id =  int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

n = [1000, 5000, 10000][task_id]
input_path =  "data/synthetic/"
output_path = "data/synthetic/BNP/sampling/"+str(n)+"/"
if not os.path.exists(output_path):

    os.makedirs(output_path)
    os.makedirs(output_path+"Check/")


name_sim = "synthdata_tau_"+str(n)+"_"

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

# Get the list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    # Enable memory growth for each GPU
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

seed_to_use = 42+n

tf.random.set_seed((seed_to_use+n))
np.random.seed((seed_to_use+n))

K_initial = 100
K_max = 2*K_initial

a_0, b_0 = 2., 1.
a_1, b_1 = 2., 1. 
a_2, b_2 = 1., 1. 
sigma = 0.1

prior_MCMC = a_0, b_0, a_1, b_1, a_2, b_2, sigma

X_ij    = tf.convert_to_tensor(np.load(input_path+"synth_X_ij_"+str(n)+".npy"), dtype = tf.int32)
one_n_j = tf.convert_to_tensor(np.load(input_path+"NY_one_n_j.npy"), dtype = tf.float32)

batch_size = 2**11
N = 712174

MCMC_iterations = 100

seed_MCMC_start, seed_MCMC_after_start  = tfp.random.split_seed( seed_to_use, n=2, salt='seed_MCMC_start_'+str(n))

output = BNP_MCMC_from_start(X_ij, one_n_j, K_initial, K_max, prior_MCMC, MCMC_iterations, seed_MCMC_start, "multiple", "fixed a,b")

string = ["Start MCMC with n "+str(n), "\n"]
f= open(output_path+"Check/"+name_sim+".txt", "a")
f.writelines(string)
f.close()

initialization_MCMC = tuple(elem[-1,...] for elem in output)

seed_MCMC_before_tau, seed_MCMC_after_tau  = tfp.random.split_seed( seed_MCMC_after_start, n=2, salt='seed_MCMC_before_after_tau_'+str(n))

before_tau_iterations = 1000
seed_MCMC_before_tau_splitted  = tfp.random.split_seed( seed_MCMC_before_tau, n=before_tau_iterations, salt='seed_MCMC_before_tau_'+str(n))
for i in range(before_tau_iterations):

    string = ["Batch number "+str(i), "\n"]
    f= open(output_path+"Check/"+name_sim+".txt", "a")
    f.writelines(string)
    f.close()

    output = BNP_MCMC_initialized(X_ij, one_n_j, K_max, prior_MCMC, initialization_MCMC, MCMC_iterations, seed_MCMC_before_tau_splitted[i], "multiple", "fixed a,b")
    initialization_MCMC = tuple(elem[-1,...] for elem in output)

K_list = []
alpha_0_list = []
alpha_i_list = []
tau_list = []

after_tau_iterations = 1000
seed_MCMC_after_tau_splitted  = tfp.random.split_seed( seed_MCMC_after_tau, n=after_tau_iterations, salt='seed_MCMC_after_tau_'+str(n))
for i in range(after_tau_iterations):
    
    string = ["Tau Batch number "+str(i), "\n"]
    f= open(output_path+"Check/"+name_sim+".txt", "a")
    f.writelines(string)
    f.close()

    start = time.time()
    tau_output = BNP_MCMC_initialized_tau(X_ij, one_n_j, K_max, prior_MCMC, initialization_MCMC, N, MCMC_iterations, batch_size, seed_MCMC_after_tau_splitted[i], "multiple", "fixed a,b", "sampling")
    initialization_MCMC = tuple(elem[-1,...] for elem in tau_output[0])
    end = time.time()- start

    K_list.append(tau_output[0][-1])
    alpha_0_list.append(tau_output[0][2])
    alpha_i_list.append(tau_output[0][3][0])

    MCMC_K = tf.concat(K_list, axis = 0)
    np.save(output_path+name_sim+"MCMC_K.npy", np.array(MCMC_K))
    mean_k = tf.reduce_mean(MCMC_K)
    
    MCMC_alpha_0 = tf.concat(alpha_0_list, axis = 0)
    np.save(output_path+name_sim+"MCMC_alpha_0.npy", np.array(MCMC_alpha_0))

    MCMC_alpha_i = tf.concat(alpha_i_list, axis = 0)
    np.save(output_path+name_sim+"MCMC_alpha_i.npy", np.array(MCMC_alpha_i))

    string = ["Mean K "+str(np.round(mean_k.numpy(), 3)), "\n"]
    string2 = ["with time "+str(end), "\n"]
    f= open(output_path+"Check/"+name_sim+".txt", "a")
    f.writelines(string)
    f.writelines(string2)
    f.close()
    
    tau_list.append(tau_output[1])
    MCMC_tau = tf.concat(tau_list, axis = 0)
    np.save(output_path+name_sim+"MCMC_tau.npy", np.array(MCMC_tau))

