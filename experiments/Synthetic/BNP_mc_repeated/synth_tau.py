import os
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import time

import sys
sys.path.append('DisclosureRisk/scripts/')
from mixed_membership_model import *
from BNP import *
from MVR import *

task_id =  int(os.getenv("SLURM_ARRAY_TASK_ID"))-1

n = [1000, 5000, 10000][task_id]
input_path =  "DisclosureRisk/data/synthetic/"
output_path = "DisclosureRisk/data/synthetic/BNP/repeated_mc/"+str(n)+"/"
if not os.path.exists(output_path):

    os.makedirs(output_path)
    os.makedirs(output_path+"Check/")


name_sim = "synthdata_tau_"+str(n)+"_"

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

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

batch_size = 256
N = 712174

MCMC_iterations = 1000

X_ij_full    = tf.convert_to_tensor(np.load(input_path+"synth_X_ij_full.npy"), dtype = tf.int32)
one_n_j = tf.convert_to_tensor(np.load(input_path+"NY_one_n_j.npy"), dtype = tf.float32)

repetitions = 50

seed_to_use_reps = tfp.random.split_seed( seed_to_use, n = repetitions, salt='seed_reps_'+str(n))

string = ["Start experiment with n "+str(n), "\n"]
f= open(output_path+"Check/"+name_sim+".txt", "a")
f.writelines(string)
f.close()

for reps in range(repetitions):
      
	seed_to_use_reps_X, seed_to_use_reps_MCMC = tfp.random.split_seed( seed_to_use_reps[reps], n=2, salt='seed_MCMC_start_'+str(n)) 
	indexes = np.random.choice(np.linspace(0, N-1, N), size=N, replace=False)
	X_ij_full = tf.gather(X_ij_full, tf.cast(indexes, tf.int32), axis = 0)

	X_ij    = X_ij_full[:n,:]

	tau_in_data = step_tau_with_data(X_ij, one_n_j, X_ij_full[n:N,:], batch_size, N)

	string = ["Simulation nr "+str(reps), "\n"]
	f= open(output_path+"Check/"+name_sim+".txt", "a")
	f.writelines(string)
	f.close()
      
	string = ["tau: "+str(tau_in_data.numpy()), "\n"]
	f= open(output_path+name_sim+"_tau.txt", "a")
	f.writelines(string)
	f.close()

	seed_MCMC_start, seed_MCMC_after_start  = tfp.random.split_seed( seed_to_use_reps_MCMC, n=2, salt='seed_MCMC_start_'+str(n))

	output = BNP_MCMC_from_start(X_ij, one_n_j, K_initial, K_max, prior_MCMC, MCMC_iterations, seed_MCMC_start, "multiple", "fixed a,b")

	initialization_MCMC = tuple(elem[-1,...] for elem in output)

	seed_MCMC_before_tau, seed_MCMC_after_tau  = tfp.random.split_seed( seed_MCMC_after_start, n=2, salt='seed_MCMC_before_after_tau_'+str(n))

	before_tau_iterations = 1
	seed_MCMC_before_tau_splitted  = tfp.random.split_seed( seed_MCMC_before_tau, n=before_tau_iterations, salt='seed_MCMC_before_tau_'+str(n))
	for i in range(before_tau_iterations):

		string = ["Batch number "+str(i), "\n"]
		f= open(output_path+"Check/"+name_sim+".txt", "a")
		f.writelines(string)
		f.close()

		output = BNP_MCMC_initialized(X_ij, one_n_j, K_max, prior_MCMC, initialization_MCMC, MCMC_iterations, seed_MCMC_before_tau_splitted[i], "multiple", "fixed a,b")
		initialization_MCMC = tuple(elem[-1,...] for elem in output)

	tau_list = []

	after_tau_iterations = 108
	seed_MCMC_after_tau_splitted  = tfp.random.split_seed( seed_MCMC_after_tau, n=after_tau_iterations, salt='seed_MCMC_after_tau_'+str(n))
	for i in range(after_tau_iterations):
	
		string = ["Tau Batch number "+str(i), "\n"]
		f= open(output_path+"Check/"+name_sim+".txt", "a")
		f.writelines(string)
		f.close()

		start = time.time()
		tau_output = BNP_MCMC_initialized_tau(X_ij, one_n_j, K_max, prior_MCMC, initialization_MCMC, N, MCMC_iterations, batch_size, seed_MCMC_after_tau_splitted[i], "multiple", "fixed a,b", "Monte Carlo")
		initialization_MCMC = tuple(elem[-1,...] for elem in tau_output[0])
		end = time.time()- start

		tau_list.append(tau_output[1])
		MCMC_tau = tf.concat(tau_list, axis = 0)

		np.save(output_path+name_sim+"MCMC_tau_n"+str(n)+"_reps"+str(reps)+".npy", np.array(MCMC_tau))

