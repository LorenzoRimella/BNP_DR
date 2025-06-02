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
from BNP_structural_zeros import *
from MVR import *

repetitions = 50

task_id_current = int(os.getenv("SLURM_ARRAY_TASK_ID"))-1
task_id_current = [146, 149, 51, 56, 62, 70, 88, 99, 111, 126, 127, 130, 134, 137, 141][task_id_current]

if task_id_current<repetitions:
	task_id = 0

else:
	if task_id_current<(2*repetitions):
		task_id = 1

	else:
		task_id = 2

reps = task_id_current%repetitions

n = [1000, 5000, 10000][task_id]

input_path =  "DisclosureRisk/data/structural_zeros/"
output_path = "DisclosureRisk/data/SZ_repeated_mc/"+str(n)+"/"
if not os.path.exists(output_path):

    os.makedirs(output_path)
    os.makedirs(output_path+"Check/")


name_sim = "SZ_repeated_tau_"+str(n)+"_rep_"+str(reps)

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

seed_to_use = 42+n

tf.random.set_seed((seed_to_use+n))
np.random.seed((seed_to_use+n))

K_initial = 50
K_max = 8*K_initial

a_0, b_0 = 2., 1.
a_1, b_1 = 2., 1. 
a_2, b_2 = 1., 1. 
sigma = 0.1

prior_MCMC = a_0, b_0, a_1, b_1, a_2, b_2, sigma

batch_size = 256
N = 953076

MCMC_iterations = 100

indexes_list = []
for iter_rep in range(repetitions):
	indexes = np.random.choice(np.linspace(0, N-1, N), size=N, replace=False)
	indexes_list.append(indexes)

X_ij_full    = tf.convert_to_tensor(np.load(input_path+"SZ_NY_X_ij_full.npy"), dtype = tf.int32)
one_n_j = tf.convert_to_tensor(np.load(input_path+"SZ_NY_one_n_j.npy"), dtype = tf.float32)

disjoint_constraint = tf.convert_to_tensor(np.load(input_path+"SZ_NY_disjoint_constraint.npy"), dtype = tf.int32)
disjoint_constraints = disjoint_constraint -1 

seed_to_use_reps = tfp.random.split_seed( seed_to_use, n = repetitions, salt='seed_reps_'+str(n))

string = ["Start experiment with n "+str(n), "\n"]
f= open(output_path+"Check/"+name_sim+".txt", "a")
f.writelines(string)
f.close()
      
seed_to_use_reps_X, seed_to_use_reps_MCMC = tfp.random.split_seed( seed_to_use_reps[reps], n=2, salt='seed_MCMC_start_'+str(n)) 

indexes = indexes_list[reps]
X_ij_full = tf.gather(X_ij_full, tf.cast(indexes, tf.int32), axis = 0)

X_ij    = X_ij_full[:n,:]

tau_in_data = step_tau_with_data(X_ij, one_n_j, X_ij_full[n:N,:], batch_size, N)

string_1 = ["Simulation nr "+str(reps), "\n"]
string_2 = ["tau: "+str(tau_in_data.numpy()), "\n"]
f= open(output_path+"Check/"+name_sim+".txt", "a")
f.writelines(string_1)
f.writelines(string_2)
f.close()

np.save(output_path+name_sim+"_tau.npy", tau_in_data.numpy())
      
seed_MCMC_start, seed_MCMC_after_start  = tfp.random.split_seed( seed_to_use_reps_MCMC, n=2, salt='seed_MCMC_start_'+str(n))

output = SZ_BNP_MCMC_from_start(X_ij, disjoint_constraints, one_n_j, K_initial, K_max, prior_MCMC, MCMC_iterations, seed_MCMC_start, "multiple", "fixed a,b")

initialization_MCMC = tuple(elem[-1,...] for elem in output)

seed_MCMC_before_tau, seed_MCMC_after_tau  = tfp.random.split_seed( seed_MCMC_after_start, n=2, salt='seed_MCMC_before_after_tau_'+str(n))

before_tau_iterations = 19
seed_MCMC_before_tau_splitted  = tfp.random.split_seed( seed_MCMC_before_tau, n=before_tau_iterations, salt='seed_MCMC_before_tau_'+str(n))
for i in range(before_tau_iterations):

	string = ["Batch number "+str(i), "\n"]
	f= open(output_path+"Check/"+name_sim+".txt", "a")
	f.writelines(string)
	f.close()

	output = SZ_BNP_MCMC_initialized(X_ij, disjoint_constraints, one_n_j, K_max, prior_MCMC, initialization_MCMC, MCMC_iterations, seed_MCMC_before_tau_splitted[i], "multiple", "fixed a,b")
	initialization_MCMC = tuple(elem[-1,...] for elem in output)

tau_list = []

after_tau_iterations = 200
seed_MCMC_after_tau_splitted  = tfp.random.split_seed( seed_MCMC_after_tau, n=after_tau_iterations, salt='seed_MCMC_after_tau_'+str(n))
for i in range(after_tau_iterations):

	string = ["Tau Batch number "+str(i), "\n"]
	f= open(output_path+"Check/"+name_sim+".txt", "a")
	f.writelines(string)
	f.close()

	start = time.time()
	tau_output = SZ_BNP_MCMC_initialized_tau(X_ij, disjoint_constraints, one_n_j, K_max, prior_MCMC, 
						initialization_MCMC, MCMC_iterations, N, batch_size, 
						seed_MCMC_after_tau_splitted[i], "multiple", "fixed a,b", "Monte Carlo")
	initialization_MCMC = tuple(elem[-1,...] for elem in tau_output[0])
	end = time.time()- start

	tau_list.append(tau_output[1])
	MCMC_tau = tf.concat(tau_list, axis = 0)

	np.save(output_path+name_sim+"MCMC_tau_n"+str(n)+"_reps"+str(reps)+".npy", np.array(MCMC_tau))

