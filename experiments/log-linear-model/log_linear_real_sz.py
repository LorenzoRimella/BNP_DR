import os
import argparse

import pandas as pd

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import time

import sys
sys.path.append('scripts/')
from mixed_membership_model import *
from BNP import *
from MVR import *
from log_linear_model import *

name_sim   = "real_log_linear"
input_path =  "data/structural_zeros/"
output_path = "data/log_linear_model/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path = output_path+"structural_zeros/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Enable JIT compilation
tf.config.optimizer.set_jit(True)

seed_to_use = 42

tf.random.set_seed((seed_to_use))
np.random.seed((seed_to_use))

n_list = [1000, 5000, 10000]

N = 953076

string_1 = ["#################################", "\n"]
string_2 = ["RUN SZ real data", "\n"]
f= open(output_path+name_sim+".txt", "a")
f.writelines(string_1)
f.writelines(string_2)
f.close()

one_n_j = tf.convert_to_tensor(np.load(input_path+"SZ_NY_one_n_j.npy"), dtype = tf.int32)
X_ij_full    = tf.convert_to_tensor(np.load(input_path+"SZ_NY_X_ij_full.npy"), dtype = tf.int32)
d_j = tf.reduce_sum(one_n_j, axis = 1)

all_cells = full_states(d_j).numpy()

for n in n_list:
      
	# Load data
	X_ij    = np.load(input_path+"SZ_NY_X_ij_"+str(n)+".npy")

	batch_size = 2048
	tau_in_data = step_tau_with_data(X_ij, one_n_j, X_ij_full[n:N,:], batch_size, N)

	df = pd.DataFrame(X_ij)

	df_count = df
	df_count["count"] = fast_frequency(X_ij, one_n_j).numpy()

	df_drop = df_count.drop_duplicates().reset_index(drop=True)

	membership_matrix_list = []
	y_list = []
	for i in range(40):
		# print(i)
		membership_matrix_batch = tf.reduce_all(tf.expand_dims(all_cells[int(256608/4)*i:int(256608/4)*(i+1)], axis = 1)==tf.expand_dims(tf.convert_to_tensor(df_drop.values[:,:-1], dtype = tf.int32), axis = 0), axis = -1)
		membership_matrix_list.append(membership_matrix_batch)

		y_batch = tf.einsum("ij,j->i", tf.cast(membership_matrix_batch, dtype = tf.float32), tf.convert_to_tensor(df_drop.values[:,-1], dtype = tf.float32))
		y_list.append(y_batch)

	string = ["n="+str(n)+" with tau: "+str(tau_in_data.numpy()), "\n"]
	f= open(output_path+name_sim+".txt", "a")
	f.writelines(string)
	f.close()

	pi = n/N

	x = tf.concat((tf.ones((tf.shape(all_cells)[0],1)), tf.convert_to_tensor(all_cells, dtype = tf.float32)), axis = -1)
	y = tf.concat(y_list, axis = 0)

	beta_tf = tf.Variable(tf.zeros(x.shape[1]), dtype = tf.float32)

	optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
      
	string_1 = ["#################################", "\n"]
	string_2 = ["independent log-linear", "\n"]
	f= open(output_path+name_sim+".txt", "a")
	f.writelines(string_1)
	f.writelines(string_2)
	f.close()

	# independent log-linear
	start = time.time()
	loss_optim, beta_optim, log_linear_tau = log_linear_disclosure_risk(beta_tf, pi, x, y, optimizer, 2000)
	string = ["n="+str(n)+" with tau from log-linear: "+str(log_linear_tau.numpy())+" in "+str((time.time()-start)/3600), "\n"]
	f= open(output_path+name_sim+".txt", "a")
	f.writelines(string)
	f.close()
	
	# interaction log-linear
	x_interaction = x

	for i in range(1, x_interaction.shape[1]):

		for j in range(i+1, x_interaction.shape[1]):
			x_interaction = tf.concat((x_interaction, x[:,i:i+1]*x[:,j:j+1]), axis = -1)

	beta_tf = tf.Variable(tf.zeros(x_interaction.shape[1]), dtype = tf.float32)

	string_1 = ["#################################", "\n"]
	string_2 = ["interaction log-linear", "\n"]
	f= open(output_path+name_sim+".txt", "a")
	f.writelines(string_1)
	f.writelines(string_2)
	f.close()
      
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

	start = time.time()
	loss_optim_interaction, beta_optim_interaction, log_linear_tau_interaction = log_linear_disclosure_risk(beta_tf, pi, x_interaction, y, optimizer, 2000)      
	string = ["n="+str(n)+" with tau from log-linear with interaction: "+str(log_linear_tau_interaction.numpy())+" in "+str((time.time()-start)/3600), "\n"]
	f= open(output_path+name_sim+".txt", "a")
	f.writelines(string)
	f.close()
      
	string_1 = ["#################################", "\n"]
	string_2 = ["interaction+penalization log-linear", "\n"]
	f= open(output_path+name_sim+".txt", "a")
	f.writelines(string_1)
	f.writelines(string_2)
	f.close()
      
	lambda_0_list = [tf.constant(1, dtype = tf.float32), tf.constant(10, dtype = tf.float32), tf.constant(100, dtype = tf.float32), 
		  	 tf.constant(200, dtype = tf.float32), tf.constant(500, dtype = tf.float32), ]

	for lambda_0 in lambda_0_list:

		beta_tf = tf.Variable(tf.zeros(x_interaction.shape[1]), dtype = tf.float32)

		optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

		start = time.time()
		loss_optim_interaction, beta_optim_interaction, log_linear_tau_interaction = penalized_log_linear_disclosure_risk(beta_tf, pi, x_interaction, y, lambda_0, optimizer, 2000)         
		string = ["n="+str(n)+" with tau from log-linear with "+str(lambda_0.numpy())+" penalized interaction: "+str(log_linear_tau_interaction.numpy())+" in "+str((time.time()-start)/3600), "\n"]
		f= open(output_path+name_sim+".txt", "a")
		f.writelines(string)
		f.close()
     
