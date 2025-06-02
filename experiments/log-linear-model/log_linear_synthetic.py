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

name_sim   = "synthetic_log_linear"
input_path =  "data/synthetic/"
output_path = "data/log_linear_model/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path = output_path+"synthetic/"
if not os.path.exists(output_path):
	os.makedirs(output_path)


# Enable JIT compilation
tf.config.optimizer.set_jit(True)

seed_to_use = 42

tf.random.set_seed((seed_to_use))
np.random.seed((seed_to_use))

n_list = [1000, 5000, 10000]

N = 712174

string_1 = ["#################################", "\n"]
string_2 = ["RUN synt data", "\n"]
f= open(output_path+name_sim+".txt", "a")
f.writelines(string_1)
f.writelines(string_2)
f.close()

one_n_j = tf.convert_to_tensor(np.load(input_path+"NY_one_n_j.npy"), dtype = tf.int32)
X_ij_full    = tf.convert_to_tensor(np.load(input_path+"synth_X_ij_full.npy"), dtype = tf.int32)
d_j = tf.reduce_sum(one_n_j, axis = 1)

all_cells = full_states(d_j).numpy()

df_dict = {}
xy_dict = {}
for n in n_list:
      
	# Load data
	X_ij    = np.load(input_path+"synth_X_ij_"+str(n)+".npy")

	batch_size = 2048
	tau_in_data = step_tau_with_data(X_ij, one_n_j, X_ij_full[n:N,:], batch_size, N)

	df = pd.DataFrame(X_ij)

	df_count = df
	df_count["count"] = fast_frequency(X_ij, one_n_j).numpy()

	df_dict[n] = df_count.drop_duplicates().reset_index(drop=True)

	membership_matrix = np.all(np.expand_dims(all_cells, axis = 1)==np.expand_dims(df_dict[n].values[:,:-1], axis = 0), axis = -1).astype(float)

	y = np.einsum("ij,j->i", membership_matrix, df_dict[n].values[:,-1])
	xy_dict[n] = np.concatenate((tf.ones((tf.shape(all_cells)[0],1)), all_cells, np.expand_dims(y,axis=-1)), axis = -1)

	string = ["n="+str(n)+" with tau: "+str(tau_in_data.numpy()), "\n"]
	f= open(output_path+name_sim+".txt", "a")
	f.writelines(string)
	f.close()

	pi = n/N

	x = tf.concat(tf.convert_to_tensor(xy_dict[n][:,:-1], dtype = tf.float32), axis = -1)
	y = tf.convert_to_tensor(xy_dict[n][:,-1], dtype = tf.float32)

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
	x = tf.concat(tf.convert_to_tensor(xy_dict[n][:,:-1], dtype = tf.float32), axis = -1)
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
      
	lambda_0_list = [tf.constant(1, dtype = tf.float32), tf.constant(10, dtype = tf.float32), tf.constant(100, dtype = tf.float32)]

	for lambda_0 in lambda_0_list:

		beta_tf = tf.Variable(tf.zeros(x_interaction.shape[1]), dtype = tf.float32)

		optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

		start = time.time()
		loss_optim_interaction, beta_optim_interaction, log_linear_tau_interaction = penalized_log_linear_disclosure_risk(beta_tf, pi, x_interaction, y, lambda_0, optimizer, 2000)         
		string = ["n="+str(n)+" with tau from log-linear with "+str(lambda_0.numpy())+" penalized interaction: "+str(log_linear_tau_interaction.numpy())+" in "+str((time.time()-start)/3600), "\n"]
		f= open(output_path+name_sim+".txt", "a")
		f.writelines(string)
		f.close()