# Hierarchical Dirichlet process to estimate disclosure risk

We propose a hierarchical Dirichlet process Gibbs sampler to estimate the disclure risk of dataset. This repository provide all the code to reproduce the results and eventually apply the algorithm to their own data.

Requirements:
- Pandas 2.2.1
- Numpy 1.26.4
- tensorflow 2.14.0
- tensorflow_probability 0.22.1

In the repository the user will find three folders. 

_data_, containing all the input and output data of our experiments:
- clean, which contains the input and output of the experiments on real data and without structural zeros;
- log_linear_model, which contains the output of the experiments on log-linear models;
- structural_zeros, which contains the input and output of the experiments on the real data with structural zeros;
- synthetic, which contains the input and output of the experiments on synthetic data;
- SZ_repeated_mc, which contains the output of the experiments on the real data with structural zeros repeated to build credible intervals.

_experiments_, containing all the experiments we run for the paper:
- log_linear_model, which contains the experiments on log-linear models;
- MCMC_repeated, which contains the experiments on the real data with structural zeros repeated to build credible intervals;
- NY, which contains the experiments on real data and without structural zeros;
- Synthetic, which contains the experiments on synthetic data;
- SZ_NY, which contains the experiments on the real data with structural zeros;

_scripts_, which contains all the .py files to run our method:
- BNP_structural_zeros.py, the algorithm from section 4.1;
- BNP.py, the algorithm from section 3.2.1;
- log_linear_model.py, the implementation of log-linear models;
- manage_constraints.py, the code to transform overlapping constraints in disjoint constraints as in Section 4.2 of reference [6] of the paper;
- mixed_membership_model.py, a simulator for the mixed membership model;
- MVR.py, the algorithm from reference [4] of the paper.
