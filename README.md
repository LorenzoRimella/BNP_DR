# Hierarchical Dirichlet process to estimate disclosure risk

We propose a hierarchical Dirichlet process Gibbs sampler to estimate the disclosure risk of the dataset. This repository provides all the code to reproduce the results and eventually apply the algorithm to their own data.

Requirements:
- Pandas 2.2.1
- Numpy 1.26.4
- tensorflow 2.14.0
- tensorflow_probability 0.22.1

In the repository, the user will find three folders. 

__data__, containing all the input and output data of our experiments:
- _clean_, which contains the input and output of the experiments on real data and without structural zeros;
- _log_linear_model_, which contains the output of the experiments on log-linear models;
- _structural_zeros_, which contains the input and output of the experiments on the real data with structural zeros;
- _synthetic_, which contains the input and output of the experiments on synthetic data;
- _SZ_repeated_mc_, which contains the output of the experiments on the real data with structural zeros repeated to build credible intervals.

__experiments__, containing all the experiments we run for the paper:
- _log_linear_model_, which contains the experiments on log-linear models;
- _MCMC_repeated_, which contains the experiments on the real data with structural zeros repeated to build credible intervals;
- _NY_, which contains the experiments on real data and without structural zeros;
- _Synthetic_, which contains the experiments on synthetic data;
- _SZ_NY_, which contains the experiments on the real data with structural zeros;

__scripts__, which contains all the .py files to run our method:
- _BNP_structural_zeros.py_, the algorithm from section 4.1;
- _BNP.py_, the algorithm from section 3.2.1;
- _log_linear_model.py_, the implementation of log-linear models;
- _manage_constraints.py_, the code to transform overlapping constraints into disjoint constraints as in Section 4.2 of reference [6] of the paper;
- _mixed_membership_model.py_, a simulator for the mixed membership model;
- _MVR.py_, the algorithm from reference [4] of the paper.

Jupiter notebooks are provided to guide the user through the different procedures of cleaning data and running algorithms. 
