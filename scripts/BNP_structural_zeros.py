import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

def nonempy_intersection_index(S_d, mu):
    expanded_mu = tf.expand_dims(mu, axis = 0)*tf.ones(tf.shape(S_d), dtype = tf.int32)

    S_d_compare_mu = tf.where(S_d==0,         expanded_mu, S_d)
    mu_compare_S_d = tf.where(expanded_mu==0, S_d,         expanded_mu)

    return tf.where(tf.reduce_all(S_d_compare_mu==mu_compare_S_d, axis = 1))[:,0]

def what_to_expand(mu, domain_j, ComparList):
    expanded_mu               = tf.expand_dims(mu, axis = 0)*tf.ones(tf.shape(ComparList), dtype = tf.int32)
    bool_components_to_expand = tf.reduce_all(tf.stack((ComparList!=0, expanded_mu==0), axis =-1), axis = -1)
    
    components_to_expand_list = []
    domain_to_expand_list     = []
    CompareList_components_list = []

    for i in range(tf.shape(bool_components_to_expand)[0]):

        components_to_expand      = tf.where(bool_components_to_expand[i])[:,0]
        components_to_expand_list.append(components_to_expand)

        domain_to_expand          = tf.gather(domain_j, components_to_expand)
        domain_to_expand_list.append(domain_to_expand)

        ComparList_components = tf.gather(ComparList[i], components_to_expand)
        CompareList_components_list.append(ComparList_components)

    return components_to_expand_list, domain_to_expand_list, CompareList_components_list

def expanded_mu(mu, components_to_expand_current, domain_to_expand_current, CompareList_components_current):
    new_mu_list = []
    new_mu = mu

    for i in range(tf.shape(domain_to_expand_current)[0]):
        increment_current_current = tf.cast(tf.one_hot(components_to_expand_current[i], tf.shape(mu)[0]), dtype = tf.int32)

        for iter in range(domain_to_expand_current[i]):
            new_mu = new_mu + increment_current_current
            # print(new_mu)
            if new_mu[components_to_expand_current[i]]==CompareList_components_current[i]:
                new_blue_print = new_mu

            else:
                new_mu_list.append(new_mu)

        new_mu = new_blue_print

    return tf.stack(new_mu_list, axis = 0)

def create_mu_to_add(mu, components_to_expand_list, domain_to_expand_list, CompareList_components_list):
    expanded_mu_list = []

    for i in range(len(components_to_expand_list)):

        components_to_expand_current = components_to_expand_list[i]
        domain_to_expand_current     = domain_to_expand_list[i]
        CompareList_components_current = CompareList_components_list[i]

        expanded_mu_list.append(expanded_mu(mu, components_to_expand_current, domain_to_expand_current, CompareList_components_current))

    return tf.concat(expanded_mu_list, axis = 0)

# def combine_Compare_list(components_to_expand_list, domain_to_expand_list, CompareList_components_list):

#     components_to_expand_list_update   = components_to_expand_list[0]
#     domain_to_expand_list_update       = domain_to_expand_list[0]
#     CompareList_components_list_update = CompareList_components_list[0]

#     for i in range(1, len(components_to_expand_list)):

#         for elem_index in range(tf.shape(components_to_expand_list[i])[0]):

#             elem_1 = components_to_expand_list[i][elem_index]
#             elem_2 = domain_to_expand_list[i][elem_index]
#             elem_3 = CompareList_components_list[i][elem_index]

#             if tf.reduce_all(components_to_expand_list_update!=elem_1):
#                 components_to_expand_list_update   = tf.concat((components_to_expand_list_update, elem_1), axis = 0)
#                 domain_to_expand_list_update       = tf.concat((domain_to_expand_list_update, elem_2), axis = 0)
#                 CompareList_components_list_update = tf.concat((CompareList_components_list_update, elem_3), axis = 0)

#     return components_to_expand_list_update, domain_to_expand_list_update, CompareList_components_list_update

def iteration_disjoint_constraint(domain_j, current_constraint):

    # sort by number of explained cells
    free_cells_dim = tf.cast(current_constraint==0, dtype = tf.int32)*tf.expand_dims(domain_j, axis = 0)
    free_cells = tf.reduce_prod(tf.where(free_cells_dim==0, tf.ones(tf.shape(free_cells_dim), dtype = tf.int32), free_cells_dim), axis =1)

    sorting_index = tf.argsort(free_cells)[::-1]
    current_constraint_sorted = tf.gather(current_constraint, sorting_index, axis = 0)

    # initialize everything
    S_d = current_constraint_sorted[0:1,:]
    Pending = current_constraint_sorted[1:,:]

    i = 0
    while tf.shape(Pending)[0]!=0:

        i = i +1

        print("inside_loop", i)

        mu = Pending[0,:]
        Pending = Pending[1:,:]

        ComparList = tf.gather(S_d, nonempy_intersection_index(S_d, mu), axis = 0)

        if tf.shape(ComparList)[0]==0:

            S_d     = tf.concat((S_d, tf.expand_dims(mu, axis = 0)), axis = 0)

        else:

            components_to_expand_list, domain_to_expand_list, CompareList_components_list = what_to_expand(mu, domain_j, ComparList)

            mu_to_add = create_mu_to_add(mu, components_to_expand_list, domain_to_expand_list, CompareList_components_list)

            if tf.shape(mu_to_add)[0]!=0:
                S_d = tf.concat((S_d, mu_to_add), axis = 0)

    return S_d

def disjoint_constraint(domain_j, current_constraint):

    new_constraint = iteration_disjoint_constraint(domain_j, current_constraint)

    counter = 0
    i = 0
    while (tf.shape(new_constraint)[0]!= tf.shape(current_constraint)[0] and counter <1000):

        i = i +1

        print("outside_loop", i)

        current_constraint = new_constraint

        new_constraint = iteration_disjoint_constraint(domain_j, current_constraint)

        counter = counter + 1

    return new_constraint



