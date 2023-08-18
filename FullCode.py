#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 12:03:16 2023

@author: marcuevas
"""
#%%

import pandas as pd  
import numpy as np
from sys import argv
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
import os
import networkx as nx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
import math
import string
import seaborn as sns
from FullCodeFunctions import *
import scienceplots
import scipy
from networkx.drawing.nx_pydot import graphviz_layout  
from matplotlib import rcParams
import distinctipy as dsp
import seaborn
from scipy.sparse import csgraph
from scipy.linalg import expm
from palettable.colorbrewer.sequential import Purples_9, Oranges_9,Greys_9,Reds_9, Blues_9 ,RdPu_9
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%%


head = 0 #header
separator=','
data_path = '/home/marcuevas/Doctorado_Victor/fpi/primeros_datos/quinta_tarea/datos/NaRedona_observed_6layers_final.csv' #str(argv[0])
abundance_path = '/home/marcuevas/Doctorado_Victor/fpi/primeros_datos/quinta_tarea/datos/Plant abundance.csv' #if needed

 
# =============================================================================
# data_path = '/home/mar/Documentos/Doctorado_Victor/fpi/primeros_datos/quinta_tarea/datos/NaRedona_observed_6layers_new.csv' #str(argv[0])
# abundance_path = '/home/mar/Documentos/Doctorado_Victor/fpi/primeros_datos/quinta_tarea/datos/Plant abundance.csv' #if needed
# 
# =============================================================================
#%%
'''saving'''

# creates a folder Results in data_path folder with folders DataFrames (df_dir) and Plots (plt_dir) to save results
df_dir, plt_dir = creat_dir(data_path)


'''Aggregate data and obtain probability of participation of each plant sp in each function '''

# For that we aggregate the dataset at the level of plant sp and functions
# For each plant sp and function we have the frequency of ocurrence for each vector present : f_{i,alpha,j}, where i are the plant sp,  alpha the functions and j the vectors
# we obtain the probability that each i participates in each function alpha as:  
# P_{i,alpha} = 1 - \prod_{j=1}^{j} (1 - f_{i,alpha,j})


#construct Pandas DataFrame with columns ['plant_sp', 'interaction_type', 'probability', 'abundance', 'cover'] 
df = df_prob(head, separator, data_path, df_dir, abundance_path = abundance_path)


'''We build a matrix dataframe where rows are all plant sp and columns are the different functions, each element is the probability of the given plant to participate in the corresponding functin'''
#additionally we build species: List of plant species and functions:  List of interaction types and compute their length


df_matrix, species, functions = df_mat(df, df_dir)
len_functions = len(functions) # this computes the number of ecological functions which will be used many times
len_species = len(species) # this computes the number of plant species functions which will be used many times


'''Building the product dataframe'''

#In order to calculate the probability that a plant species participates in two ecological functions it is assumed that the probabilities of  participating in each function are independent
# Hence we obtain the probability that plant species i participates in the ecological functions (alpha,beta) as: P_{i,apha,beta} = P_{i,apha} x P_{i,beta}
# we obtain df_matrix_prod with columns ['plant_sp', 'interaction_1 times interaction_2', ...] and functions_pairs List of lists that constructs all possible functions pairs

df_matrix_prod, functions_pairs = df_prod(df_matrix, df_dir, len_functions, functions)



'''Labels for plots'''

#Ecological function labels
#We are going to create acronyms for ecological functions and plant species names that we will use for plots
#ecological functions acronyms are obtained using the first letter of each word 
#plant species are obtained using the first three letters of each word

# create acronym for ecological function for labels
dic_functions= dic_fun(functions)

# create acronym for plant species
dic_species= dic_sp(species,len_species)

#%%
'''
Species-Functions map
---------------------------------------------
'''

'''P'''

#we are going to construct the P_matrix_df where columns represent functions and rows plant species 
#This is equivalent to df_matrix but sorting rows and columns
# rows and columns are sorted separetly such that higher probabilities appear first 
# we also compute the transpose P_matrix equivalent to P_matrix_df in array form and P_matrix_t the corresponding transpose
P_matrix_df, P_matrix, P_matrix_t = p_matrix(df_matrix, functions)

function_sorted = list(P_matrix_df.columns[1:].to_numpy())
species_sorted =  list(P_matrix_df['plant_sp'].to_numpy())
#Functions labels
labels_f = []
for i in P_matrix_df.columns[1:].to_numpy():
    labels_f.append(dic_functions[i])
#Plant species labels
labels_p = []
for i in P_matrix_df['plant_sp'].to_numpy():
    labels_p.append(dic_species[i])

 

#italic  plant labels
labels_p_ent = P_matrix_df["plant_sp"].to_list()
labels_p_it = []
for strg in labels_p_ent:
    words = strg.split(" ", 1)
    if len(words) >1:
        strg_it = r'%s $\mathit{%s}$' % (words[0].capitalize(), words[1:][0])
        #strg_it = rf"{strg_it}"
    else:
        strg_it = f'{words[0].capitalize()}'
    labels_p_it.append(strg_it)


'''P_t plot'''

#imshow of p_matrix transpose 
fontsize = 15
fig, ax = plt.subplots(figsize=(12,8), dpi=100)   
shw = ax.imshow(P_matrix_t, cmap = 'viridis', vmin= 0 , vmax=1)
plt.xticks(np.arange(0,len(P_matrix),1),labels_p_it , fontsize=fontsize, rotation = 90)
# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(np.shape(P_matrix)[1]), minor=False)
ax.xaxis.tick_top()
ax.set_yticklabels(labels_f, minor=False, fontsize=fontsize)
#colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(shw, cax=cax, ticks= np.linspace(0, 1,5))
cb.ax.tick_params(labelsize=fontsize,width = 2, length = 5)
cb.set_label('Participation strength', labelpad=30,rotation=-90, fontsize = fontsize)
# ticks
for t in cb.ax.get_yticklabels():
     t.set_fontsize(fontsize)   
ax.tick_params(axis='both', which='major', labelsize=fontsize, width=2, length=5)
 
#plant.savefig(plt_dir + '/P_t.pdf', bbox_inches='tight' ,  dpi=100 ) #save

#%%
# =============================================================================
# fig, ax = plt.subplots(figsize=(12,8), dpi=100)   
# for fun in range(len_functions):
#     ax.hist(P_matrix_t[fun], bins=20)
# 
# =============================================================================

'P_t * P'
# imshow of P_t * P
fontsize = 30
P_t_times_P = np.matmul(P_matrix_t, P_matrix) #calculate
# v_min and v_max 
v_min = int(np.min(P_t_times_P))
v_max = int(np.max(P_t_times_P))+1
fig, ax = plt.subplots(figsize=(12,8), dpi=100)   
shw = ax.imshow(P_t_times_P, cmap = 'viridis', vmin=v_min , vmax=v_max)
#colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(shw, cax=cax, ticks= np.linspace(v_min, v_max,5))
cb.ax.tick_params(labelsize=fontsize,width = 3, length = 8)
cb.set_label('No. plaths connecting two functions', labelpad=30,rotation=-90, fontsize = fontsize)
#ticks
for t in cb.ax.get_yticklabels():
     t.set_fontsize(fontsize)    
# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(np.shape(P_matrix)[1]), minor=False)
ax.set_xticks(np.arange(np.shape(P_matrix)[1]), minor=False)
ax.xaxis.tick_top()
ax.set_yticklabels(labels_f, minor=False, fontsize=fontsize);
ax.set_xticklabels(labels_f, minor=False, fontsize=fontsize);
fig.tight_layout()
ax.tick_params(axis='both', which='major', labelsize=fontsize, width=3, length=7)
 
#plant.savefig(plt_dir + '/P_t_times_P.pdf', bbox_inches='tight',  dpi=100  ) #save



'''P *P_t '''

# imshow of P_t * P
fontsize = 30
P_times_P_t = np.matmul(P_matrix, P_matrix_t)
# v_min and v_max 
v_min = int(np.min(P_times_P_t))
v_max = int(np.max(P_times_P_t))+1
fig, ax = plt.subplots(figsize=(12,12), dpi=100)   
shw = ax.imshow(P_times_P_t, cmap = 'viridis', vmin=v_min , vmax=v_max)
#colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(shw, cax=cax,ticks= np.linspace(v_min, v_max,5))
cb.ax.tick_params(labelsize=fontsize,width = 3, length = 8)
cb.set_label('No. of shared functions by two plant species', labelpad=30,rotation=-90, fontsize = fontsize)
#ticks
for t in cb.ax.get_yticklabels():
     t.set_fontsize(fontsize)
# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(np.shape(P_matrix)[0]), minor=False)
ax.set_xticks(np.arange(np.shape(P_matrix)[0]), minor=False)
ax.xaxis.tick_top()
fig.tight_layout()
ax.set_yticklabels(labels_p, minor=False, fontsize=fontsize);
ax.set_xticklabels(labels_p, minor=False,rotation =90, fontsize=fontsize);
ax.tick_params(axis='both', which='major', labelsize=fontsize, width=3, length=7)

 
#plant.savefig(plt_dir + '/P_times_P_t.pdf', bbox_inches='tight',  dpi=100  ) #save

#%%

# '''Nestedness'''

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from nestedness_Lib import *
# import scipy as sp
# import numpy as np
# import random
# import copy
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import networkx as nx




# def randomization(G0,nodesA, nodesB, threshold, save_dir, max_reps=max_reps):
#     types = nx.get_node_attributes(G0, "bipartite")

#     CA = []
#     CB = []
#     list_TypeA = []
#     list_TypeB = []
#     for nodea in G0.nodes():
#         if types[nodea] == "TypeA":
#             if nodea not in list_TypeA:
#                 list_TypeA.append(nodea)
#             for nodeb in G0.nodes():
#                 if types[nodeb] == "TypeB":
#                     if nodeb not in list_TypeB:
#                         list_TypeB.append(nodeb)
#                     if G0.has_edge(nodea, nodeb):
#                         CA.append(nodea)
#                         CB.append(nodeb)

#     num_edges = len(CA)
#     num_A = len(list_TypeA)
#     num_B = len(list_TypeB)
#     num_sp = num_A + num_B
#     print('species and edges:', num_A, num_B, num_sp, num_edges)


#     # Edgelist bootstrap
    
#     rep = 0
#     while rep <= max_reps:
#         random.shuffle(CA)
#         random.shuffle(CB)
#         forbidden = {}
#         edgelist = []
#         CBworking = copy.deepcopy(CB)
#         for i in range(len(CA)):
#             CBcopy = copy.deepcopy(CBworking)
#             if CA[i] in forbidden:
#                 for item in forbidden[CA[i]]:
#                     CBcopy = remove_items(CBcopy, item)
#             else:
#                 forbidden[CA[i]] = []
#             try:
#                 choice = random.choice(range(len(CBcopy)))
#                 if choice != "":
#                     forbidden[CA[i]].append(CBcopy[choice])
#                     edgelist.append((CA[i], CBcopy[choice]))
#                     CBworking.remove(CBcopy[choice])
#             except:
#                 continue
#         G_rep = nx.Graph()
#         G_rep.add_nodes_from(range(len(nodesA)), bipartite = 'TypeA')
#         G_rep.add_nodes_from((nodesB), bipartite = 'TypeB')
#         G_rep.add_edges_from(edgelist)
#         num_edges_rep = len(G_rep.edges())
#         if num_edges_rep == num_edges:
#             rep += 1
#             nx.write_pajek(G_rep, df_dir +"/Networks" + f"/NullModels_G_{threshold}_" + str(rep) + ".gpickle")
#             #break
#             print(rep)
#             cycles = nx.cycle_basis(G_rep)
#             repeated_edges = []
#             for cycle in cycles:
#                 for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
#                     if G_rep.has_edge(u, v):
#                         repeated_edges.append((u, v))
#             print("Repeated edges:", repeated_edges)
#             break

#         else:
#             continue
#             #print('Failure', len(edgelist), num_edges, num_edges_rep)


# def weighted_to_binary(matrix, threshold):
#     """
#     Converts a weighted adjacency matrix to a binary matrix of presence and absence.
    
#     Parameters:
#         matrix (numpy array): A weighted adjacency matrix.
#         threshold (float): A threshold value for the weights. Values above the threshold are considered present, 
#                            while values below the threshold are considered absent.
    
#     Returns:
#         numpy array: A binary matrix of presence and absence.
#     """
#     binary_matrix = np.zeros_like(matrix)
#     binary_matrix[matrix >= threshold] = 1
    
#     return binary_matrix

# max_reps=1
# thresholds = [0, 0.1, 0.3, 0.7]
# for threshold in thresholds:
    
#     binary_matrix = weighted_to_binary(P_matrix, threshold)
#     condA = ~np.all(binary_matrix == 0, axis=1)
#     nodesA = [label for label, condition in zip(labels_p, condA) if condition == True]
#     condB = ~np.all(binary_matrix == 0, axis=0)
#     nodesB = [label for label, condition in zip(labels_f, condA) if condition == True]
#     binary_matrix = binary_matrix[condA]
#     binary_matrix = binary_matrix[:,condB]

#     G0 = nx.Graph()
#     G0.add_nodes_from(range(len(nodesA)), bipartite = 'TypeA')
#     G0.add_nodes_from((nodesB), bipartite = 'TypeB')
#     node_type = nx.get_node_attributes(G0, 'bipartite')
#     for i, fs in enumerate(nodesB):
#         #print (i,fs)
#         #print(i)
#         for j, ps in enumerate(nodesA):
#             if binary_matrix[j][i] !=0:
#                 G0.add_edge(fs, j, weight=binary_matrix[j][i])

#     nP = [node for node in node_type if node_type[node] == 'TypeA']
#     nM = [node for node in node_type if node_type[node] == 'TypeB']
#     M = np.zeros((len(nP),len(nM)))
#     for i, node1 in enumerate(nP):
#         for j, node2 in enumerate(nM):
#             if G0.has_edge(node1,node2):
#                 M[i,j] = 1
#     [WholeNest, ColNest, RowNest] = nodf(M)
#     randomization(G0,nodesA, nodesB, threshold, df_dir, max_reps=max_reps)
    
#     WholeNest_rep = 0
#     ColNest_rep = 0
#     RowNest_rep = 0
#     #df=pd.DataFrame(columns= ['rep', 'WholeNest', 'ColNest', 'RowNest'])
    
#     for rep in range(max_reps):
#         # Read pajek
#         filename =  df_dir +"/Networks" + f"/NullModels_G_{threshold}_" + str(rep+1) + ".gpickle"
#         Graph = nx.read_pajek(filename)

#         #node_type = nx.get_node_attributes(Graph,'type')
#         nP = [node for node in node_type if node_type[node] == 'TypeA']
#         nM = [node for node in node_type if node_type[node] == 'TypeB']
#         M = np.zeros((len(nP),len(nM)))
#         for i, node1 in enumerate(nP):
#             for j, node2 in enumerate(nM):
#                 if Graph.has_edge(node1,node2):
#                     M[i,j] = 1

#         #Compute Nestedness
#         [a, b, c] = nodf(M)
#         WholeNest_rep += a
#         ColNest_rep += b
#         RowNest_rep += c
#     WholeNest = WholeNest/max_reps
#     ColNest_rep =ColNest_rep/max_reps
#     RowNest_rep = RowNest_rep/max_reps

#     print(WholeNest,WholeNest_rep)
#     print(ColNest, ColNest_rep)
#     print(RowNest, RowNest_rep)
#         #df_ = pd.DataFrame([[rep, WholeNest, ColNest, RowNest]], columns = ['rep', 'WholeNest', 'ColNest', 'RowNest'])
#         #df = pd.concat((df, df_))




#     df.to_csv('Nestedness_'+analysis_type+'_LAST.csv')







#%%

















# from nestedness_calculator import NestednessCalculator


# threshold_range = np.arange(0.6,0.1)

# def expected_nodf_values(binary_matrix, num_randomizations=1000):

#     expected_nodf = np.zeros(num_randomizations)

#     # Calculate the row and column sums
#     row_sums = np.sum(binary_matrix, axis=1)
#     col_sums = np.sum(binary_matrix, axis=0)


#     # Generate a null model by preserving row and column sums
#     null_model = np.zeros_like(binary_matrix)

#     n_rows, n_cols = binary_matrix.shape
#     row_i = np.arange(n_rows)
#     col_i = np.arange(n_cols)
#     for i in range(num_randomizations):
#         np.random.shuffle(row_i)
#         np.random.shuffle(col_i)
#         null_model = binary_matrix[row_i, :][:, col_i]

#         # Calculate the NODF index for the null model
#         nodf_result = NestednessCalculator(null_model).nodf(null_model)
#         expected_nodf[i] = nodf_result
        
#     return expected_nodf



# def calculate_nodf(mat, threshold_range):
#     """Calculate the NODF index, expected NODF index, p-values, and z-scores for a binary matrix.

#     :param mat: binary input matrix
#     :type mat: numpy.array
#     :param threshold_range: range of threshold values to use
#     :type threshold_range: numpy.array
#     :return: NODF index, expected NODF index, p-values, and z-scores
#     :rtype: numpy.array, numpy.array, numpy.array, numpy.array
#     """
#     nodf_values = []
#     p_values = []
#     z_scores = []

#     # Calculate row and column sums of binary matrix
#     row_sums = np.sum(mat, axis=1)
#     col_sums = np.sum(mat, axis=0)

#     # Loop over threshold values
#     for threshold in threshold_range:

#         # Convert binary matrix to presence-absence matrix using threshold
#         binary_mat = np.where(mat > threshold, 1, 0)

#         # Calculate NODF index for binary matrix
#         nodf = NestednessCalculator(binary_mat).nodf(binary_mat)
#         nodf_values.append(nodf)

#         print(binary_mat)

#         #Calculate NODF of randomized binary_mat 
#         random_nodf_values = expected_nodf_values(binary_mat, num_randomizations=10)


#     # Calculate expected NODF as mean of randomized NODF values
#     expected_nodf = np.mean(random_nodf_values)
#     for nodf in nodf_values:
#         # Calculate p-value and z-score for each NODF value
#         p_value = 1 - (sum(random_nodf_values > nodf) / len(random_nodf_values))
#         z_score = (nodf - expected_nodf) / np.std(random_nodf_values)
#         p_values.append(p_value)
#         z_scores.append(z_score)

#     return nodf_values, expected_nodf, p_values, z_scores


# # Generate example binary matrix
# mat = P_matrix

# # Define range of threshold values to use
# threshold_range = np.linspace(0, 1, 10)

# nodf_values, expected_nodf, p_values, z_scores = calculate_nodf(mat, threshold_range)

# # Convert the lists to numpy arrays
# nodf_values = np.array(nodf_values)
# expected_nodf_values = np.array(expected_nodf_values)
# p_values = np.array(p_values)
# z_scores = np.array(z_scores)

# # Plot the NODF and expected NODF values as a function of the threshold
# plt.plot(threshold_range, nodf_values, label='NODF')
# plt.plot(threshold_range, expected_nodf_values, label='Expected NODF')
# plt.legend()
# plt.xlabel('Threshold')
# plt.ylabel('NODF')
# plt.show()

# # Plot the p-values and z-scores as a function of the threshold
# plt.plot(threshold_range, p_values, label='p-value')
# plt.plot(threshold_range, z_scores, label='z-score')
# plt.legend()
# plt.xlabel('Threshold')
# plt.show()
#%%


'''Redes'''






def sort_df_matrix(df_matrix, axis_1_sort, axis_0_sort):
   
    df_matrix = df_matrix.loc[:, list(df_matrix.columns[np.invert(np.isin(df_matrix.columns, axis_1_sort))]) + axis_1_sort]
   
    df_matrix = df_matrix.drop([x for x in df_matrix.columns.to_list() if x not in axis_1_sort +['plant_sp']], axis=1) #drop columns of cover and abundance
    df_matrix['plant_sp'] = df_matrix['plant_sp'].astype("category")
    df_matrix['plant_sp'] = df_matrix['plant_sp'].cat.set_categories(axis_0_sort)
    df_matrix = df_matrix.sort_values(['plant_sp'])
    df_matrix = df_matrix.reset_index(drop=True)
   
    return df_matrix
   
   
def p_matrix_elem(df_matrix, elem, axis):
   
    if axis ==1:
       
        remove = df_matrix.columns.to_list()
        for el in elem:
            remove.remove(el)
        remove.remove('plant_sp')
       
        P_matrix_df =df_matrix.drop(remove, axis=1)
       
        #P_matrix_df = df_matrix[df_matrix.columns.intersection(['plant_sp',elem])]
       
    if axis == 0:
       
        remove = df_matrix['plant_sp'].to_list()
        for el in elem:
            remove.remove(el)
        P_matrix_df =df_matrix[~df_matrix['plant_sp'].isin(remove)]
       
        #P_matrix_df = df_matrix[~df_matrix['plant_sp'].isin(remove)]

    P_matrix    = P_matrix_df.values[:, 1:].astype(float)
    P_matrix_t  = np.transpose(P_matrix_df.values[:, 1:].astype(float))
   
    return  P_matrix, P_matrix_t


def layer_layout(G, layout='spring', tree= False, k=2, seed=1234):
    if tree == True:
        pos = nx.nx_agraph.graphviz_layout(G,prog="twopi")
    if len(G.nodes) == 5:
        pos_arr = [np.array([1., 0.]), np.array([0.30901695, 0.95105657]), np.array([-0.80901706,  0.58778526]), np.array([-0.809017  , -0.58778532]), np.array([ 0.3090171 , -0.95105651])]
        pos = {np.array(G.nodes)[i]: pos_arr[i] for i in range(len(G.nodes))}
    elif layout == 'spring':
        pos = nx.spring_layout(G, weight = "weight", seed=seed, k=k)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    else:
        pos = nx.shell_layout(G)
    return pos



#rcParams["text.usetex"] = True

#plt.ylabel(r"\textrm{\textbf{\textit{habvrflke}}}")
# def item_words(strg, line_sep):
   
#     if line_sep==True:

#         words = strg.split(" ", 1)
#         if strg == 'Olea europaea var. sylvestris':
#             strg = 'Olea Europaea\n var. Sylvestris'
#         elif len(words)> 2:
#             strg = [word.capitalize()+ ' \n ' for word in words]  
#             #strg = [word.capitalize() for word in words]  
#         elif any([len(word)>12 for word in words]) and len(words)>1:
#             strg = words[0].capitalize() + ' \n ' + words[1]
#         elif len(words)>1:
#             strg = words[0].capitalize() +' '+ words[1] + '\n '
#             #strg = words[0].capitalize() +' '+ words[1].capitalize() + '\n '
#         else:
#             strg = words[0].capitalize()
           
#     else:
       
#         words = strg.split(" ", 1)
#         if strg == 'Olea europaea var. sylvestris':
#             strg = 'Olea Europaea var. sylvestris'
#         elif len(words)> 2:
#             strg = [word.capitalize()+ ' ' for word in words]  
#         elif any([len(word)>11 for word in words]) and len(words)>1:
#             strg = words[0].capitalize() + ' ' + words[1]
#         elif len(words)>1:
#             strg = words[0].capitalize() +' '+ words[1]
#         else:
#             strg = words[0].capitalize()

       
#     return strg
   
#strg_it = r'%s $\mathit{%s}$' % (words[0].capitalize(), words[1:][0])


# def item_words(strg, line_sep):
   
#     if line_sep==True:

#         words = strg.split(" ", 1)
#         if strg == 'Olea europaea var. sylvestris':
#             strg = r'%s $\mathit{%s}$' % ('Olea Europaea\n', 'var. sylvestris') 
#         elif len(words)> 2:
#             strg = r'%s $\mathit{%s}$' % (words[0].capitalize(), words[1:][0] +'\n ') 
#             #strg = [word.capitalize() for word in words]  
#         elif any([len(word)>12 for word in words]) and len(words)>1:
#             strg = words[0].capitalize() + ' \n ' + rf'$\mathit{words[1]}$'
#         elif len(words)>1:
#             strg = r'%s $\mathit{%s}$' % (words[0].capitalize(), words[1:][0]) 
#             strg = strg +'\n '
#             print('here')
#             #strg = words[0].capitalize() +' '+ words[1].capitalize() + '\n '
#         else:
#             strg = words[0].capitalize()
           
#     else:
       
#         words = strg.split(" ", 1)
#         if strg in [ 'Olea europaea var. sylvestris', 'Olea Europaea Var. sylvestris','Olea europaea Var. sylvestris','Olea Europaea Var. sylvestris','Olea Europaea Var. Sylvestris']:
#             strg = r'%s $\mathit{%s}$' % ('Olea Europaea', 'var. sylvestris')
#         elif len(words)> 2:
#             strg = r'%s $\mathit{%s}$' % (words[0].capitalize(), words[1:][0]) 
#             #strg = [word.capitalize() for word in words]  
#         elif any([len(word)>12 for word in words]) and len(words)>1:
#             strg = words[0].capitalize() + ' \n ' + rf'$\mathit{words[1]}$'
#         elif len(words)>1:
#             strg = r'%s $\mathit{%s}$' % (words[0].capitalize(), words[1:][0]) 
#             #strg = words[0].capitalize() +' '+ words[1].capitalize() + '\n '
#         else:
#             strg = words[0].capitalize()

       
#     return strg


def item_words(strg, line_sep):
   
    if line_sep==True:

        words = strg.split(" ", 1)
        if strg == 'Olea europaea var. sylvestris':
            strg = r'%s $\mathit{%s}$' % ('','Olea europaea') + ' \n ' 
        elif len(words)> 2:
            strg = r'%s $\mathit{%s}$' % ('', words[0].capitalize()+'\,'+ words[1:][0] +'\n ') 
            #strg = [word.capitalize() for word in words]  
        elif any([len(word)>12 for word in words]) and len(words)>1:
            strg = rf'$\mathit{words[0].capitalize()}$' + ' \n ' + rf'$\mathit{words[1]}$'
        elif len(words)>1:
            strg = r'%s $\mathit{%s}$' % ('', words[0].capitalize() +'\,'+ words[1:][0]) 
            strg = strg +'\n '
            print('here')
            #strg = words[0].capitalize() +' '+ words[1].capitalize() + '\n '
        else:
            strg =r'%s $\mathit{%s}$' % ('', words[0].capitalize()) 
           
    else:
       
        words = strg.split(" ", 1)
        if strg in [ 'Olea europaea var. sylvestris', 'Olea Europaea Var. sylvestris','Olea europaea Var. sylvestris','Olea Europaea Var. sylvestris','Olea Europaea Var. Sylvestris']:
            strg = r'%s $\mathit{%s}$' % ('','Olea\ europaea') 
        elif len(words)> 2:
            strg = r'%s $\mathit{%s}$' % ('', words[0].capitalize()+'\,' +words[1:][0]) 
            print('aqui')
            #strg = [word.capitalize() for word in words]  
        elif any([len(word)>12 for word in words]) and len(words)>1:
            strg = rf'$\mathit{words[0].capitalize()}$' + ' \n ' + rf'$\mathit{words[1]}$'
            print('aqui2')
        elif len(words)>1:
            strg = r'%s $\mathit{%s}$' % ('', words[0].capitalize()+'\,'+ words[1:][0]) 
            #strg = words[0].capitalize() +' '+ words[1].capitalize() + '\n '
            print('aqui3')
        else:
            strg = r'%s $\mathit{%s}$' % ('', words[0].capitalize()) 

       
    return strg



def titles(list_, line_sep=True):
   
    titles_list =[]
    titles_list = [item_words(i,line_sep) for i in list_]
   
    return titles_list
       


def plot_shp(length, ncols):
   
    plot_str =string.ascii_uppercase[:(length+1)]
    plot_shape = '\n'.join(plot_str[i:i + ncols] for i in range(0, len(plot_str), ncols))
    plot_str = plot_str[:-1]
    plot_shape = plot_shape[:-1] +'.'
   
    last  = plot_shape.split('\n')[-1]
    nrows = math.ceil((length+1)/ncols)
   
    if len(last)<ncols:
        plot_shape = plot_shape +'.'*(ncols - len(last))
   
    return plot_str, plot_shape, nrows




df_matrix_s = sort_df_matrix(df_matrix, function_sorted, species_sorted)



'''BIPARTITE'''

size_label =25
#lb = {i: n for i, n in enumerate(labels_f)}
fontsize=22

# v_min and v_max
v_min = int(np.min(P_t_times_P))
v_max = int(np.max(P_t_times_P))+1

G = nx.Graph()
G.add_nodes_from(labels_p)
G.add_nodes_from(labels_f)

for i, fs in enumerate(labels_f):
    #print(i)
    for j, ps in enumerate(labels_p):
       
        G.add_edge(fs, ps, weight=P_matrix[j][i])

def bipartite_pos(groupL, groupR, ratioL=4/3, ratioR = 4/3):
    pos = {}
    pos_labels = {}
    gapR = ratioR/ len(groupR)
    ycord = ratioL / 2 - gapR/2

    for r in groupR:
        #print(ycord)
        pos[r] = (1, ycord)
        pos_labels[r] = (1.1, ycord)
        ycord -= gapR
       
   
    gapL = ratioL/ len(groupL)
    ycord = ratioL / 2 - gapL/2
    for l in groupL:
        #print(ycord)

        pos[l] = (-1, ycord)
        pos_labels[l] = (-1.1, ycord)

        ycord -= gapL

    return pos, pos_labels

pos, pos_labels = bipartite_pos(labels_f, labels_p)

#Total
   
fig, ax = plt.subplots(figsize=(12,8), dpi=100)    
width =np.array([d["weight"] for u,v,d in G.edges(data=True)])
#print(width.max())
#e = nx.draw_networkx_edges(G, pos=pos, edge_color = np.array([mpl.cm.get_cmap('magma')(.2)]), width=(width)*2, ax=ax)
e = nx.draw_networkx_edges(G, pos=pos, edge_color = np.array([(100/256,85/256,160/256,1.0)]), width=(width)*2, ax=ax)
node_size  = np.array([G.degree(weight='weight')[node] for node in G.nodes])*50
#nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color=np.array([mpl.cm.get_cmap('magma')(.2)]), node_shape="o", alpha=0.8, linewidths=4, ax=ax)
nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color=np.array([(100/256,85/256,160/256,1.0)]), node_shape="o", alpha=0.8, linewidths=4, ax=ax)
#nx.draw_networkx_labels(G, pos=pos, ax=axs[item], font_size=fontsize*1.8, font_color='w')  
nx.draw_networkx_labels(G, pos={k: item for k,item in pos_labels.items() if k in labels_p}, labels={label:label for label in labels_p} , ax=ax, font_size=fontsize*1.2, horizontalalignment = "left")  
nx.draw_networkx_labels(G, pos={k: item for k,item in pos_labels.items() if k in labels_f}, labels={label:label for label in labels_f} , ax=ax, font_size=fontsize*1.2, horizontalalignment = "right")  
plt.axis('off')
#ax.set_title('Multifunctional',{'fontsize':fontsize, 'fontweight':"bold"}, pad=-20)
xlim = np.array(list(ax.get_xlim()))*1.4 #make plot lims wider in the x direction not to crop nodes
ylim = np.array(list(ax.get_ylim()))*0.9 #same in y direction
ax.set_xlim(xlim);
ax.set_ylim(ylim);
degree_sp = np.array([G.degree(weight='weight')[node] for node in labels_p])
degree_sp = degree_sp / np.sum(np.array([G.degree(weight='weight')[node] for node in labels_p]) )
degree_fn = np.array([G.degree(weight='weight')[node] for node in labels_f])
degree_fn = degree_fn / np.sum(np.array([G.degree(weight='weight')[node] for node in labels_f]) )
######plant.savefig(plt_dir + '/plant_function_netwotks.pdf', bbox_inches='tight', dpi=100)
#plant.savefig(plt_dir + '/plant_function_netwotks.png', bbox_inches='tight', dpi=100)



#%%

'''PLANTAS'''


loop_size_x = .6
loop_size_y = .6


size_label =25
fontsize=22


class SelfLoop():
    def __init__(self, v_scale=0.25, h_scale=0.25, nodesize=300):
        self.v_scale = v_scale
        self.h_scale = h_scale
        self.nodesize = nodesize

    def selfloopstyle(self, posA, posB, *args, **kwargs):
        from matplotlib.path import Path

        selfloop_ht = 0.005 * self.nodesize

        data_loc = ax.transData.inverted().transform(posA)
        v_shift = self.v_scale * selfloop_ht
        #h_shift = v_shift * self.h_scale
        h_shift = self.h_scale * selfloop_ht
        path = [
            data_loc + np.asarray([0, v_shift]),
            data_loc + np.asarray([h_shift, v_shift]),
            data_loc + np.asarray([h_shift, 0]),
            data_loc,
            data_loc + np.asarray([-h_shift, 0]),
            data_loc + np.asarray([-h_shift, v_shift]),
            data_loc + np.asarray([0, v_shift]),
        ]

        ret = Path(ax.transData.transform(path), [1, 4, 4, 4, 4, 4, 4])

        return ret
   
    def style(self):
        return self.selfloopstyle
   
G_T = nx.from_numpy_array(P_t_times_P, parallel_edges=False, create_using=None)  
G_T = nx.relabel_nodes(G_T, {np.array(G_T.nodes)[i]: labels_f[i] for i in range(len(G_T.nodes))} , copy=False)
pos = layer_layout(G_T, layout='shell')
dic_labels = {}
coord_scale = 1.35
for elem in pos:
    dic_labels[elem] = pos[elem] * coord_scale
   
fig, ax = plt.subplots(figsize=(12,8), dpi=100)    
width =np.array([d["weight"] for u,v,d in G_T.edges(data=True)])
#print(width.max())
#e = nx.draw_networkx_edges(G_T, pos=pos, edge_color = np.array([mpl.cm.get_cmap('magma')(.2)]), width=(width)*2, ax=ax)
e = nx.draw_networkx_edges(G_T, pos=pos, edge_color = np.array([(100/256,85/256,160/256,1.0)]), width=(width)*2, ax=ax)
for self_loop_fap in ax.patches:
    self_loop_fap._connector = SelfLoop(loop_size_x,loop_size_y).style()
    #print("a")
ax.autoscale_view()
node_size  = np.array([G_T.degree(weight='weight')[node] for node in G_T.nodes])*50
#nx.draw_networkx_nodes(G_T, pos=pos, node_size=node_size,node_color=np.array([mpl.cm.get_cmap('magma')(.2)]), node_shape="o", alpha=0.8, linewidths=4, ax=ax)
nx.draw_networkx_nodes(G_T, pos=pos, node_size=node_size,node_color=np.array([(100/256,85/256,160/256,1.0)]), node_shape="o", alpha=0.8, linewidths=4, ax=ax)
#nx.draw_networkx_labels(G, pos=pos, ax=axs[item], font_size=fontsize*1.8, font_color='w')  
nx.draw_networkx_labels(G_T, pos=dic_labels, ax=ax, font_size=fontsize*1.6)  
plt.axis('off')
ax.set_title('Multifunctional',{'fontsize':fontsize}, pad=-20)
xlim = np.array(list(ax.get_xlim()))*1.3 #make plot lims wider in the x direction not to crop nodes
ylim = np.array(list(ax.get_ylim()))*1.3 #same in y direction
ax.set_xlim(xlim);
ax.set_ylim(ylim);
 
#plant.savefig("example.pdf", dpi=100)
#%%
plt.style.context('science'+'nature')
loop_size_x = .2
loop_size_y = .2
titles_list = titles(species_sorted)  
ncols=4
plot_str, plot_shape, nrows= plot_shp(len(species),ncols)
stregnth_sp=[]
     
fig, axs = plt.subplot_mosaic(plot_shape, figsize=(12*ncols,8*nrows), dpi=100)
fig.subplots_adjust(hspace = 0.3, wspace=0)  
for i in range(len_species):  
    item = plot_str[i]
    p, p_t= p_matrix_elem(df_matrix_s, elem = [species_sorted[i]] , axis=0)
    A =np.matmul(p_t,p)  
    #print(A)      
    G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
    G = nx.relabel_nodes(G, {np.array(G.nodes)[i]: labels_f[i] for i in range(len(G.nodes))} , copy=False)
    #G.remove_edges_from(nx.selfloop_edges(G))
    node_color  = np.array([G.degree(weight='weight')[node]/G_T.degree(weight='weight')[node] for node in G.nodes])
    stregnth_sp.append(node_color)
    #print(np.max(node_color))
    node_size  = np.array([G.degree(weight='weight')[node] for node in G.nodes])*400
    edge_color = [dict(nx.get_edge_attributes(G, 'weight'))[edge]/dict(nx.get_edge_attributes(G_T, 'weight'))[edge] for edge in G.edges()]
    #print(max(edge_color))
    pos = layer_layout(G, layout='shell')
    dic_labels = {}
    coord_scale = 1.29
    for elem in pos:
        dic_labels[elem] = pos[elem] * coord_scale
    #pos =dict((lb[key], value) for (key, value) in pos.items())
    #weight = np.array(list(nx.get_edge_attributes(G,"weight").values())).flatten()

    width =np.array([d["weight"] for u,v,d in G.edges(data=True)])
    #edges = nx.draw_networkx_edges(G, pos=pos, edge_color = 'cadetblue', edge_cmap=plt.cm.viridis, edge_vmax=.1, edge_vmin=vmin, width=(width/np.max(width))*6, ax=axs[item])
    e = nx.draw_networkx_edges(G, pos=pos, edge_color = edge_color, edge_cmap=plt.cm.magma, edge_vmax=.8, edge_vmin=0, width=(width)*8, ax=axs[item])
   
    for self_loop_fap in axs[item].patches:
        self_loop_fap._connector = SelfLoop(loop_size_x,loop_size_y).style()
        #print("a")
    axs[item].autoscale_view()

    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color=node_color, node_shape="o", alpha=0.6, linewidths=4, ax=axs[item], vmin=0, vmax=1, cmap=plt.cm.magma    )
    #nx.draw_networkx_labels(G, pos=pos, ax=axs[item], font_size=fontsize*1.8, font_color='w')  
    nx.draw_networkx_labels(G, pos=dic_labels, ax=axs[item], font_size=fontsize*1.6)  
    plt.axis('off')
    xlim =ax.get_xlim()
    ylim =ax.get_ylim()
    axs[item].set_title(titles_list[i],{'fontsize':size_label*1.6}, pad=-15)
    xlim = np.array(list(axs[item].get_xlim()))*1.3 #make plot lims wider in the x direction not to crop nodes
    ylim = np.array(list(axs[item].get_ylim()))*1.3 #same in y direction
    axs[item].set_xlim(xlim)
    axs[item].set_ylim(ylim)
    axs[item].axis('off')



cax = fig.add_axes([0.4, 0.225, 0.25, 0.025])
cb = mpl.colorbar.ColorbarBase(cax, orientation='horizontal',
                               cmap=plt.cm.magma  ,
                               norm=mpl.colors.Normalize(0,.8),  # vmax and vmin
                               ticks=np.arange(0, 1.2,.2))
cb.ax.tick_params(labelsize=size_label*1.6,width = 6, length = 12)
cb.set_label('Relative contribution of species ' + rf'$i$' + ' in function ' + rf'$\alpha$', labelpad=30, fontsize = size_label*1.6)
 
#plant.savefig(plt_dir + '/function_function_netwotks.png', bbox_inches='tight', dpi=100)
######plant.savefig(plt_dir + '/function_function_netwotks_try.pdf', bbox_inches='tight', dpi=100)

#%%


loop_size_x = .2
loop_size_y = .2
stregnth_sp=[]
for i in range(len_species):
   
    fig, axs = plt.subplots(figsize=(12,8), dpi=100)
    p, p_t= p_matrix_elem(df_matrix_s, elem = [species_sorted[i]] , axis=0)
    A =np.matmul(p_t,p)
    #print(np.max(A))      
    G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
    G = nx.relabel_nodes(G, {np.array(G.nodes)[i]: labels_f[i] for i in range(len(G.nodes))} , copy=False)
    node_color  = np.array([G.degree(weight='weight')[node]/G_T.degree(weight='weight')[node] for node in G.nodes])
    stregnth_sp.append(node_color)
    node_size  = np.array([G.degree(weight='weight')[node] for node in G.nodes])*400
    edge_color = [dict(nx.get_edge_attributes(G, 'weight'))[edge]/dict(nx.get_edge_attributes(G_T, 'weight'))[edge] for edge in G.edges()]
    pos = layer_layout(G, layout='shell')
    dic_labels = {}
    coord_scale = 1.29
    for elem in pos:
        dic_labels[elem] = pos[elem] * coord_scale

    width =np.array([d["weight"] for u,v,d in G.edges(data=True)])
    #print(np.max(width))
    e = nx.draw_networkx_edges(G, pos=pos, edge_color = edge_color, edge_cmap=plt.cm.magma, edge_vmax=.8, edge_vmin=0, width=(width)*8, ax=axs)
   
    for self_loop_fap in axs.patches:
        self_loop_fap._connector = SelfLoop(loop_size_x,loop_size_y).style()
    axs.autoscale_view()
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color=node_color, node_shape="o", alpha=0.6, linewidths=4, ax=axs, vmin=0, vmax=1, cmap=plt.cm.magma    )
    #nx.draw_networkx_labels(G, pos=pos, ax=axs[item], font_size=fontsize*1.8, font_color='w')  
    nx.draw_networkx_labels(G, pos=dic_labels, ax=axs, font_size=fontsize)  
    plt.axis('off')
    xlim =axs.get_xlim()
    ylim =axs.get_ylim()
    axs.set_title(titles_list[i],{'fontsize':size_label}, pad=0)
    xlim = np.array(list(axs.get_xlim()))*1.3 #make plot lims wider in the x direction not to crop nodes
    ylim = np.array(list(axs.get_ylim()))*1.3 #same in y direction
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.axis('off')
    
    #plant.savefig(plt_dir + f'/function_function_netwotks_{species_sorted[i]}.png', bbox_inches='tight',  dpi=100 )
    ######plant.savefig(plt_dir + f'/function_function_netwotks_{species_sorted[i]}.jpeg', bbox_inches='tight',  dpi=100 )
#colorbar
plt.figure(figsize=(1,12), dpi=100)
#img = plt.imshow(a,  cmap = 'viridis', vmin=v_min , vmax=v_max)
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cb = mpl.colorbar.ColorbarBase(cax, orientation='vertical',
                               cmap=plt.cm.magma  ,
                               norm=mpl.colors.Normalize(0,.8),  # vmax and vmin
                               ticks=np.arange(0, 1.2,.2))
cb.ax.tick_params(labelsize=size_label*1.6,width = 3, length = 8)
cb.set_label('Relative contribution of \n species ' + rf'$i$' + ' in function ' + rf'$\alpha$', labelpad=100, fontsize = size_label*1.6, rotation=-90)
 
#plant.savefig(plt_dir + '/function_function_netwotks_cbar.png', bbox_inches='tight', dpi=100  )
######plant.savefig(plt_dir + '/function_function_netwotks_cbar.pdf', bbox_inches='tight', dpi=100  )
#%%




'''FUNCIONES'''

loop_size_x = .2
loop_size_y = .2

# v_min and v_max
v_min = int(np.min(P_times_P_t))
v_max = int(np.max(P_times_P_t))+1

G_T = nx.from_numpy_array(P_times_P_t, parallel_edges=False, create_using=None)  
G_T = nx.relabel_nodes(G_T, {np.array(G_T.nodes)[i]: labels_p[i] for i in range(len(G_T.nodes))} , copy=False)
pos = layer_layout(G_T, layout='spring', seed=1234, k = 2.1)
dic_labels = {}
coord_scale = 1.13
for elem in pos:
    dic_labels[elem] = pos[elem] * coord_scale
fig, ax = plt.subplots(figsize=(12,8), dpi=100)    
width =np.array([d["weight"] for u,v,d in G_T.edges(data=True)])
#print(width.max())
#e = nx.draw_networkx_edges(G_T, pos=pos, edge_color = np.array([mpl.cm.get_cmap('magma')(.2)]), width=(width)*2, ax=ax)
e = nx.draw_networkx_edges(G_T, pos=pos, edge_color = np.array([(100/256,85/256,160/256,1.0)]), width=(width)*2, ax=ax)
for self_loop_fap in ax.patches:
    self_loop_fap._connector = SelfLoop(loop_size_x,loop_size_y).style()
    #print("a")
ax.autoscale_view()
node_size  = np.array([G_T.degree(weight='weight')[node] for node in G_T.nodes])*50
#nx.draw_networkx_nodes(G_T, pos=pos, node_size=node_size,node_color=np.array([mpl.cm.get_cmap('magma')(.2)]), node_shape="o", alpha=0.8, linewidths=4, ax=ax)
nx.draw_networkx_nodes(G_T, pos=pos, node_size=node_size,node_color=np.array([(100/256,85/256,160/256,1.0)]), node_shape="o", alpha=0.8, linewidths=4, ax=ax)
#nx.draw_networkx_labels(G, pos=pos, ax=axs[item], font_size=fontsize*1.8, font_color='w')  
nx.draw_networkx_labels(G_T, pos=dic_labels, ax=ax, font_size=fontsize*1)  
plt.axis('off')
ax.set_title('Multifunctional',{'fontsize':size_label}, pad=-15)
xlim = np.array(list(ax.get_xlim()))*1.1 #make plot lims wider in the x direction not to crop nodes
ylim = np.array(list(ax.get_ylim()))*1.2 #same in y direction
ax.set_xlim(xlim);
ax.set_ylim(ylim);


stregnth_fn = []
titles_list = [function.capitalize() for function in function_sorted]
ncols=3
plot_str, plot_shape, nrows= plot_shp(len(functions),3)

fig, axs = plt.subplot_mosaic(plot_shape, figsize=(12*ncols,8*ncols), dpi=100)
fig.subplots_adjust(hspace = 0.1, wspace=0.1)  
for i in range(len_functions):  
    item = plot_str[i]
    p, p_t= p_matrix_elem(df_matrix_s, elem = [function_sorted[i]] , axis=1)
    A = np.matmul(p,p_t)
    #print(np.max(A))      
    G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
    G = nx.relabel_nodes(G, {np.array(G.nodes)[i]: labels_p[i] for i in range(len(G.nodes))} , copy=False)
    node_color  = np.array([G.degree(weight='weight')[node]/G_T.degree(weight='weight')[node] for node in G.nodes])
    stregnth_fn.append(node_color)
    node_size  = np.array([G.degree(weight='weight')[node] for node in G.nodes])*400
    edge_color = [dict(nx.get_edge_attributes(G, 'weight'))[edge]/dict(nx.get_edge_attributes(G_T, 'weight'))[edge] for edge in G.edges()]
    pos = layer_layout(G, layout='spring', k=2.2, seed=13454)
    dic_labels = {}
    coord_scale = 1.13
    for elem in pos:
        dic_labels[elem] = pos[elem] * coord_scale

    width =np.array([d["weight"] for u,v,d in G.edges(data=True)])
    #print(np.max(width))
    e = nx.draw_networkx_edges(G, pos=pos, edge_color = edge_color, edge_cmap=plt.cm.magma, edge_vmax=.8, edge_vmin=0, width=(width)*8, ax=axs[item])
   
    for self_loop_fap in axs[item].patches:
        self_loop_fap._connector = SelfLoop(loop_size_x,loop_size_y).style()
    axs[item].autoscale_view()
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color=node_color, node_shape="o", alpha=0.6, linewidths=4, ax=axs[item], vmin=0, vmax=1, cmap=plt.cm.magma    )
    #nx.draw_networkx_labels(G, pos=pos, ax=axs[item], font_size=fontsize*1.8, font_color='w')  
    nx.draw_networkx_labels(G, pos=dic_labels, ax=axs[item], font_size=fontsize)  
    plt.axis('off')
    xlim =ax.get_xlim()
    ylim =ax.get_ylim()
    axs[item].set_title(titles_list[i],{'fontsize':size_label}, pad=-15)
    xlim = np.array(list(axs[item].get_xlim()))*1.1 #make plot lims wider in the x direction not to crop nodes
    ylim = np.array(list(axs[item].get_ylim()))*1.3 #same in y direction
    axs[item].set_xlim(xlim)
    axs[item].set_ylim(ylim)
    axs[item].axis('off')

cax = fig.add_axes([0.38, 0.35, 0.25, 0.025])
cb = mpl.colorbar.ColorbarBase(cax, orientation='horizontal',
                               cmap=plt.cm.magma  ,
                               norm=mpl.colors.Normalize(0,1),  # vmax and vmin
                               ticks=np.arange(0, 1.2,.2))
cb.ax.tick_params(labelsize=size_label,width = 6, length = 12)
cb.set_label('Relative contribution of function '+ rf'$\alpha$' +' to the strenght of species ' + rf'$i$', labelpad=30, fontsize = size_label)
#plant.savefig(plt_dir + '/plant_plant_netwotks.png', bbox_inches='tight', dpi=100)
######plant.savefig(plt_dir + '/plant_plant_netwotks.pdf', bbox_inches='tight',dpi=100  )


loop_size_x = .2
loop_size_y = .2

stregnth_fn = []

for i in range(len_functions):
   
    fig, axs = plt.subplots(figsize=(12,8), dpi=100)
    p, p_t= p_matrix_elem(df_matrix_s, elem = [function_sorted[i]] , axis=1)
    A = np.matmul(p,p_t)
    #print(np.max(A))      
    G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
    G = nx.relabel_nodes(G, {np.array(G.nodes)[i]: labels_p[i] for i in range(len(G.nodes))} , copy=False)
    node_color  = np.array([G.degree(weight='weight')[node]/G_T.degree(weight='weight')[node] for node in G.nodes])
    #print(np.max(node_color))    
    stregnth_fn.append(node_color)
    node_size  = np.array([G.degree(weight='weight')[node] for node in G.nodes])*400
    edge_color = [dict(nx.get_edge_attributes(G, 'weight'))[edge]/dict(nx.get_edge_attributes(G_T, 'weight'))[edge] for edge in G.edges()]
    pos = layer_layout(G, layout='spring', k=2.2, seed=13454)
    dic_labels = {}
    coord_scale = 1.13
    for elem in pos:
        dic_labels[elem] = pos[elem] * coord_scale

    width =np.array([d["weight"] for u,v,d in G.edges(data=True)])
    #print(np.max(width))
    e = nx.draw_networkx_edges(G, pos=pos, edge_color = edge_color, edge_cmap=plt.cm.magma, edge_vmax=.8, edge_vmin=0, width=(width)*8, ax=axs)
   
    for self_loop_fap in axs.patches:
        self_loop_fap._connector = SelfLoop(loop_size_x,loop_size_y).style()
    axs.autoscale_view()
    nx.draw_networkx_nodes(G, pos=pos, node_size=node_size,node_color=node_color, node_shape="o", alpha=0.6, linewidths=4, ax=axs, vmin=0, vmax=1, cmap=plt.cm.magma    )
    #nx.draw_networkx_labels(G, pos=pos, ax=axs[item], font_size=fontsize*1.8, font_color='w')  
    nx.draw_networkx_labels(G, pos=dic_labels, ax=axs, font_size=fontsize)  
    plt.axis('off')
    xlim =axs.get_xlim()
    ylim =axs.get_ylim()
    axs.set_title(titles_list[i],{'fontsize':size_label}, pad=-100)
    xlim = np.array(list(axs.get_xlim()))*1.2 #make plot lims wider in the x direction not to crop nodes
    ylim = np.array(list(axs.get_ylim()))*1.3 #same in y direction
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.axis('off')
     
    #plant.savefig(plt_dir + f'/plant_plant_netwotks_{function_sorted[i]}.png', bbox_inches='tight' , dpi=100 )
    ######plant.savefig(plt_dir + f'/plant_plant_netwotks_{function_sorted[i]}.pdf', bbox_inches='tight' , dpi=100  )
   
#colorbar
plt.figure(figsize=(1,12), dpi=100)
#img = plt.imshow(a,  cmap = 'viridis', vmin=v_min , vmax=v_max)
plt.gca().set_visible(False)
cax = plt.axes([0.1, 0.2, 0.8, 0.6])
cb = mpl.colorbar.ColorbarBase(cax, orientation='vertical',
                               cmap=plt.cm.magma  ,
                               norm=mpl.colors.Normalize(0,.8),  # vmax and vmin
                               ticks=np.arange(0, 1.2,.2))
cb.ax.tick_params(labelsize=size_label*1.6,width = 3, length = 8)
cb.set_label('Relative contribution of\n function ' + rf'$\alpha$' + ' to the strength' +'\n'+'of species ' + rf'$i$', labelpad=150, fontsize = size_label*1.6, rotation=-90) 
#plant.savefig(plt_dir + '/plant_plant_netwotks_cbar.png', bbox_inches='tight' , dpi=100  )
######plant.savefig(plt_dir + '/plant_plant_netwotks_cbar.png', bbox_inches='tight' , dpi=100 )



#%%

'''Ranks'''

def rank(strength, elem_order, axis =1):
   
    sums = np.sum(strength, axis=axis)
    sorter = np.argsort(sums)[::-1]
    species_rank = np.array(elem_order)[sorter]
    strength_rank = np.array(strength)[sorter]
   
    return sums, strength_rank,species_rank





def func_color(att_list,cmp = plt.cm.rainbow):
     
    '''Returns a colormap dic based on the palette 'plt.cm.rainbow',
       discretized into the number of (different) colors
       found in the attribute.
    
    Parameters
    ----------
    att_list: attributes list
    
    Returns
    ----------
    cmp_edges: cmap
        Cmap of edges.
    
    colors: array
        array of edges attributs in G.edges order.
        
    func_to_color: dict
        Dict object attr: color.
        
    '''
     
    
    #If data is default dataset or has the same ecological functions uses pre-established colors
    if Counter(att_list) == Counter(['decomposition','fungal pathogenicity','herbivory','nutrient uptake','pollination','seed dispersal']):
        func_to_color= {'herbivory': 'tab:purple',
         'plant': 'tab:green',
         'pollination': 'tab:orange',
         'decomposition': 'tab:brown',
         'fungal pathogenicity': 'tab:olive',
         'seed dispersal': 'tab:blue',
         'nutrient uptake': 'tab:pink'}
    
    # If not, used the input colormap
    else:
        colors = dsp.get_colors(len(att_list), pastel_factor=0.7)
        func_to_color = {att_list[item]: colors[item] for item in range(len(att_list))}
    
    return func_to_color
###Total
dic_colors_fn = func_color(function_sorted,cmp = plt.cm.rainbow)
dic_colors_sp = func_color(species_sorted,cmp = plt.cm.turbo)

fig, axs = plt.subplots(figsize=(7,8), dpi=100)  
fontsize=20
position = np.arange(1,len_species+1)
max_len = -1
left = np.zeros(len_species)
position = np.arange(14,14+len_functions+1)
position = np.arange(0,0+len_functions+1)
for ele in labels_f:
    if len(ele) > max_len:
        max_len = len(ele)
for i in range(len_functions):
    axs.text((max_len/40)*(-1), position[i]+ 0.1, labels_f[i], fontsize =fontsize, horizontalalignment = 'left')
    axs.barh(position[i], degree_fn[i],left=0, color = dic_colors_fn[function_sorted[i]])
axs.invert_yaxis()
for pos in ['right', 'top', 'left']:
    axs.spines[pos].set_visible(False)
# for axis in ['bottom']:
#     axs.spines[axis].set_linewidth(3)
axs.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
axs.set_xticks(np.arange(-.1,.5,0.1))
axs.set_xticklabels(["", '0', '0.1', '0.2', '.3', '.4'])
axs.set_yticks([])
#axs.set_ylim(20,0)
axs.set_xlabel('Generalist-specialist function ranking', fontsize = fontsize)
#plant.savefig(plt_dir + '/rank_fn_figure2.pdf', bbox_inches='tight' ,  dpi=100)


fig, axs = plt.subplots(figsize=(7,8), dpi=100)  
fontsize=20
max_len = -1
left = np.zeros(len_species)
position = np.arange(1,len_species+1)
for ele in labels_p:
    if len(ele) > max_len:
        max_len = len(ele)
for i in range(len_species):
    axs.text((max_len/160)*(-1), position[i]+ 0.2, labels_p[i], fontsize =fontsize, horizontalalignment = 'left')
    axs.barh(position[i], degree_sp[i],left=0, color = dic_colors_sp[species_sorted[i]])
axs.invert_yaxis()
for pos in ['right', 'top', 'left']:
    axs.spines[pos].set_visible(False)
# for axis in ['bottom']:
#     axs.spines[axis].set_linewidth(3)
axs.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
axs.set_xticks(np.arange(-.05,.15,0.05))
axs.set_xticklabels(["", '0', '0.05', '0.1'])
axs.set_yticks([])
axs.set_xlabel('Generalist-specialist species ranking', fontsize = fontsize)

 
#plant.savefig(plt_dir + '/rank_sp_figure2.pdf', bbox_inches='tight' ,  dpi=100)

#%%
##plantas

sums_sp_rank, stregnth_sp_rank, labels_sp_rank = rank(stregnth_sp, species_sorted)

labels_sp_rank_tit = titles(labels_sp_rank, line_sep=False)

fig, axs = plt.subplots(figsize=(12,8), dpi=100)  
fontsize=20
position = np.arange(1,len_species+1)
max_len = -1
left = np.zeros(len_species)
label_width = np.zeros(len_species)
position = np.arange(1,len_species+1)
for ele in labels_sp_rank_tit:
    if len(ele) > max_len:
        max_len = len(ele)
for i in range(len_species):
    axs.text((max_len/25)*(-1), position[i]+ 0.3, labels_sp_rank_tit[i], fontsize =fontsize, horizontalalignment = 'left')
    label_width = np.zeros(len_species)
    for j in range(len(function_sorted)):
        axs.barh(position[i], stregnth_sp_rank[i][j],left=label_width, color = dic_colors_fn[function_sorted[j]])
        label_width += stregnth_sp_rank[i][j]
axs.invert_yaxis()
for pos in ['right', 'top', 'left']:
    axs.spines[pos].set_visible(False)
# for axis in ['bottom']:
#     axs.spines[axis].set_linewidth(3)
axs.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
axs.set_xticks(np.arange(-1.2,1.5,0.4))
axs.set_xticklabels(["","","", '0', '0.4', '0.8', '1'])
axs.set_yticks([])
axs.set_xlabel('Multifunctional species keystoness', fontsize = fontsize)
#plant.savefig(plt_dir + '/rank_figure3.pdf', bbox_inches='tight' ,  dpi=100)



labels = [function.capitalize() for function in function_sorted]
handles = [plt.Rectangle((0,0),1,1, color=dic_colors_fn[fn]) for fn in function_sorted]
############ARREGLAR TITLES FINCTION SORTED NO FUNCIONA LA FUNCION
fig, ax = plt.subplots(figsize=(5,2), dpi=100)
ax.legend(handles =handles,labels=labels,fontsize= fontsize,loc= 'center', frameon=False, ncol= 2)
plt.axis('off')
 
#plant.savefig(plt_dir + '/rank_figure3_legend.pdf', bbox_inches='tight',  dpi=100 )

#%%



#Funciones

sums_fn_rank, stregnth_fn_rank, labels_fn_rank = rank(stregnth_fn, function_sorted)
labels_fn_rank = [function.capitalize() for function in labels_fn_rank]
fig, axs = plt.subplots(figsize=(12.3,8), dpi=100)  
fontsize=20
position = np.arange(1,len_functions+1)
max_len = -1
left = np.zeros(len_functions)
label_width = np.zeros(len_functions)
position = np.arange(1,len_functions+1)
for ele in labels_fn_rank:
    if len(ele) > max_len:
        max_len = len(ele)
for i in range(len_functions):
    axs.text((max_len/5)*(-1), position[i]+ 0.1, labels_fn_rank[i], fontsize =fontsize, horizontalalignment = 'left')
    label_width = np.zeros(len_functions)
    for j in range(len(species_sorted)):
        axs.barh(position[i], stregnth_fn_rank[i][j],left=label_width, color = dic_colors_sp[species_sorted[j]])
        label_width += stregnth_fn_rank[i][j]
axs.invert_yaxis()
for pos in ['right', 'top', 'left']:
    axs.spines[pos].set_visible(False)
# for axis in ['bottom']:
#     axs.spines[axis].set_linewidth(3)
axs.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
axs.set_xticks(np.arange(-4,10,2))
axs.set_xticklabels(["","", '0', '2', '4', '6', '8'])
axs.set_yticks([])
axs.set_xlabel('Multispecies function keystoness', fontsize = fontsize)
 
#plant.savefig(plt_dir + '/rank_figure4.pdf', bbox_inches='tight' ,  dpi=100)

labels = species_sorted
handles = [plt.Rectangle((0,0),1,1, color=dic_colors_sp[sp]) for sp in species_sorted]
labels = titles(species_sorted, line_sep = False)
fig, ax = plt.subplots(figsize=(5,1), dpi=100)
ax.legend(handles =handles,labels=labels,fontsize= fontsize,loc= 'center', frameon=False, ncol= 2)
plt.axis('off')
 
#plant.savefig(plt_dir + '/rank_figure4_legend.pdf', bbox_inches='tight',  dpi=100 )

#%%



#Vale ahora vmos a hacer los ranking del SI

#Ahora calculamos R_i

def rank_SI(df_matrix, df_matrix_prod, function_sorted, sort_rank = 'function'):
    c_i_alpha = df_matrix[['plant_sp']+function_sorted].copy()
    for column in function_sorted:
        denominator = c_i_alpha[column].sum()
        c_i_alpha[column] = c_i_alpha[column]/ denominator
    c_i_alpha['R_i'] =  c_i_alpha.loc[:,[c for c in c_i_alpha.columns if c!= "plant_sp"]].mean(axis=1)
    ranks = pd.DataFrame()

    ranks['plant_sp'] = c_i_alpha['plant_sp']
    ranks['R_i'] = c_i_alpha['R_i']
    #c_i_alpha = c_i_alpha.sort_values(by=['R_i'], ascending=False)
    #Ahora calculamos rho_i
    rho_i_alpha = df_matrix_prod.copy()

    for function in function_sorted:

        rho_i_alpha[function + ' times ' +function] = df_matrix[function].to_numpy() **2
    for column in rho_i_alpha.columns[1:]:
        denominator = rho_i_alpha[column].sum()
        rho_i_alpha[column] = rho_i_alpha[column]/ denominator
    rho_i_alpha['rho_i'] =  rho_i_alpha.loc[:,[c for c in rho_i_alpha.columns if c!= "plant_sp"]].mean(axis=1)
    ranks['rho_i'] = rho_i_alpha['rho_i']

    if sort_rank == 'Function':
        ranks = ranks.sort_values(by=['R_i'], ascending=False)
    else:
        ranks = ranks.sort_values(by=['rho_i'], ascending=False)
    #rho_i_alpha = rho_i_alpha.sort_values(by=['rho_i'], ascending=False)

    return ranks

ranks = rank_SI(df_matrix, df_matrix_prod, function_sorted, sort_rank = 'function')
labels =[plant.title() for plant in ranks['plant_sp'].to_list()]
plot_str = 'A'            
width = 0.35       # the width of the bars: can also be len(x) sequence


fig, axs = plt.subplots(figsize=(12,8), dpi=100)
fontsize=20
position = np.arange(1,len_species+1)
max_len = -1
for ele in labels:
    if len(ele) > max_len:
        max_len = len(ele)
axs.barh(position, ranks['rho_i'], color = 'navy')
axs.barh(position, ranks['R_i']* (-1), color = 'darkred')
for j in range(len_species):
    #print(labels[j])
    text = item_words(labels[j], line_sep=False)
    #print(text)
    axs.text((0+np.max(ranks['R_i'].to_numpy())+.2)* (-1), position[j]+ 0.2, text, fontsize =fontsize, horizontalalignment = 'left')  
#axs.text(-0.03,0, r"$\mathbf{R_i}$", fontsize=fontsize, horizontalalignment = 'right')
#axs.text(0.03,0, r"$\mathbf{\rho_i}$", fontsize=fontsize, horizontalalignment = 'left');
axs.text(-0.03,0, r"$R_i$", fontsize=fontsize, horizontalalignment = 'right')
axs.text(0.03,0, r"$\rho_i$", fontsize=fontsize, horizontalalignment = 'left');
axs.invert_yaxis()
for pos in ['right', 'top', 'left']:
    axs.spines[pos].set_visible(False)
# for axis in ['bottom']:
#     axs.spines[axis].set_linewidth(3)
axs.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
axs.set_xticks(np.arange(-.2,.35,.1))
axs.set_xticklabels(['0.2','0.1','0.0', '0.1', '0.2', '0.3'])
axs.set_yticks([])
#plt.setp(axs,xticks = np.arange(-.6,.35,.1), xticklabels = [' ',' ', ' ' ,' ',' ', ' ',0.0', '0.1', '0.2', '0.3'], yticks =[])
 
#plant.savefig(plt_dir + '/ranks_SI.pdf', bbox_inches='tight',  dpi=100 )



#%%

'''Rankin con vegetation cover'''


labels_sp_rank_low = [lab.capitalize() for lab in labels_sp_rank]
contr_sp = np.sum(stregnth_sp_rank, axis=1)
abundance_sp = [df_matrix[df_matrix['plant_sp'] ==plant]['abundance'].to_numpy()[0] for plant in labels_sp_rank_low ]
cover_sp = [df_matrix[df_matrix['plant_sp'] ==plant]['cover'].to_numpy()[0] for plant in labels_sp_rank_low ]


fontsize=20
# create figure and axis objects with subplots()
fig,ax = plt.subplots(figsize=(12,8), dpi=100)  
ax.scatter(contr_sp,abundance_sp,marker="^",s=300,facecolors='none', edgecolors='darkred')
ax.set_ylabel("Abundance  (number of individuals)",color='darkred',fontsize=fontsize)
ax2=ax.twinx()
#ax.set_yticks(np.arange(0,300,50))
#ax.set_yticklabels(np.arange(0,300,50), minor=False, fontsize=fontsize)
ax.set_xlabel("Specie total contribution",color="k",fontsize=fontsize)
ax.set_xticks(np.arange(0,1.6,.2))
ax.set_xticklabels(['0', '0.2','0.4','0.6','0.8','1.0','1.2','1.4'], minor=False, fontsize=fontsize)
ax2.scatter(contr_sp, cover_sp,marker="x",c='navy', s=300)
ax2.set_ylabel("Vegetation cover ("+ r'$m^2$'+')' ,color="navy",fontsize=fontsize)
#ax2.set_yticks(np.arange(0,30000,5000))
#ax2.set_yticklabels(np.arange(0,30000,5000), minor=False, fontsize=fontsize)
ax.set_yscale('log')
ax2.set_yscale('log')
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
# for axis in ['bottom', 'left']:
#     ax.spines[axis].set_linewidth(3)
# for axis in ['right']:
#     ax2.spines[axis].set_linewidth(3)
ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
ax2.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
 
#plant.savefig(plt_dir + '/abundance_cover_corr.pdf', bbox_inches='tight',  dpi=100 )

print('ab pearson:', scipy.stats.pearsonr(contr_sp, abundance_sp))
print('ab spearma:', scipy.stats.spearmanr(contr_sp, abundance_sp))
print('log ab pearson:', scipy.stats.pearsonr(contr_sp, np.log10(abundance_sp)))
print('cov pearson:', scipy.stats.pearsonr(contr_sp, cover_sp))
print('cov spearma', scipy.stats.spearmanr(contr_sp, cover_sp))
print('log cov pearson',scipy.stats.pearsonr(contr_sp, np.log10(cover_sp)))
print('cov ab',scipy.stats.pearsonr(abundance_sp, cover_sp))

# a = pd.DataFrame()
# a['w'] = contr_sp
# a['abundance'] = abundance_sp
# a['cover'] = cover_sp


# a['w_rank'] = a['w'].rank(method = 'average',ascending=True)
# a['abundance_rank'] = a['abundance'].rank(method = 'average',ascending=True)
# a['cover_rank'] = a['cover'].rank(method = 'average',ascending=True)

# order_1 = contr_sp.argsort()
# ranks_1 = order_1.argsort()

# order_2 = np.array(abundance_sp).argsort()
# ranks_2 = order_2.argsort()

# order_3 = np.array(cover_sp).argsort()
# ranks_3 = order_2.argsort()

# one =scipy.stats.spearmanr(ranks_1, ranks_2)
# two = scipy.stats.spearmanr(ranks_1, ranks_3)
# three = scipy.stats.spearmanr(ranks_2, ranks_3)

# stats.spearmanr(a['w_rank'].to_numpy(), a['abundance_rank'].to_numpy())
# stats.spearmanr(a['w_rank'].to_numpy(), a['cover_rank'].to_numpy())
# stats.spearmanr(a['abundance_rank'].to_numpy(), a['cover_rank'].to_numpy())

#%%


'''Percolacion agregadas'''

def func_color(att_list,cmp = plt.cm.rainbow):
     
    '''Returns a colormap dic based on the palette 'plt.cm.rainbow',
       discretized into the number of (different) colors
       found in the attribute.
   
    Parameters
    ----------
    att_list: attributes list
   
    Returns
    ----------
    cmp_edges: cmap
        Cmap of edges.
   
    colors: array
        array of edges attributs in G.edges order.
       
    func_to_color: dict
        Dict object attr: color.
       
    '''
     
   
    #If data is default dataset or has the same ecological functions uses pre-established colors
    if Counter(att_list) == Counter(['herbivory','pollination','saprotrophic-fungi','seed-dispersal', 'symbiontic-fungi','pathongen-fungi']):
        func_to_color= {'herbivory': 'tab:purple',
         'plant': 'tab:green',
         'pollination': 'tab:orange',
         'saprotrophic-fungi': 'tab:brown',
         'pathongen-fungi': 'tab:red',
         'seed-dispersal': 'tab:blue',
         'symbiontic-fungi': 'tab:pink'}
   
    # If not, used the input colormap
    else:
        Ncolors          = len(np.unique(att_list))
        cmp_n       = cmp(np.linspace(0,1,Ncolors))
        cmp_fun        = mpl.colors.ListedColormap(cmp_n)
        func_to_int      = {f:i for i, f in enumerate(functions)}
        func_to_color    = {k:cmp_fun.colors[i] for k,i in func_to_int.items()}
   
    return func_to_color

#dic_colors = func_color(function_sorted,cmp = plt.cm.rainbow)
dic_colors_fn['total'] = 'tab:grey'


for proj in ['plant', 'function']:
#for proj in ['function']:
   
    fig,ax = plt.subplots(figsize=(12,8), dpi=300)
    if proj == 'function':
        A = np.matmul(P_matrix,P_matrix_t)  
    else:
        A = np.matmul(P_matrix_t,P_matrix)
   
    df = pd.DataFrame()
    fontsize = 25    
    legend_elements = []
    for row in [1,0]:
        G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
        dic_edges = {(i,j):G[i][j]["weight"] for i,j in G.edges }
        Y0 = len(max(nx.connected_components(G), key=len))    
        edges = []
        weight = []
        for elem in dic_edges:
            edges.append(elem)
            weight.append(dic_edges[elem])
        if row ==1:
            sort = np.argsort(np.array(weight))[::-1]
        else:
            sort = np.argsort(np.array(weight))
           
        weigth_sort = np.array(weight)[sort]
        print(row)
        print(weigth_sort)
        edges_sort = list(np.array(edges)[sort])
        edges_sort = tuple(tuple(sub) for sub in edges_sort)
        nodes = [k for k in G.nodes]
        T = []
        T_2 = []
        T_2_2 = []
        x = []
        for edge in edges_sort:
            #print(G[edge[0]][edge[1]])
            G.remove_edge(edge[0], edge[1])
            #P.remove_edge(edge[0], edge[1])
            components = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            #print(components)
            T.append(len(components[0]))
            if len(components) >1:
                length = np.array([len(k) for k in components[1:]])
                areas = length **2
                T_2.append(np.mean(areas))
                T_2_2.append(len(components[1]))
            else:
                T_2.append(0)
                T_2_2.append(0)
            x.append(dic_edges[edge]+np.sum(x))
           
        if row==0:
            pc_asc = np.max(np.array(T_2))
            T_asc = T
            w_asc = weigth_sort
           
            color = 'darkred'
            marker = "v"
        if row==1:
            pc_des = np.max(np.array(T_2))
            T_des = T
            w_des = weigth_sort
            color = 'navy'
            marker = "^"
           
               
       

        y = np.array([Y0] + T) / max(T)
        x = np.arange(0,len(y),1)
        x = x / x.max()
        dx = x[2] - x[1]

        if row ==0:
   
            df['Edges counter (norm)'] = x
            df['Size largest comp (norm) (asc)'] = y
        if row == 1:
            df['Size largest comp (norm) (des)'] = y

        auc = lambda x,dx: dx * (x[1:-1].sum()) + dx/2 *(x[0] + x[-1])
        Ax = auc(y, dx)

        area = scipy.integrate.simpson(y,dx=dx, even='avg')

        # if row ==0 :
        #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'
        # if row ==1 :
        #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'


        if row ==0 :
            label  = f'Ascending order, AUC ={round(area, 2)}'

        if row ==1 :
            label  = f'Descending order, AUC = {round(area, 2)}'
            #print('here')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10, linestyle='none', markeredgecolor = color))
        #legend_elements.append(Line2D([0], [0], label=label, color=color, lw=5))
        #ax.scatter(x, y, marker=marker,s=300, color=color)
        #ax.legend(fontsize= fontsize, frameon=False, loc = 'lower center',bbox_to_anchor = (.45,-.3))
        plt.fill_between(x, y, y2=0, color = color, alpha = .2)
        plt.scatter(x, y, c = color)
        #ax.bar(x[1:-1], y[1:-1], width = dx, alpha = .2, color = color)
        #ax.bar([x[0],x[-1]-dx/2], [y[0],y[-1]], width = dx/2, align='edge', alpha = .2, color =color)
        #ax.set_title('Percolation in Plant-Plant ',{'fontsize':size_label, 'fontweight':"bold"}, pad=-15)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
    ax.set_xticks(np.arange(0,1.2,.2))
    ax.set_xlabel(r'$m/M$'+'\n'+'Fraction of pruned edges', fontsize= fontsize)
    ax.set_xticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(np.arange(0,1.2,.2))
    ax.set_ylabel(r'$S(m)/n$'+'\n'+'(normalized) size of LCC', fontsize= fontsize)
    ax.set_yticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    for pos in [ 'top', 'right']:
        ax.spines[pos].set_visible(False)
    # for axis in ['left','bottom']:
    #     ax.spines[axis].set_linewidth(3)  
    ax.set_xlim([0-.02,x[-1]+.02])
    ax.set_ylim([0-.05,x[-1]+.05])
    #print(legend_elements)
    ax.legend(handles=legend_elements,fontsize= fontsize, frameon=False, loc = 'lower left')
    plt.savefig(plt_dir + f'/percolation_{proj}.pdf', bbox_inches='tight' ,  dpi=300)
    plt.savefig(plt_dir + f'/percolation_{proj}.png', bbox_inches='tight' ,  dpi=300)
    df.to_csv(df_dir+ f'/df_{proj}.csv',index=False)


#%%

'''Percolacion agregadas pesos'''

for proj in ['plant', 'function']:
#for proj in ['function']:
   
    fig,ax = plt.subplots(figsize=(12,8), dpi=300)
    if proj == 'function':
        A = np.matmul(P_matrix,P_matrix_t)  
    else:
        A = np.matmul(P_matrix_t,P_matrix)
   
    df = pd.DataFrame()
    fontsize = 25    
    legend_elements = []
    for row in [1,0]:
        G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
        dic_edges = {(i,j):G[i][j]["weight"] for i,j in G.edges }
        Y0 = len(max(nx.connected_components(G), key=len))    
        edges = []
        weight = []
        for elem in dic_edges:
            edges.append(elem)
            weight.append(dic_edges[elem])
        if row ==1:
            sort = np.argsort(np.array(weight))[::-1]
        else:
            sort = np.argsort(np.array(weight))
           
        weigth_sort = np.array(weight)[sort]
        total_strength = np.sum(weigth_sort)
        print(row)
        print(weigth_sort)
        edges_sort = list(np.array(edges)[sort])
        edges_sort = tuple(tuple(sub) for sub in edges_sort)
        nodes = [k for k in G.nodes]
        T = []
        T_2 = []
        T_2_2 = []
        x = [0]
        for edge in edges_sort:
            #print(G[edge[0]][edge[1]])
            edge_weight = G[edge[0]][edge[1]]["weight"]
            x.append(x[-1]+edge_weight)
            G.remove_edge(edge[0], edge[1])
            #P.remove_edge(edge[0], edge[1])
            components = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            #print(components)
            T.append(len(components[0]))
            if len(components) >1:
                length = np.array([len(k) for k in components[1:]])
                areas = length **2
                T_2.append(np.mean(areas))
                T_2_2.append(len(components[1]))
            else:
                T_2.append(0)
                T_2_2.append(0)
           
        if row==0:
            pc_asc = np.max(np.array(T_2))
            T_asc = T
            w_asc = weigth_sort
           
            color = 'darkred'
            marker = "v"
        if row==1:
            pc_des = np.max(np.array(T_2))
            T_des = T
            w_des = weigth_sort
            color = 'navy'
            marker = "^"
           
               
       

        y = np.array([Y0] + T) / max(T)
        x = np.array(x) / total_strength

        if row ==0:
   
            df['Edges counter (norm)'] = x
            df['Size largest comp (norm) (asc)'] = y
        if row == 1:
            df['Size largest comp (norm) (des)'] = y

        auc = lambda x,dx: dx * (x[1:-1].sum()) + dx/2 *(x[0] + x[-1])
        Ax = auc(y, dx)

        area = scipy.integrate.simpson(y,x=x, even='avg')

        # if row ==0 :
        #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'
        # if row ==1 :
        #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'


        if row ==0 :
            label  = f'Ascending order, AUC ={round(area, 2)}'

        if row ==1 :
            label  = f'Descending order, AUC = {round(area, 2)}'
            #print('here')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10, linestyle='none', markeredgecolor = color))
        #legend_elements.append(Line2D([0], [0], label=label, color=color, lw=5))
        #ax.scatter(x, y, marker=marker,s=300, color=color)
        #ax.legend(fontsize= fontsize, frameon=False, loc = 'lower center',bbox_to_anchor = (.45,-.3))
        plt.fill_between(x, y, y2=0, color = color, alpha = .2)
        plt.scatter(x, y, c = color)
        #ax.bar(x[1:-1], y[1:-1], width = dx, alpha = .2, color = color)
        #ax.bar([x[0],x[-1]-dx/2], [y[0],y[-1]], width = dx/2, align='edge', alpha = .2, color =color)
        #ax.set_title('Percolation in Plant-Plant ',{'fontsize':size_label, 'fontweight':"bold"}, pad=-15)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
    ax.set_xticks(np.arange(0,1.2,.2))
    ax.set_xlabel(r'$s_{removed}/s_0$', fontsize= fontsize)
    ax.set_xticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(np.arange(0,1.2,.2))
    ax.set_ylabel(r'$S(m)/n$'+'\n'+'(normalized) size of LCC', fontsize= fontsize)
    ax.set_yticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    for pos in [ 'top', 'right']:
        ax.spines[pos].set_visible(False)
    # for axis in ['left','bottom']:
    #     ax.spines[axis].set_linewidth(3)  
    ax.set_xlim([0-.02,x[-1]+.02])
    ax.set_ylim([0-.05,x[-1]+.05])
    #print(legend_elements)
    ax.legend(handles=legend_elements,fontsize= fontsize, frameon=False, loc = 'lower left')
    #plt.savefig(plt_dir + f'/percolation_{proj}_original_pesos.pdf', bbox_inches='tight' ,  dpi=300)
    #plt.savefig(plt_dir + f'/percolation_{proj}_original_pesos.png', bbox_inches='tight' ,  dpi=300)
#%%
'''Ordenar Matriz'''




#%%

'''Percolacion agregadas maxima nestedness'''


for proj in ['plant', 'function']:
#for proj in ['function']:
   
    fig,ax = plt.subplots(figsize=(12,8), dpi=300)
    
    
    if proj == 'function':
        A = np.matmul(P_matrix,P_matrix_t)  
    else:
        A = np.matmul(P_matrix_t,P_matrix)
   
    df = pd.DataFrame()
    fontsize = 25    
    legend_elements = []
    for row in [1,0]:
        G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
        dic_edges = {(i,j):G[i][j]["weight"] for i,j in G.edges }
        Y0 = len(max(nx.connected_components(G), key=len))    
        edges = []
        weight = []
        for elem in dic_edges:
            edges.append(elem)
            weight.append(dic_edges[elem])
        if row ==1:
            sort = np.argsort(np.array(weight))[::-1]
        else:
            sort = np.argsort(np.array(weight))
           
        weigth_sort = np.array(weight)[sort]
        print(row)
        print(weigth_sort)
        edges_sort = list(np.array(edges)[sort])
        edges_sort = tuple(tuple(sub) for sub in edges_sort)
        nodes = [k for k in G.nodes]
        T = []
        T_2 = []
        T_2_2 = []
        x = []
        for edge in edges_sort:
            #print(G[edge[0]][edge[1]])
            G.remove_edge(edge[0], edge[1])
            #P.remove_edge(edge[0], edge[1])
            components = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            #print(components)
            T.append(len(components[0]))
            if len(components) >1:
                length = np.array([len(k) for k in components[1:]])
                areas = length **2
                T_2.append(np.mean(areas))
                T_2_2.append(len(components[1]))
            else:
                T_2.append(0)
                T_2_2.append(0)
            x.append(dic_edges[edge]+np.sum(x))
           
        if row==0:
            pc_asc = np.max(np.array(T_2))
            T_asc = T
            w_asc = weigth_sort
           
            color = 'darkred'
            marker = "v"
        if row==1:
            pc_des = np.max(np.array(T_2))
            T_des = T
            w_des = weigth_sort
            color = 'navy'
            marker = "^"
           
               
       

        y = np.array([Y0] + T) / max(T)
        x = np.arange(0,len(y),1)
        x = x / x.max()
        dx = x[2] - x[1]

        if row ==0:
   
            df['Edges counter (norm)'] = x
            df['Size largest comp (norm) (asc)'] = y
        if row == 1:
            df['Size largest comp (norm) (des)'] = y

        auc = lambda x,dx: dx * (x[1:-1].sum()) + dx/2 *(x[0] + x[-1])
        Ax = auc(y, dx)

        area = scipy.integrate.simpson(y,dx=dx, even='avg')

        # if row ==0 :
        #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'
        # if row ==1 :
        #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'


        if row ==0 :
            label  = f'Ascending order, AUC ={round(area, 2)}'

        if row ==1 :
            label  = f'Descending order, AUC = {round(area, 2)}'
            #print('here')
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=10, linestyle='none', markeredgecolor = color))
        #legend_elements.append(Line2D([0], [0], label=label, color=color, lw=5))
        #ax.scatter(x, y, marker=marker,s=300, color=color)
        #ax.legend(fontsize= fontsize, frameon=False, loc = 'lower center',bbox_to_anchor = (.45,-.3))
        plt.fill_between(x, y, y2=0, color = color, alpha = .2)
        plt.scatter(x, y, c = color)
        #ax.bar(x[1:-1], y[1:-1], width = dx, alpha = .2, color = color)
        #ax.bar([x[0],x[-1]-dx/2], [y[0],y[-1]], width = dx/2, align='edge', alpha = .2, color =color)
        #ax.set_title('Percolation in Plant-Plant ',{'fontsize':size_label, 'fontweight':"bold"}, pad=-15)
    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
    ax.set_xticks(np.arange(0,1.2,.2))
    ax.set_xlabel(r'$m/M$'+'\n'+'Fraction of pruned edges', fontsize= fontsize)
    ax.set_xticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(np.arange(0,1.2,.2))
    ax.set_ylabel(r'$S(m)/n$'+'\n'+'(normalized) size of LCC', fontsize= fontsize)
    ax.set_yticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    for pos in [ 'top', 'right']:
        ax.spines[pos].set_visible(False)
    # for axis in ['left','bottom']:
    #     ax.spines[axis].set_linewidth(3)  
    ax.set_xlim([0-.02,x[-1]+.02])
    ax.set_ylim([0-.05,x[-1]+.05])
    #print(legend_elements)
    ax.legend(handles=legend_elements,fontsize= fontsize, frameon=False, loc = 'lower left')
    plt.savefig(plt_dir + f'/percolation_{proj}.pdf', bbox_inches='tight' ,  dpi=300)
    plt.savefig(plt_dir + f'/percolation_{proj}.png', bbox_inches='tight' ,  dpi=300)
    df.to_csv(df_dir+ f'/df_{proj}.csv',index=False)


#%%


# '''Percolacion desagregadas'''


# for proj in ['function', 'plant']:

#     if proj == 'function':
#         length= len_functions
#         el =function_sorted
#         plot_str = 'ABCDEF'
#         fig, axs = plt.subplot_mosaic('ABC\nDEF', figsize=(12*3,8*2), dpi=100)
#     if proj == 'plant':
#         length = len_species
#         el = species_sorted
#         plot_str = 'ABCDEFGHIJKLMNOP'
#         fig, axs = plt.subplot_mosaic('ABCD\nEFGH\nIJKL\nMNOP', figsize=(12*4,8*4), dpi=100)

   
#     pc_asc = np.zeros(length)
#     pc_des = np.zeros(length)
#     fig.subplots_adjust(wspace=.15,top=1.03,hspace=.35)
#     for i in range(length):
#         df = pd.DataFrame()
#         fontsize = 25    
#         legend_elements = []
#         item = plot_str[i]
#         for row in [0,1]:
           
#             if proj =='function':
#                 p, p_t= p_matrix_elem(df_matrix_s, elem = [el[i]] , axis=1)
#                 A = np.matmul(p,p_t)  
#             if proj == 'plant':
#                 p, p_t= p_matrix_elem(df_matrix_s, elem = [species_sorted[i]] , axis=0)
#                 A = np.matmul(p_t,p)  
           
               
#             G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
#             dic_edges = {(i,j):G[i][j]["weight"] for i,j in G.edges }
#             Y0 = len(max(nx.connected_components(G), key=len))  
#             edges = []
#             weight = []
#             for elem in dic_edges:
#                 edges.append(elem)
#                 weight.append(dic_edges[elem])
#             if row ==1:
#                 sort = np.argsort(np.array(weight))[::-1]
#             else:
#                 sort = np.argsort(np.array(weight))
               
#             weigth_sort = np.array(weight)[sort]
#             edges_sort = list(np.array(edges)[sort])
#             edges_sort = tuple(tuple(sub) for sub in edges_sort)
#             nodes = [k for k in G.nodes]
           
#             T = []
#             T_2 = []
#             T_2_2 = []
#             x = []
#             for edge in edges_sort:
               
#                 G.remove_edge(edge[0], edge[1])
#                 #P.remove_edge(edge[0], edge[1])
#                 components = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
#                 #print(components)
#                 T.append(len(components[0]))
#                 if len(components) >1:
#                     length = np.array([len(k) for k in components[1:]])
#                     areas = length **2
#                     T_2.append(np.mean(areas))
#                     T_2_2.append(len(components[1]))
#                 else:
#                     T_2.append(0)
#                     T_2_2.append(0)
#                 x.append(dic_edges[edge]+np.sum(x))
               
#             if row==0:
#                 pc_asc = np.max(np.array(T_2))
#                 T_asc = T
#                 w_asc = weigth_sort
               
#                 color = 'darkred'
#                 marker = "v"
#             if row==1:
#                 pc_des = np.max(np.array(T_2))
#                 T_des = T
#                 w_des = weigth_sort
#                 color = 'navy'
#                 marker = "^"
               
                   
#             y = np.array([Y0] + T) / max(T)
#             x = np.arange(0,len(y),1)
#             x = x / x.max()
#             dx = x[2] - x[1]

#             if row ==0:
       
#                 df['Edges counter (norm)'] = x
#                 df['Size largest comp (norm) (asc)'] = y
#             if row == 1:
#                 df['Size largest comp (norm) (des)'] = y

#             auc = lambda x,dx: dx * (x[1:-1].sum()) + dx/2 *(x[0] + x[-1])
#             Ax = auc(y, dx)
#             area = scipy.integrate.simpson(y,dx=dx, even='avg')
#             if row ==0 :
#                 label  = f'Ascending order, AUC ={round(area, 2)}'

#             if row ==1 :
#                 label  = f'Descending order, AUC = {round(area, 2)}'
#                 #print('here')
#             legend_elements.append(Patch(label=label,facecolor=color, edgecolor=color, alpha = .8))
#             #axs[item].scatter(x, y, marker=marker,s=300, color=color,alpha = .2)
#             axs[item].fill_between(x, y, y2=0, color = color, alpha = .2, label ='3')
#         axs[item].tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
#         axs[item].set_xticks(np.arange(0,1.2,.2))
#         axs[item].set_xlabel(r'$m/M$', fontsize= fontsize)
#         axs[item].set_xticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
#         axs[item].set_yticks(np.arange(0,1.2,.2))
#         axs[item].set_ylabel(r'$S(m)/n$', fontsize= fontsize)
#         axs[item].set_yticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
#         for pos in [ 'top', 'right']:
#             axs[item].spines[pos].set_visible(False)
#         # for axis in ['left','bottom']:
#         #     axs[item].spines[axis].set_linewidth(3)  
#         axs[item].set_xlim([0-.02,x[-1]+.02])
#         axs[item].set_ylim([0-.05,x[-1]+.05])
#         axs[item].set_title(el[i].title(),{'fontsize':fontsize, 'fontweight':"bold"}, pad=30)
#         axs[item].legend(handles=legend_elements,fontsize= fontsize, frameon=False, loc = 'lower left')
     
#     #plant.savefig(plt_dir + f'/percolation_des{proj}.pdf', bbox_inches='tight',  dpi=100 ) #save

# #%%


# '''Percolacion'''
# plot_str = 'ABC'            


# fontsize = 25    
# fig, axs = plt.subplot_mosaic('ABC', figsize=(12*3,8), dpi=100)
# fig.subplots_adjust(wspace=.15,top=1.03,hspace=.35)
# i=0
# row=0
# for function in ['total']:
   
#     A = np.matmul(P_matrix, P_matrix_t)

#     #color= dic_colors[function]
 
#     ascending_labels = 'Adding edges in ascending weight order, ' + r'$\sum_{\alpha} P_{\alpha}^i P_{\alpha}^j$'
#     descending_labels ='Removing edges in descending weight order ' + r'$\sum_{\alpha} P_{\alpha}^i P_{\alpha}^j$'

           
#     G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)  
#     dic_edges = {(i,j):G[i][j]["weight"] for i,j in G.edges }
   
#     ascending_color = 'darkred'
#     descending_color = 'navy'
   

   
#     for row in [0,1]:
#         edges = []
#         weight = []
#         for elem in dic_edges:
#             edges.append(elem)
#             weight.append(dic_edges[elem])
#         if row ==1:
#             sort = np.argsort(np.array(weight))[::-1]
#         else:
#             sort = np.argsort(np.array(weight))
           
#         weigth_sort = np.array(weight)[sort]
#         edges_sort = list(np.array(edges)[sort])
#         edges_sort = tuple(tuple(sub) for sub in edges_sort)
#         nodes = [k for k in G.nodes]
       
#         P = nx.Graph()      
#         P.add_nodes_from(nodes)
#         T = []
#         T_2 = []
#         T_2_2 = []
#         x = []
#         for edge in edges_sort:
           
#             P.add_edge(edge[0], edge[1], weight = dic_edges[edge])
#             #P.remove_edge(edge[0], edge[1])
#             components = [c for c in sorted(nx.connected_components(P), key=len, reverse=True)]
#             T.append(len(components[0]))
#             if len(components) >1:
#                 #print(len(components[1]))
#                 length = np.array([len(k) for k in components[1:]])
#                 areas = length **2
#                 #T_2.append(np.mean(areas/ np.sum(areas)))
#                 T_2.append(np.mean(areas))
#                 T_2_2.append(len(components[1]))
#                 #T_2_2.append(areas[0]/np.sum(areas))
#             else:
#                 T_2.append(0)
#                 T_2_2.append(0)
#             x.append(dic_edges[edge]+np.sum(x))
   
   
#         for j in range(3):
#             item = plot_str[i]
#             axs1 = axs[item]
#             if row==1:
#                 axs2 = axs1.twiny()
#                 axs2.cla()
           
           
#             if j ==0:
#                 if row == 1:
#                     axs2.scatter(weigth_sort,T, c=descending_color, s=100, alpha =.5)
#                     axs2.set_yticks(np.arange(1,len(species)+2,2))
#                     axs2.set_yticklabels(np.arange(1,len(species)+2,2), fontsize=fontsize)
#                     axs2.set_xticks(np.arange(0,3+1,1)[::-1])
#                     axs2.set_xticklabels([3, 2, 1, 0], fontsize=fontsize)
#                     axs2.set_ylim(-.1,20)  
#                     axs2.set_xlim(-.1,4)
#                     axs2.invert_xaxis()
#                     axs2.xaxis.label.set_color(descending_color)        #setting up X-axis label color
#                     axs2.tick_params(axis='x', colors=descending_color)    #setting up X-axis tick color
#                     for pos in ['right', 'bottom']:
#                         axs2.spines[pos].set_visible(False)
#                     # for axis in ['left', 'top']:
#                     #     axs2.spines[axis].set_linewidth(5)
#                     axs2.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#                 else:
#                     axs1.scatter(weigth_sort,T, c=ascending_color, s=100, alpha =.5)
#                     axs1.set_yticks(np.arange(1,len(species)+2,2))
#                     axs1.set_yticklabels(np.arange(1,len(species)+2,2), fontsize=fontsize)
#                     axs1.set_xticks(np.arange(0,3+1,1))
#                     axs1.set_xticklabels([0,1,2,3], fontsize=fontsize)
#                     axs1.set_ylim(-.1,17)  
#                     axs1.set_xlim(-.1,4)
#                     axs1.xaxis.label.set_color(ascending_color)        #setting up X-axis label color
#                     axs1.tick_params(axis='x', colors=ascending_color)    #setting up X-axis tick color
#                     axs1.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#                     for pos in ['right', 'top']:
#                         axs1.spines[pos].set_visible(False)
#                     # for axis in ['bottom','left']:
#                     #     axs1.spines[axis].set_linewidth(5)
                   
#                 #axs1.set_title('',{'fontsize':fontsize, 'fontweight':"bold"}, pad=30)
 
#             if j==1:
#                 if row == 1:
#                     axs2.scatter(weigth_sort,T_2, c=descending_color, s=100, alpha =.5)
#                     axs2.set_yticks(np.arange(0,4+1,1))
#                     axs2.set_yticklabels([0,1,2,3,4], fontsize=fontsize)
#                     axs2.set_xticks(np.arange(0,3+1,1)[::-1])
#                     axs2.set_xticklabels([3, 2, 1, 0], fontsize=fontsize)
#                     axs2.set_ylim(-.1,4.1)  
#                     axs2.set_xlim(-.1,4)
#                     axs2.invert_xaxis()
#                     axs2.xaxis.label.set_color(descending_color)        #setting up X-axis label color
#                     axs2.tick_params(axis='x', colors=descending_color)    #setting up X-axis tick color
#                     for pos in ['right', 'bottom']:
#                         axs2.spines[pos].set_visible(False)
#                     # for axis in ['left', 'top']:
#                     #     axs2.spines[axis].set_linewidth(5)
#                     axs2.set_xlabel(descending_labels, fontsize=fontsize, color='k', labelpad=20)
#                     axs2.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#                 else:
#                     axs1.scatter(weigth_sort,T_2, c=ascending_color, s=100, alpha =.5)
#                     axs1.set_yticks(np.arange(0,4+1,1))
#                     axs1.set_yticklabels([0,1,2,3,4], fontsize=fontsize)
#                     axs1.set_xticks(np.arange(0,3+1,1))
#                     axs1.set_xticklabels([0,1,2,3], fontsize=fontsize)
#                     axs1.set_ylim(-.1,4.1)  
#                     axs1.set_xlim(-.1,4)
#                     axs1.xaxis.label.set_color(ascending_color)        #setting up X-axis label color
#                     axs1.tick_params(axis='x', colors=ascending_color)    #setting up X-axis tick color
#                     axs1.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#                     for pos in ['right', 'top']:
#                         axs1.spines[pos].set_visible(False)
#                     # for axis in ['bottom','left']:
#                     #     axs1.spines[axis].set_linewidth(5)
#                     axs1.set_ylabel('Mean area non-largest components', fontsize=fontsize)
#                     axs1.set_xlabel(ascending_labels, fontsize=fontsize, color='k', labelpad=20)

           
#             if j==2:
#                 if row == 1:
#                     axs2.scatter(weigth_sort,T_2_2, c=descending_color, s=100, alpha =.5)
#                     axs2.set_yticks(np.arange(0,4+1,1))
#                     axs2.set_yticklabels([0,1,2,3,4], fontsize=fontsize)
#                     axs2.set_xticks(np.arange(0,3+1,1)[::-1])
#                     axs2.set_xticklabels([3, 2, 1, 0], fontsize=fontsize)
#                     axs2.set_ylim(-.1,4.1)  
#                     axs2.set_xlim(-.1,4)
#                     axs2.invert_xaxis()
#                     axs2.xaxis.label.set_color(descending_color)        #setting up X-axis label color
#                     axs2.tick_params(axis='x', colors=descending_color)    #setting up X-axis tick color
#                     for pos in ['right', 'bottom']:
#                         axs2.spines[pos].set_visible(False)
#                     # for axis in ['left', 'top']:
#                     #     axs2.spines[axis].set_linewidth(5)
#                     axs2.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#                 else:
#                     axs1.scatter(weigth_sort,T_2_2, c=ascending_color, s=100, alpha =.5)
#                     axs1.set_yticks(np.arange(0,4+1,1))
#                     axs1.set_yticklabels([0,1,2,3,4], fontsize=fontsize)
#                     axs1.set_xticks(np.arange(0,3+1,1))
#                     axs1.set_xticklabels([0,1,2,3], fontsize=fontsize)
#                     axs1.set_ylim(-.1,4.1)  
#                     axs1.set_xlim(-.1,4)
#                     axs1.xaxis.label.set_color(ascending_color)        #setting up X-axis label color
#                     axs1.tick_params(axis='x', colors=ascending_color)    #setting up X-axis tick color
#                     axs1.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#                     for pos in ['right', 'top']:
#                         axs1.spines[pos].set_visible(False)
#                     # for axis in ['bottom','left']:
#                     #     axs1.spines[axis].set_linewidth(5)
#                 #axs[item].set_title('',{'fontsize':fontsize, 'fontweight':"bold"}, pad=30)
               
#             if i<2:
#                 i+=1
#             else:
#                 i=0
#         row +=1
       
          
# #plant.savefig(plt_dir + '/percolation_agr.pdf', bbox_inches='tight' ,  dpi=100 ) #save



# #%%


# plot_str = 'ABCDEF'            


# fontsize = 25    

# for proj in ['function']:

#     if proj == 'function':
#         length= len_functions
#         el =function_sorted
#         plot_str = 'ABCDEF'
#         fig, axs = plt.subplot_mosaic('ABC\nDEF', figsize=(12*3,8*2), dpi=100)
#         ytick = np.arange(1,len_species+2,2)
#         ylab = np.arange(1,len_species+2,2)
#     if proj == 'plant':
#         length = len_species
#         el = species_sorted
#         plot_str = 'ABCDEFGHIJKLMNOP'
#         fig, axs = plt.subplot_mosaic('ABCD\nEFGH\nIJKL\nMNOP', figsize=(12*4,8*4), dpi=100)
#         ytick = np.arange(1,len_functions+2,2)
#         ylab = np.arange(1,len_functions+2,2)
   
   
#     fig.subplots_adjust(wspace=.15,top=1.03,hspace=.7)


#     for i, elem in enumerate(el):

#         #print(elem)
       
#         if proj =='function':
#             p, p_t= p_matrix_elem(df_matrix_s, elem = [elem] , axis=1)
#             A = np.matmul(p,p_t)  
#         if proj == 'plant':
#             p, p_t= p_matrix_elem(df_matrix_s, elem = [elem] , axis=0)
#             A = np.matmul(p_t,p)  
       

#         if proj == 'function':
   
#             labels =  r'$P^i_{' + dic_functions[elem] +r'} P^j_{' + dic_functions[elem] +r'}$'
#         if proj == 'plant':

#             labels =  r'$P^i_{' + dic_species[elem] +r'} P^j_{' + dic_species[elem] +r'}$'
       

           
#         G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)  
#         dic_edges = {(i,j):G[i][j]["weight"] for i,j in G.edges }
       
#         ascending_color = 'darkred'
#         descending_color = 'navy'
       
#         item = plot_str[i]
#         axs1 = axs[item]
#         for row in [0,1]:
#             edges = []
#             weight = []
#             for ielem in dic_edges:
#                 edges.append(ielem)
#                 weight.append(dic_edges[ielem])
#             if row ==1:
#                 sort = np.argsort(np.array(weight))[::-1]
#             else:
#                 sort = np.argsort(np.array(weight))
               
#             weigth_sort = np.array(weight)[sort]
#             edges_sort = list(np.array(edges)[sort])
#             edges_sort = tuple(tuple(sub) for sub in edges_sort)
#             nodes = [k for k in G.nodes]
           
#             P = nx.Graph()      
#             P.add_nodes_from(nodes)
#             T = []
#             T_2 = []
#             T_2_2 = []
#             x = []
#             for edge in edges_sort:
               
#                 P.add_edge(edge[0], edge[1], weight = dic_edges[edge])
#                 #P.remove_edge(edge[0], edge[1])
#                 components = [c for c in sorted(nx.connected_components(P), key=len, reverse=True)]
#                 T.append(len(components[0]))
#                 if len(components) >1:
#                     #print(len(components[1]))
#                     length = np.array([len(k) for k in components[1:]])
#                     areas = length **2
#                     #T_2.append(np.mean(areas/ np.sum(areas)))
#                     T_2.append(np.mean(areas))
#                     T_2_2.append(len(components[1]))
#                     #T_2_2.append(areas[0]/np.sum(areas))
#                 else:
#                     T_2.append(0)
#                     T_2_2.append(0)
#                 x.append(dic_edges[edge]+np.sum(x))
       
       


#             if row==1:
#                 axs2 = axs1.twiny()
#                 axs2.cla()
               
               
#             if row == 1:
#                 axs2.scatter(weigth_sort,T, c=descending_color, s=100, alpha =.5)
#                 axs2.set_yticks(ytick)
#                 axs2.set_yticklabels(ylab, fontsize=fontsize)
#                 axs2.set_xticks(np.arange(0,1+.2,.2)[::-1])
#                 axs2.set_xticklabels([0,.2,.4,.6,.8,1][::-1], fontsize=fontsize)
#                 axs2.set_ylim(-.1, 17.1)  
#                 axs2.set_xlim(-.1,1.1)
#                 axs2.invert_xaxis()
#                 #axs2.xaxis.label.set_color(descending_color)        #setting up X-axis label color
#                 axs2.tick_params(axis='x', colors=descending_color)    #setting up X-axis tick color
#                 for pos in ['right', 'bottom']:
#                     axs2.spines[pos].set_visible(False)
#                 # for axis in ['left', 'top']:
#                 #     axs2.spines[axis].set_linewidth(5)
#                 #axs2.set_xlabel(descending_labels, fontsize=fontsize, color='k', labelpad=20)
#                 axs2.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#             else:
#                 axs1.scatter(weigth_sort,T, c=ascending_color, s=100, alpha =.5)
#                 axs1.set_yticks(np.arange(1,len(species)+2,2))
#                 axs1.set_yticklabels(np.arange(1,len(species)+2,2), fontsize=fontsize)
#                 axs1.set_xticks(np.arange(0,1+.2,.2))
#                 axs1.set_xticklabels([0,.2,.4,.6,.8,1], fontsize=fontsize)
#                 axs1.set_ylim(-.1,17.1)  
#                 axs1.set_xlim(-.1,1.1)
#                 axs1.xaxis.label.set_color(ascending_color)        #setting up X-axis label color
#                 axs1.tick_params(axis='x', colors=ascending_color)    #setting up X-axis tick color
#                 axs1.tick_params(axis='both', which='major', labelsize=fontsize, width=5, length=15)
#                 for pos in ['right', 'top']:
#                     axs1.spines[pos].set_visible(False)
#                 # for axis in ['bottom','left']:
#                 #     axs1.spines[axis].set_linewidth(5)

#                 axs1.set_title(elem.title(),{'fontsize':fontsize, 'fontweight':"bold"}, pad=20)
               
#             if row ==0:
#                 if i == 1 or i==4:
#                     axs1.set_ylabel('Size of largest component', fontsize=fontsize)
#                     axs1.set_xlabel('Adding edges in ascending weight order, '+labels, fontsize=fontsize, color='k', labelpad=17)
#                 else:
#                     axs1.set_xlabel(labels, fontsize=fontsize, color='k', labelpad=17)
#             else:
#                 if i == 1 or i==4:
#                     axs2.set_xlabel('Removing edges in descending weight order, '+labels, fontsize=fontsize, color='k', labelpad=17)
#                 else:
#                     axs2.set_xlabel(labels, fontsize=fontsize, color='k', labelpad=17)
           
     
#     #plant.savefig(plt_dir + f'/percolation_ad_elim_{proj}.pdf', bbox_inches='tight' ) #save


# #%%
#%%
sorted_indices_low = np.argsort(P_matrix, axis=None)
sorted_indices_up = np.flip(sorted_indices_low)
 
fontsize = 25  


for proj in ['plant', 'function']:
    print(proj)
    P_low = np.copy(P_matrix)
    P_up = np.copy(P_matrix)

    if proj=='function':
        row=0
        A = np.matmul(P_matrix,P_matrix_t)
    else:
        row=1
        A = np.matmul(P_matrix_t,P_matrix) 



    G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
    Y0 = len(max(nx.connected_components(G), key=len))
    #print(Y0)
    # plt.figure()
    # plt.imshow(A)
    # plt.show()

    fig,ax = plt.subplots(figsize=(12,8), dpi=300)
    legend_elements = []
    #edge=0
    T_low = []
    T_2_low = []
    T_2_2_low = []
    x_low = []

    T_up = []
    T_2_up = []
    T_2_2_up = []
    x_up = []
    for index_low, index_up in zip(sorted_indices_low, sorted_indices_up):

        lowest_weight_index = np.unravel_index(index_low, P_matrix.shape)
        edge_low = P_matrix[lowest_weight_index]
        #print(edge_low)
        P_low[lowest_weight_index] = 0
        P_low_t = np.transpose(P_low)


        highest_weight_index = np.unravel_index(index_up, P_matrix.shape)
        edge_up = P_matrix[highest_weight_index]
        #print(edge_up)
        P_up[highest_weight_index] = 0
        P_up_t = np.transpose(P_up)

        if proj == 'function':
            A_low = np.matmul(P_low,P_low_t)  
            A_up = np.matmul(P_up,P_up_t)  
        else:
            A_low = np.matmul(P_low_t,P_low)
            A_up = np.matmul(P_up_t,P_up)

        G_low = nx.from_numpy_array(A_low, parallel_edges=False, create_using=None)
        G_up = nx.from_numpy_array(A_up, parallel_edges=False, create_using=None)

        components_low = [c for c in sorted(nx.connected_components(G_low), key=len, reverse=True)]
        components_up = [c for c in sorted(nx.connected_components(G_up), key=len, reverse=True)]
        
        
        T_low.append(len(components_low[0]))
        if len(components_low) >1:
            length = np.array([len(k) for k in components_low[1:]])
            areas = length **2
            T_2_low.append(np.mean(areas))
            T_2_2_low.append(len(components_low[1]))
        else:
            T_2_low.append(0)
            T_2_2_low.append(0)
        x_low.append(edge_low+np.sum(x_up))

        T_up.append(len(components_up[0]))
        if len(components_up) >1:
            length = np.array([len(k) for k in components_up[1:]])
            areas = length **2
            T_2_up.append(np.mean(areas))
            T_2_2_up.append(len(components_up[1]))
        else:
            T_2_up.append(0)
            T_2_2_up.append(0)
        x_up.append(edge_up+np.sum(x_low))
           
        pc_low = np.max(np.array(T_2_low))
        T_low = T_low
        
        color_low = 'darkred'
        marker_low = "v"

        pc_up = np.max(np.array(T_2_low))
        T_up = T_up
        color_up = 'navy'
        marker_up = "^"
           
               
    y_low = np.array([Y0] + T_low) / max([Y0] + T_low)
    x_low = np.arange(0,len(y_low),1)
    x_low = x_low / x_low.max()
    dx_low = x_low[2] - x_low[1]

    y_up =  np.array([Y0] + T_up) / max([Y0] + T_up)
    x_up = np.arange(0,len(y_up),1)
    x_up = x_up / x_up.max()
    dx_up = x_up[2] - x_up[1]


    auc = lambda x,dx: dx * (x[1:-1].sum()) + dx/2 *(x[0] + x[-1])
    Ax_low = auc(y_low, dx_low)
    Ax_up = auc(y_up, dx_low)

    area_low = scipy.integrate.simpson(y_low,dx=dx_low, even='avg')

    area_up = scipy.integrate.simpson(y_up,dx=dx_up, even='avg')

    #     # if row ==0 :
    #     #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'
    #     # if row ==1 :
    #     #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'


    label_low  = f'Ascending order, AUC ={round(area_low, 2)}'

    label_up  = f'Descending order, AUC = {round(area_up, 2)}'
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label_up,
                       markerfacecolor=color_up, markersize=10, linestyle='none', markeredgecolor = color_up))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label_low,
                       markerfacecolor=color_low, markersize=10, linestyle='none', markeredgecolor = color_low))
    plt.fill_between(x_low, y_low, y2=0, color = color_low, alpha = .2)
    plt.scatter(x_low, y_low, c = color_low)
    plt.fill_between(x_up, y_up, y2=0, color = color_up, alpha = .2)
    plt.scatter(x_up, y_up, c = color_up)

    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
    ax.set_xticks(np.arange(0,1.2,.2))
    ax.set_xlabel(r'$m/M$'+'\n'+'Fraction of pruned edges', fontsize= fontsize)
    ax.set_xticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(np.arange(0,1.2,.2))
    ax.set_ylabel(r'$S(m)/n$'+'\n'+'(normalized) size of LCC', fontsize= fontsize)
    ax.set_yticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    for pos in [ 'top', 'right']:
        ax.spines[pos].set_visible(False) 
    ax.set_xlim([0-.02,x_low[-1]+.02])
    ax.set_ylim([0-.05,x_low[-1]+.05])


    # #print(legend_elements)
    ax.legend(handles=legend_elements,fontsize= fontsize, frameon=False, loc = 'lower left')
    plt.savefig(plt_dir + f'/percolation_{proj}_nueva.pdf', bbox_inches='tight' ,  dpi=300)
    plt.savefig(plt_dir + f'/percolation_{proj}_nueva.png', bbox_inches='tight' ,  dpi=300)

#%%
sorted_indices_low = np.argsort(P_matrix, axis=None)
sorted_indices_up = np.flip(sorted_indices_low)
 
fontsize = 25  


for proj in ['plant', 'function']:
    print(proj)
    P_low = np.copy(P_matrix)
    P_up = np.copy(P_matrix)

    if proj=='function':
        row=0
        A = np.matmul(P_matrix,P_matrix_t)
    else:
        row=1
        A = np.matmul(P_matrix_t,P_matrix) 



    G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)
    Y0 = len(max(nx.connected_components(G), key=len))
    #print(Y0)
    # plt.figure()
    # plt.imshow(A)
    # plt.show()

    fig,ax = plt.subplots(figsize=(12,8), dpi=300)
    legend_elements = []
    #edge=0
    T_low = []
    T_2_low = []
    T_2_2_low = []

    T_up = []
    T_2_up = []
    T_2_2_up = []

    # propuesta eje X Gorka
    x_low = [0]
    x_up = [0]

    for index_low, index_up in zip(sorted_indices_low, sorted_indices_up):

        lowest_weight_index = np.unravel_index(index_low, P_matrix.shape)
        edge_low = P_matrix[lowest_weight_index]
        # add to the list
        x_low.append(x_low[-1] + edge_low)
        #print(edge_low)
        P_low[lowest_weight_index] = 0
        P_low_t = np.transpose(P_low)


        highest_weight_index = np.unravel_index(index_up, P_matrix.shape)
        edge_up = P_matrix[highest_weight_index]
        # add to the list
        x_up.append(x_up[-1] + edge_up)
        #print(edge_up)
        P_up[highest_weight_index] = 0
        P_up_t = np.transpose(P_up)

        if proj == 'function':
            A_low = np.matmul(P_low,P_low_t)  
            A_up = np.matmul(P_up,P_up_t)  
        else:
            A_low = np.matmul(P_low_t,P_low)
            A_up = np.matmul(P_up_t,P_up)

        G_low = nx.from_numpy_array(A_low, parallel_edges=False, create_using=None)
        G_up = nx.from_numpy_array(A_up, parallel_edges=False, create_using=None)

        components_low = [c for c in sorted(nx.connected_components(G_low), key=len, reverse=True)]
        components_up = [c for c in sorted(nx.connected_components(G_up), key=len, reverse=True)]
        
        
        T_low.append(len(components_low[0]))
        if len(components_low) >1:
            length = np.array([len(k) for k in components_low[1:]])
            areas = length **2
            T_2_low.append(np.mean(areas))
            T_2_2_low.append(len(components_low[1]))
        else:
            T_2_low.append(0)
            T_2_2_low.append(0)

        T_up.append(len(components_up[0]))
        if len(components_up) >1:
            length = np.array([len(k) for k in components_up[1:]])
            areas = length **2
            T_2_up.append(np.mean(areas))
            T_2_2_up.append(len(components_up[1]))
        else:
            T_2_up.append(0)
            T_2_2_up.append(0)
           
        pc_low = np.max(np.array(T_2_low))
        T_low = T_low
        
        color_low = 'darkred'
        marker_low = "v"

        pc_up = np.max(np.array(T_2_low))
        T_up = T_up
        color_up = 'navy'
        marker_up = "^"
           
               
    y_low = np.array([Y0] + T_low) / max([Y0] + T_low)
    x_low = np.array(x_low)
    x_low = x_low / x_low[-1]
    dx_low = np.diff(x_low)

    y_up =  np.array([Y0] + T_up) / max([Y0] + T_up)
    x_up = np.array(x_up)
    x_up = x_up / x_up[-1]
    dx_up = np.diff(x_up)

 
    def calcular_area(x, y):
        dx = np.diff(x)  # Calcula los intervalos dx a partir de los valores de x
        area = np.sum(dx * y[1:])  # Calcula el área sumando el producto de dx y los valores de y
        return area
    Ax_low = auc(x_low, y_low)
    Ax_up = auc(x_up, y_up)

    area_low = scipy.integrate.simpson(y_low,x=x_low, even='avg')

    area_up = scipy.integrate.simpson(y_up,x=x_up, even='avg')

    #     # if row ==0 :
    #     #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'
    #     # if row ==1 :
    #     #     label  = f'Removing edges in ascending w order. AUC = {round(Ax, 2)}'


    label_low  = f'Ascending order, AUC ={round(area_low, 2)}'

    label_up  = f'Descending order, AUC = {round(area_up, 2)}'
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label_up,
                       markerfacecolor=color_up, markersize=10, linestyle='none', markeredgecolor = color_up))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=label_low,
                       markerfacecolor=color_low, markersize=10, linestyle='none', markeredgecolor = color_low))
    plt.fill_between(x_low, y_low, y2=0, color = color_low, alpha = .2)
    plt.scatter(x_low, y_low, c = color_low)
    plt.fill_between(x_up, y_up, y2=0, color = color_up, alpha = .2)
    plt.scatter(x_up, y_up, c = color_up)

    ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
    #ax.set_xticks(np.arange(0,1.2,.2))
    ax.set_xlabel(r'$P_i^{\alpha}(removed)/P_i^{\alpha}(0)$', fontsize= fontsize)
    #ax.set_xticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_yticks(np.arange(0,1.2,.2))
    ax.set_ylabel(r'$S(m)/n$'+'\n'+'(normalized) size of LCC', fontsize= fontsize)
    ax.set_yticklabels(["0", '0.2', '0.4', '0.6', '0.8', '1.0'])
    for pos in [ 'top', 'right']:
        ax.spines[pos].set_visible(False) 
    #ax.set_xlim([0-.02,x_low[-1]+.02])
    #ax.set_ylim([0-.05,x_low[-1]+.05])
    #
    # ax.set_xscale('log')


    # #print(legend_elements)
    ax.legend(handles=legend_elements,fontsize= fontsize, frameon=False, loc = 'lower left')
    #plt.savefig(plt_dir + f'/percolation_{proj}_removing_strength.pdf', bbox_inches='tight' ,  dpi=300)
    #plt.savefig(plt_dir + f'/percolation_{proj}_removing_strength.png', bbox_inches='tight' ,  dpi=300)


#%%


P_try = np.array([[0, 2, 3], [4, 0, 1], [2, 5, 0]])  # Example adjacency matrix

print(P_try)
# Get the indices that would sort the array in ascending and descending order
sorted_indices_low = np.argsort(P_try, axis=None)
sorted_indices_up = np.flip(sorted_indices_low)

P_low = np.copy(P_try)
print(P_low)
print('----')
P_up = np.copy(P_try)

for index_low, index_up in zip(sorted_indices_low, sorted_indices_up):
    # Prune edge with the lower weight in P_low
    #print('----')
    #print(index_low)
    lowest_weight_index = np.unravel_index(index_low, P_try.shape)
    #print(lowest_weight_index)
    P_low[lowest_weight_index] = 0

    # Prune edge with the higher weight in P_up
    highest_weight_index = np.unravel_index(index_up, P_try.shape)
    P_up[highest_weight_index] = 0

    #print("P_low:")
    #print(P_low)
    #print("P_up:")
    #print(P_up)
    #print("---------")


#%%
'''Entropy and Hill number'''

def calc_entropy(partial_strength):
    x = partial_strength[partial_strength > 0]
    N = len(x)
    r = len(partial_strength) - N
   
    H = - (x * np.log(x)).sum()
    n = np.exp(H)
    return H, N, r, n

def hill_number(distribution, q):
    if q >= 0 and q != 1:
        D = ((distribution**q).sum()) **(1 / (1 - q))
    elif q == 1:
        D = calc_entropy(distribution)[3]
    return D
   
denominator  = np.sum(stregnth_fn_rank) #basicamente vamos a doble normalizar strength_fn_rank
#Ahora mismo está hecho de tal manera que para cada planta suma 1. Pero las barras de las funciones en la FDigura 5 no suman1,
# suman 16 (numero de especies). Vamos a hacer una normalización porque necesitamos una distribución para los Hill Number
pi_distribution = np.array([i/j for i,j in zip(stregnth_fn_rank, np.sum(stregnth_fn_rank, axis = 1))])
#Ahora queremos la distribución para las funciones, la desagregación por plantas nos da igual. LO que hacemos es sumar las barras



fontsize = 40
q_sample = [1,2,3]
ncols =3
plot_str, plot_shape, nrows= plot_shp(len(q_sample),ncols)
plot_shape = plot_shape
df_Hill_numbers = pd.DataFrame(columns = ['q','function','Hill Number'])              
fig, axs = plt.subplot_mosaic('ABC', figsize=(12*ncols,8), dpi=300)
fig.subplots_adjust(wspace=.24)  

for i in range(len(q_sample)):
    q = q_sample[i]
    item = plot_str[i]
    Hill_numbers = pd.DataFrame(columns = ['q','function', 'Hill Number'])

    for function in range(len(function_sorted)):
        D = hill_number(pi_distribution[function], q)        # Hill num of p_i = k_i^alpha / o_i of order q
        new_row = [q,function_sorted[function],D]
        Hill_numbers.loc[function] = new_row

    Hill_numbers = Hill_numbers.sort_values('Hill Number', ascending = False).reset_index(drop=True)
   
   
    for index, row in Hill_numbers.iterrows():
        #axs[item].bar(index, row['Hill Number'],color= dic_colors_fn[row['function']], alpha =.5)
        axs[item].bar(index, row['Hill Number'],color= dic_colors_fn[row['function']])
    axs[item].set_xticks(np.arange(0,len_functions))
    axs[item].set_xticklabels(labels=[dic_functions[x] for x in Hill_numbers['function']], fontsize = fontsize)
    axs[item].set_yticks(np.arange(0,10+2,2))
    axs[item].set_yticklabels(np.arange(0,10+2,2),fontsize = fontsize)
    axs[item].set_xlim(-0.5, len(functions)-0.5)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)  
    axs[item].set_ylim(0, 10)
    axs[item].set_ylabel(rf'Hill number ($^{q}D$)', fontsize = fontsize)
    axs[item].spines['right'].set_visible(False)
    axs[item].spines['top'].set_visible(False)
   
    # for axis in ['bottom','left']:
    #     axs[item].spines[axis].set_linewidth(4)
   
    axs[item].tick_params(axis='both', which='major', labelsize=fontsize, width=4, length=20)
   
    df_Hill_numbers= pd.concat([df_Hill_numbers, Hill_numbers], ignore_index=True)
   
    i+=1
   
#plant.savefig(plt_dir + '/Hill_numbers.pdf', bbox_inches='tight',  dpi=300 )
df_Hill_numbers.to_csv(df_dir + '/Hill_numbers.csv',index=False)





#%%


'''3d plot'''


def func_cmap(att_list,dic_colors_fn={},cmp = 'Reds'):
     
    '''Returns a colormap dic based on the palette 'plt.cm.rainbow',
       discretized into the number of (different) colors
       found in the attribute.
   
    Parameters
    ----------
    att_list: attributes list
   
    Returns
    ----------
    cmp_edges: cmap
        Cmap of edges.
   
    colors: array
        array of edges attributs in G.edges order.
       
    func_to_color: dict
        Dict object attr: color.
       
    '''
     

    N = 256
   
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(90/256, 1, N)
    vals[:, 1] = np.linspace(39/256, 1, N)
    vals[:, 2] = np.linspace(41/256, 1, N)
    SPF_cmp  = ListedColormap(vals)

    corr = {'tab:purple': (148/256,103/256,189/256),
            'tab:orange': (255/256,127/256,14/256),
            'tab:green':  (44/256,160/256,44/256),
            'tab:brown':  (140/256,86/256,75/256),
            'tab:olive':  (188/256,189/256,34/256),
            'tab:blue':   (31/256,119/256,180/256),
            'tab:pink':   (227/256,119/256,194/256),
            'tab:grey':   (127/256,127/256,127/256)}

    if dic_colors_fn == {'herbivory': 'tab:purple',
         'plant': 'tab:green',
         'pollination': 'tab:orange',
         'decomposition': 'tab:brown',
         'fungal pathogenicity': 'tab:olive',
         'seed dispersal': 'tab:blue',
         'nutrient uptake': 'tab:pink',
        'total': 'tab:grey'}:
        new = {}
        for elem in list(dic_colors_fn.keys()):
            new[elem] = corr[dic_colors_fn[elem]]
        dic_colors_fn = new


   
    #Build a sequential cmap for each function color that goes from the function color to white
    if len(dic_colors_fn)!=0:
       
        func_to_cmp={}
        for function in function_sorted:

            color = dic_colors_fn[function]
            vals = np.ones((N, 3))
            vals[:, 0] = np.linspace(color[0], 1, N)
            vals[:, 1] = np.linspace(color[1], 1, N)
            vals[:, 2] = np.linspace(color[2], 1, N)
            cmp  = ListedColormap(vals)
            func_to_cmp[function]= cmp
   
    # If not, used the input colormap
    else:
        func_to_cmp    = {i:'Reds_9' for i in att_list}
   
    return func_to_cmp



def threedplot_sort_speces(head, separator, data_path, df_dir, matrix_order):
   
    df = pd.read_csv(data_path,  header= head, sep = separator) # original dataset with frequencies #str(argv[0])
    species = matrix_order['plant_sp'].to_list()
    species_order = {}
   
    for i in range(len(species)):
        species_order[species[i]] = len(species) -i
       
    df['plant_sp order'] = [species_order[i] for i in df['plant_sp']]
    functions = matrix_order.columns[1:].to_list()
   
    functions_order = {}
    for i in range(len(functions)):
        functions_order[functions[i]] =len(functions) -  i
       
    df['interaction_type order'] = [functions_order[i] for i in df['interaction_type']]
   
   
    animals_order_arr = []
    animals_order = {}
   
    last = 0
    for function in functions:
       
        animals_fun = list(set(df['animal_sp'][df['interaction_type']  == function].to_list()))
        animals_fun = [x for x in animals_fun if x not in animals_order_arr]
        animals_av_freq = []
   
        for animals_fun_i in animals_fun:
           
            average_freq = np.sum(df['frequency_of_occurrence_weighted_by_sampling_effort'][df['animal_sp'] == animals_fun_i].to_numpy())
            animals_av_freq.append(average_freq)
           
        animals_df = pd.DataFrame()
        animals_df['animals'] = animals_fun
        animals_df['average freq'] = animals_av_freq
        animals_df = animals_df.sort_values(by=['average freq'], ascending =False)
        animals_df['order'] = np.arange(last+1,len(animals_df)+1+last)
        #print(last+1, len(animals_df)+1+last)
        animals_order_arr.append(animals_fun)
       
        for i in animals_df['animals'].to_list():
            animals_order[i] = animals_df['order'][animals_df['animals'] == i].to_list()[0]
           
        last += len(animals_df)
       
       
   
    df['animal_sp order'] = [animals_order[i] for i in df['animal_sp']]
    #df_sort = df.sort_values(by=['interaction_type order', 'plant_sp order']).reset_index()
    df_sort = df.sort_values(by=['interaction_type order', 'plant_sp order']).reset_index()
    df_sort = df_sort.drop(['interaction_type order', 'plant_sp order', 'animal_sp order'], axis=1)
   

    animals_array = [animals_order[i] for i in df_sort['animal_sp']]
    animals_axis  = [i for i in animals_order]
    species_array = [species_order[i] for i in df_sort['plant_sp']]
    species_axis  = [i for i in species_order]
    functions_array = [functions_order[i] for i in df_sort['interaction_type']]
    functions_axis = [i for i in functions_order][::-1]

    interaction_array = df_sort['frequency_of_occurrence_weighted_by_sampling_effort'].to_list()
   
    return species_array, species_axis, functions_array, functions_axis, animals_array, animals_axis, interaction_array


func_to_cmp = func_cmap(functions,dic_colors_fn, cmp = 'Reds')
species_array, species_axis, functions_array, functions_axis, animal_array, animal_axis, interaction_array = threedplot_sort_speces(head, separator, data_path, df_dir, P_matrix_df)
#from mpl_toolkits.mplot3d.axes3d import Axes3D


def legend_cmap(att_list):
   
    legend_elements= []
   
    if Counter(att_list) == Counter(['decomposition','fungal pathogenicity','herbivory','nutrient uptake','pollination','seed dispersal']):
       
        for fun in att_list:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=fun.capitalize(), markerfacecolor=dic_colors_fn[fun], markersize=15))

    else:

        for i in range(len(functions_axis))[::-1]:
            cmap = func_to_cmp[functions_axis[i]]
            cmap = mpl.cm.get_cmap(cmap)
            color = cmap(0)
           
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=functions_axis[i].capitalize(), markerfacecolor=list(color)[:3], markersize=15))

    return legend_elements
   

legend_elements = legend_cmap(functions_axis[::-1])


fontsieze= 22




x = species_array
y = animal_array
z = functions_array
c = np.log(np.array(interaction_array))

fig = plt.figure(figsize=(12,12), dpi=300)
ax = fig.add_subplot(projection='3d')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#bbox = fig.bbox_inches.from_bounds(1, 1, 12, 8)
#ax = plt.axes(projection='3d')
labels_p = []
for i in P_matrix_df['plant_sp'].to_numpy():
    labels_p.append(dic_species[i])
sctt = ax.scatter3D(xs=x, ys=y, zs=z, c=c, cmap='viridis', s=50)
#ax.scatter3D(xs=x, ys=y, zs=z, c=c, cmap=Blues_9.mpl_colormap, s=50)
ax.set_xticks(np.arange(0,len(species_axis)),labelpad = -.5)
ax.set_xlabel('Plant species',labelpad = -.5,fontsize=fontsieze)
ax.set_ylabel('Animal/fungi species',labelpad = -5,fontsize=fontsieze)
ax.set_zticks(np.arange(0,len(functions_axis)), labelpad=1000)
ax.set_zticklabels( [dic_functions[i] for i in functions_axis],fontsize=fontsieze)
ax.view_init(elev =10, azim=10)
ax.set_yticks([])
ax.set_yticklabels([], fontdict={'fontsize': 0})
ax.set_xticks([])
ax.set_xticklabels([], fontdict={'fontsize': 0})
ax.view_init(10, 5)
axins = inset_axes(ax,
                    width="5%",  
                    height="50%",
                    loc='right',
                    borderpad=3
                   )
cb = fig.colorbar(sctt, shrink = 0.8, aspect = 15, ticks = [-8,-6,-4,-2,0], cax=axins, orientation="vertical")
cb.ax.tick_params(labelsize=fontsieze,width = 3, length = 8)
cb.set_label('Participation strength', labelpad=30,rotation=-90, fontsize = fontsieze)
#plant.savefig(plt_dir + '/3d_plot.pdf', bbox_inches='tight',  dpi=300 ) #save




fontsize=20
fig, ax = plt.subplots(figsize= (12,8), dpi=300)
x = np.array(x)
y = np.array(y)
z = np.array(z)
for i in range(1,len(functions_axis)+1)[::-1]:
    ax.scatter(y[z==i], x[z==i], c=c[z==i],  cmap=func_to_cmp[functions_axis[i-1]], vmin = c.min(), vmax = c.max(), label=functions_axis[i-1], marker='|',linewidths=1)
    #ax.scatter(y[z==i], x[z==i], c=c[z==i], cmap=func_to_cmp[functions_axis[i-1]].mpl_colormap, vmin = c.min(), vmax = c.max(), label=functions_axis[i-1])
ax.legend(handles=legend_elements,loc='upper center', frameon=False, fontsize=fontsieze,bbox_to_anchor=(0.43, 1.18), ncol=3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# for axis in ['bottom','left']:
#     ax.spines[axis].set_linewidth(2)
ax.set_xticks([])
#ax.set_xticklabels(np.arange(0, 1800,200), minor=False, fontsize=fontsize)
ax.set_yticks(np.arange(1,len_species+1,1))
ax.set_yticklabels(labels_p[::-1], minor=False, fontsize=fontsieze)
ax.set_xlabel('Animal/fungi species', fontsize=fontsieze)
ax.tick_params(axis='both', which='major', labelsize=fontsieze, width=1, length=12)
 
#plant.savefig(plt_dir + '/3d_plot_proy_log_scale.pdf', bbox_inches='tight',  dpi=300 ) #save


#%%

'''Autovalores'''  

def Laplacian(functions, P_matrix, P_matrix_t):
   
    lamda_2 = []
    rlxT = []
    lamdas_r=[]
    for function in ['total'] + function_sorted:
       
        if function == 'total':
            A = np.matmul(P_matrix, P_matrix_t)
        else:
            P_matrix_df_func = P_matrix_df[['plant_sp', function]] #drop columns of cover and abundance
            P_matrix_func    = P_matrix_df_func.values[:, 1:].astype(float)
            P_matrix_t_func  = np.transpose(P_matrix_df_func.values[:, 1:].astype(float))
            A = np.matmul(P_matrix_func, P_matrix_t_func)

        L, Ldiag = csgraph.laplacian(A,return_diag=True)
        eigenvals, eign = np.linalg.eig(L)
        lamdas = np.sort(eigenvals) #los ordenamos de menor a mayor
        lamdas_r.append(lamdas)
        lamdas = lamdas[np.abs(lamdas)>10**(-9)]
       
        lamda_2.append(lamdas[0])
        rlxT.append( 1/lamdas[0]) #nos quedamos con el segundo menor
       
    return lamda_2, rlxT, lamdas_r


lamda_2, rlxT, lamdas = Laplacian(function_sorted, P_matrix, P_matrix_t)


label = ['Multifunctional'] + [fun.capitalize() for fun in function_sorted]
colors = [dic_colors_fn[function] for function in ['total'] + function_sorted]
               
fontsize=18
fig, ax = plt.subplots(figsize= (12,8), dpi=300)
for i in range(len(colors))[::-1]:
    lambda_i = lamdas[i][lamdas[i] >10**(-9)]
    #ax.scatter( lambda_i,np.ones(len(lambda_i))*(len(colors)-1-i), c=colors[i], s=100, label=label[i], alpha =.5)
    ax.scatter( lambda_i,np.ones(len(lambda_i))*(len(colors)-1-i), c=np.array([colors[i]]), s=100, label=label[i])
ax.set_yticks(np.arange(0,len(colors)))
ax.set_yticklabels(label[::-1], minor=False, fontsize=fontsize)
ax.set_xlabel(r'$\lambda$', fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
# for axis in ['bottom','left']:
#     ax.spines[axis].set_linewidth(3)
ax.set_xscale('log')
 
#plant.savefig(plt_dir + '/lambdas.pdf', bbox_inches='tight' ,  dpi=300 ) #save

#%%
'''Maximum spanning treee'''



def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)





fontsize = 18

abundance_p = []
for i in P_matrix_df['plant_sp'].to_numpy():
    #abundance_p.append((np.log(df_matrix[df_matrix['plant_sp'] == i ]['abundance'].iloc[0]+1))*200)
    abundance_p.append(   (    df_matrix[df_matrix['plant_sp'] == i ]['abundance'].iloc[0]   /   np.max(df_matrix['abundance'].to_numpy())        )*1000 +50)
    #abundance_p.append(df_matrix[df_matrix['plant_sp'] == i ]['abundance'].iloc[0]*100)
     
A = np.matmul(P_matrix, P_matrix_t)      
G = nx.from_numpy_array(A, parallel_edges=False, create_using=None)  
tree = nx.maximum_spanning_tree(G, weight='weight')
widths = np.array([d['weight'] for u,v,d in tree.edges(data=True)])
vmin = my_floor(np.min(widths), precision=1)
vmax = my_ceil(np.max(widths), precision=1)
pos = nx.nx_agraph.graphviz_layout(tree,prog="twopi")
lb = {i: n for i, n in enumerate(labels_p)}
fig, ax = plt.subplots(figsize=(12,8), dpi=300)
nx.draw_networkx_nodes(tree, pos=pos, node_size=abundance_p,node_color="tab:green", node_shape="o", alpha=0.6, linewidths=4, ax=ax)
shw = nx.draw_networkx_edges(tree, pos=pos, edge_color = [d["weight"] for u,v,d in tree.edges(data=True)], edge_cmap=plt.cm.viridis, edge_vmax=vmax, width=2, ax=ax, edge_vmin=vmin)
nx.draw_networkx_labels(tree, pos=pos, labels= lb, ax=ax, font_size=fontsize)
# cbar = fig.colorbar(edges)
# cbar.ax.tick_params(labelsize=17,width = 2, length = 8)
#colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(shw, cax=cax, ticks= [my_floor(a, precision = 2) for a in np.linspace(vmin, vmax,5)])
cb.ax.tick_params(labelsize=fontsize,width = 1, length = 8)
#ticks
for t in cb.ax.get_yticklabels():
     t.set_fontsize(fontsize)    
ax.set_title('Multifunctional',{'fontsize':fontsize})
ax.set_frame_on(False)
cb.outline.set_edgecolor('black')
xlim =ax.get_xlim()
ylim =ax.get_ylim()
 
#plant.savefig(plt_dir + '/tree_total.pdf', bbox_inches='tight',  dpi=300 ) #save

#%%



'''Likelihood'''

#%%
   

def Likelihood(df_matrix, species,len_species):
    df_likelihood = df_matrix[['plant_sp', 'abundance', 'cover']].copy()
    likelihood = np.zeros(len_species)
    number_zeros = np.zeros(len_species, dtype =int)
   
    for row, index in zip(df_matrix.loc[:, functions].to_numpy(), df_matrix.index):
   
        likelihood_index = 0
        number_zeros_index = 0
       
        for elem in row:
            if elem !=0:
                likelihood_index += np.log(elem)  
            else:
                number_zeros_index +=1
           
        likelihood[index] = likelihood_index
        number_zeros[index] = number_zeros_index
       
    df_likelihood['number_zeros'] = number_zeros
    df_likelihood['likelihood'] = likelihood
   
   
    df_likelihood['abundance_ranking'] = df_likelihood['abundance'].rank(method = 'average',ascending=False)
    df_likelihood['cover_ranking'] = df_likelihood['cover'].rank(method = 'average', ascending=False)
   
    return df_likelihood

df_likelihood= Likelihood(df_matrix, species,len_species)

def Likelihood_by_zeros(df_likelihood, functions,len_species,len_functions):
   

    number_zeros = np.arange(0,len_functions, 1)
    likelihood_zeros_plant = []
    likelihood_zeros = []
    likelihood_zeros_ax = []
    last=0
    for n_zero in number_zeros:
       
        df_plants_n_zeros = df_likelihood.loc[df_likelihood['number_zeros'] == n_zero]
        df_plants_n_zeros  = df_plants_n_zeros.sort_values('likelihood', ascending= False)
        likelihood_n_zero =df_plants_n_zeros['likelihood'].to_list()
        likelihoof_n_zero_plant = df_plants_n_zeros['plant_sp'].to_list()
        likelihood_n_zero_ax = np.arange(last+1, len(likelihood_n_zero)+1+last).tolist()
       
       
        likelihood_zeros_plant.append(likelihoof_n_zero_plant)
        likelihood_zeros.append(likelihood_n_zero)
        likelihood_zeros_ax.append(likelihood_n_zero_ax)
        last += len(df_plants_n_zeros)
   
    likelihood_rankin = [item for sublist in likelihood_zeros for item in sublist]
    rankin = np.arange(1, len_species+1,1)
    df_likelihood = df_likelihood.set_index('likelihood').loc[likelihood_rankin].reset_index()
    df_likelihood['likelihood_ranking'] = rankin
   
    abundance_ranking_inv = df_likelihood['abundance'].rank(method = 'average',ascending=True)
    cover_ranking_inv = df_likelihood['cover'].rank(method = 'average', ascending=True)

    df_likelihood.to_csv(df_dir + '/Likelihood.csv',index=False)
       
    return number_zeros, df_likelihood, likelihood_zeros_ax,abundance_ranking_inv,cover_ranking_inv

number_zeros, df_likelihood, likelihood_zeros_ax,abundance_ranking_inv,cover_ranking_inv= Likelihood_by_zeros(df_likelihood, functions,len_species,len_functions)

corr_abundance_s = stats.spearmanr(df_likelihood['likelihood_ranking'].to_numpy(), df_likelihood['abundance_ranking'].to_numpy())
corr_cover_s    = stats.spearmanr(df_likelihood['likelihood_ranking'].to_numpy(), df_likelihood['cover_ranking'].to_numpy())


def legend_plot_like(number_zeros,likelihood_zeros_ax, pallette = sns.color_palette("viridis_r", as_cmap=True)):

   
   

    plants_zeros = [len(zeros) for zeros in likelihood_zeros_ax]

    number_zeros = number_zeros[np.nonzero(plants_zeros)]
   
    colors  = mpl.colors.ListedColormap(pallette(np.linspace(0,1,len(number_zeros))))
   
    legend_elements = []
    #legend_elements.append(Patch(facecolor='w', edgecolor='w',label="$\\bf{Ranks}$"))
    #legend_elements.append(Line2D([0], [0], marker='P', color='y', label='Abundance', markerfacecolor='seagreen', markersize=50, linewidth=0,markeredgecolor='seagreen'))
    #legend_elements.append(Line2D([0], [0], marker=r'$\clubsuit$', color='y', label='Cover', markerfacecolor='seagreen', markersize=50, linewidth=0,markeredgecolor='seagreen'))
    #legend_elements.append(Patch(facecolor='w', edgecolor='w',label=''))
    #legend_elements.append(Patch(facecolor='w', edgecolor='w',label= "$\\bf{Function}$"+ '\n' + "$\\bf{counter}$"))
    legend_elements.append(Patch(facecolor='w', edgecolor='w',label= "Function"+ '\n' + "counter"))
   
    for zero in number_zeros:
        legend_elements.append(Patch(facecolor=colors.colors[zero], edgecolor=colors.colors[zero],label=str(len(plants_zeros) -zero)))              
                           
    return legend_elements, colors, number_zeros


legend_elements, colors, number_zeros = legend_plot_like(number_zeros,likelihood_zeros_ax)


scale = 3
fontsize =45
fig, ax = plt.subplots(figsize= (35,8), dpi=300)
ax = plt.gca()
ax.plot(np.arange(len_species+2), np.zeros(len_species+2), 'k-.')
#ax.plot(np.arange(len_species+2), np.zeros(len_species+2), 'k-.',linewidth=1)
for zero in number_zeros:
    ax.bar(likelihood_zeros_ax[zero],df_likelihood[df_likelihood['number_zeros']== zero]['likelihood'].to_numpy(), color = colors.colors[zero])
for row in df_likelihood['likelihood_ranking']:
    plant = df_likelihood['plant_sp'][df_likelihood['likelihood_ranking'] == row].to_numpy()[0]
    like = df_likelihood['likelihood'][df_likelihood['likelihood_ranking'] == row].to_numpy()[0]
    ax.text(row-0.3, like -0.7, item_words(plant, line_sep=False), fontsize =fontsize, rotation = 'vertical', verticalalignment = 'top')
   
ax.legend(handles =legend_elements,fontsize= fontsize,loc= 'center right', frameon=False, title_fontsize=fontsize,bbox_to_anchor = (1.2,0.2))
ax.set_yticks(np.arange(-10,6,4))
ax.set_xticks(np.arange(len_species+3))
ax.set_xlabel('Ranking', fontsize=fontsize)
ax.set_ylabel('Likelihood', fontsize=fontsize)
ax.set_xlim(0.5,len_species+1)
ax.get_xaxis().set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
# for axis in ['bottom','left']:
#     ax.spines[axis].set_linewidth(3)
ax.tick_params(axis='both', which='major', labelsize=fontsize, width=1, length=12)
 
#plant.savefig(plt_dir + '/Likelihood_new.pdf', bbox_inches='tight' ,  dpi=300)

#%%

	



