#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:28:13 2022

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
import scienceplots



def creat_dir(path):
    
    ''' Create folder for saving.
    
    Parameters
    ----------
    path: str
        Directoy path.
    
    Returns
    ----------
    df_dir: str 
        Path to the created folder to save output DataFrames.
    plt_dir: str 
        Path to the created folder to save output Plots.  
        
    '''
    
    directory  = os.path.dirname(path) ## directory of file
    result_dir = directory+ '/Results' # First folder
    os.makedirs(result_dir, exist_ok=True)# works also if the dir already exists
    df_dir     = result_dir + '/DataFrames' # First subfolder
    os.makedirs(df_dir, exist_ok=True)
    plt_dir    = result_dir + '/Plots'  # Second subfolder
    os.makedirs(plt_dir, exist_ok=True)
    
    return df_dir, plt_dir



def df_prob(head, separator, data_path, save_path, abundance_path = []):
    
    ''' Build probability dataframe.
    
    Parameters
    ----------
    head: str
        Header of datasets.
    separator: str
        Separator of datasets.
    data_path: str
        Path to dataset with frequencies of occurrence by sampling_effort.
        Columns must be ['plant_sp', 'animal_sp','frequency_of_occurrence_weighted_by_sampling_effort']. Additionally ['plant_sp',abundance, 'cover'].
    save_path: str
        Path to save output df.
    abundance_path: str
        Optional. Path to plant abundance and cover dataset. Default empty if abundance already included in data.
        Columns must be ['plant_sp',abundance, 'cover'].
    
    Returns
    ----------
    df: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'interaction_type', 'probability', 'abundance', 'cover'], 
        probabilities are obtained using: P_{i,alpha} = 1 - \prod_{j=1}^{j} (1 - f_{i,alpha,j}).
        Where P_{i,alpha} is the probability that plant species i participates in interaction_alpha. 
    
    '''
    #df.replace({'pathongen-fungi': 'fungal pathogenicity', 'symbiontic-fungi':'nutrient uptake', 'saprotrophic-fungi': 'decomposition'	}, regex=True)

    df = pd.read_csv(data_path,  header= head, sep = separator) # original dataset with frequencies #str(argv[0])

    
    if all(x in df.columns.to_list() for x in ['abundance', 'cover']):
        #aggregate directly with columns for abundance and cover and constructing the probabilities from the frequencies of occurrence
    	df = df.groupby(['plant_sp', 'interaction_type']).agg(probability=('frequency_of_occurrence_weighted_by_sampling_effort', lambda x: 1 - np.product(1 - x)), abundance = ('Estimated_number_of_individuals', np.mean), cover = ('species_cover (cm)', np.mean)).reset_index() #aggregated data with probabilities 
    
    else:
        #construct the probabilities from the frequencies of occurrence
        df = df.groupby(['plant_sp', 'interaction_type']).agg(probability=('frequency_of_occurrence_weighted_by_sampling_effort', lambda x: 1 - np.product(1 - x))).reset_index() #aggregated data with probabilities
        #We add two columns with the corresponding abundance and species cover of each plant sp
        abundance = pd.read_csv(abundance_path, header= head,  index_col=0) # abundance dataset 
        #initialization of columns
        abundance_col = np.zeros(len(df), dtype = int)
        cover_col     =  np.zeros(len(df), dtype = float)
        #fill column
        i = 0
        for plant in df.plant_sp: #for each plant species obtain the corresponding abundance and cover
            abundance_col[i] = abundance[abundance.index == plant]['Estimated_number_of_individuals'].to_numpy()[0]
            cover_col[i]     = abundance[abundance.index == plant]['species_cover (cm)'].to_numpy()[0] 
            i += 1
        df['abundance'] = abundance_col #add column 
        df['cover']     = cover_col #add column
        
    df.to_csv(save_path + '/probabilities_with_abundance_cover.csv',index=False) #save df
        
    return df

def df_mat(df, save_path):
    
    ''' Build df in matrix mode.
    
    Parameters
    ----------
    df: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'interaction_type', 'probability', 'abundance', 'cover'], 
    save_path: str
        Path to save output df.
    
    Returns
    ----------
    df_matrix: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'abundance', 'cover', 'interaction_1', 'interaction_2', ...], 
        element (i,alpha) represent the probability that plant species i participates in interaction_alpha 
        ie. P_{i,alpha} = 1 - \prod_{j=1}^{j} (1 - f_{i,alpha,j}).
    species: list
        List of plant species.
    functions: list
        List of interaction types.
    
    '''
    
    # initialize matrix dataframe
    df_matrix = pd.DataFrame()

    #additionally we add the plant abundance and the cover
    species = np.unique(df['plant_sp']).tolist() #we will need this also later
    df_matrix['plant_sp']  = species
    df_matrix['abundance'] = [df['abundance'][df['plant_sp']== plant].mean() for plant in df_matrix['plant_sp']]
    df_matrix['cover']     = [df['cover'][df['plant_sp']== plant].mean() for plant in df_matrix['plant_sp']]
    #construc matrix
    functions = np.unique(df['interaction_type'].to_numpy()).tolist() # array of ecological functions
    
    for function in functions: # we run the functions one by one 
        df2 = df[df['interaction_type'] == function]  # we consider the reduced dataframe of the selected function 
        # for each function we build a column in the df_matrix dataframe
        array = []
        for i in df_matrix['plant_sp']:# we run every possible plant species (all plant species that participate in any function)
            try:
                value = df2[df2.plant_sp == i]['probability'].to_numpy()[0]  # if the plant i participates in the function we take the probability  
            except: 
                value = 0 #if it does nor participate, the probability is assigned to 0
            array.append(value)
        df_matrix[function] = array
        
    df_matrix.to_csv(save_path + '/probabilities_with_abundance_cover_matrix_mode.csv',index=False) #save df
    
    return df_matrix, species, functions

def df_prod(df_matrix, df_dir, len_functions, functions):
    
    ''' Build product dataframe.
    
    Parameters
    ----------
    df: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'interaction_type', 'probability', 'abundance', 'cover'], 
        probabilities are obtained using: P_{i,alpha} = 1 - \prod_{j=1}^{j} (1 - f_{i,alpha,j}).
        Where P_{i,alpha} is the probability that plant species i participates in interaction_alpha.
    df_dir: str 
        Path to the created folder to save output DataFrames.
    len_functions: int 
        Number of functions.
    functions: list
        List of interaction types.
        
    Returns
    ----------
    df_matrix_prod: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'interaction_1 times interaction_2', ...], 
        product probabilities are obtained using: P_{i,alpha, beta} = P_{i,alpha} x P_{i, beta}.
        Where P_{i,alpha} is the probability that plant species i participates in interaction_alpha.
    functions_pairs: list
        List of lists that constructs all possible functions pairs.
    
    '''
    
    df_matrix_prod = df_matrix['plant_sp'].to_frame()  ## we create a dataframe where rows correspond to every possible plant sp 
    # In order to avoid double counting pairs of functions we construct functions_pairs where it is taken into account that we count ecological functions in the same order as in functions
    functions_pairs = [functions[i+1:] for i in range(len_functions)]
    
    #calculate all df_matrix_prod probabilities
    for function1, functions_pair in zip(functions, functions_pairs): #We run each function and the corresponding function pairs we have to consider
        for function2 in functions_pair: # we run the possible pairs
            if function1 != function2: # we do not calculate the probability of one function with itself
                df_matrix_prod[function1 + ' times ' + function2]  = df_matrix[function1].to_numpy() * df_matrix[function2].to_numpy() # the probability of participatin in function and function2        
                
    df_matrix_prod.to_csv(df_dir + '/probabilities_pairs_matrix_mode.csv',index=False) #save df
    
    return df_matrix_prod, functions_pairs

def dic_fun(function_list):
    
    ''' Build functions dictionary for labels using the first letter of each word .
    
    Parameters
    ----------
    function_list: list
        List of interaction types.
        
    Returns
    ----------
    dic_functions: dict
        Dictionary: interaction_type: label (acronym).
    
    '''
    
    def acronyms(stng):
        
        ''' obtain acronyms.
        
        Parameters
        ----------
        stng: str

        Returns
        ----------
        out: str
            output acronym.
        
        '''

        # add first letter
        oupt = stng[0]
        # iterate over string
        for i in range(1, len(stng)):
            if stng[i-1] == ' ' or stng[i-1] =='-':
                # add letter next to space
                oupt += stng[i]        
        # uppercase oupt
        oupt = oupt.upper()
        return oupt
    
    # #If data is default dataset or has the same ecological functions uses pre-established acronyms
    # if Counter(function_list) == Counter(['herbivory','pathongen-fungi','pollination','saprotrophic-fungi','seed-dispersal','symbiontic-fungi']): #if you are using our ecological functions we have predefine acronyms
    #     dic_functions = {'seed-dispersal':'D',
    #                    'herbivory':'H',
    #                    'pollination':"P",
    #                    'saprotrophic-fungi':"SPF",
    #                    'symbiontic-fungi':"SMF",
    #                    'pathongen-fungi':'PTG'}
    # else: #if you are using different ecological functions we create acronyms from the strings using the first letter of each word 
    #     dic_functions = {}
    #     for function in function_list:
    #         dic_functions[function] = acronyms(function)  

    #If data is default dataset or has the same ecological functions uses pre-established acronyms
    if Counter(function_list) == Counter(['decomposition','fungal pathogenicity','herbivory','nutrient uptake','pollination','seed dispersal']): #if you are using our ecological functions we have predefine acronyms
        dic_functions = {'seed dispersal':'SD',
                       'herbivory':'H',
                       'pollination':"P",
                       'decomposition':"D",
                       'nutrient uptake':"NU",
                       'fungal pathogenicity':'FP'}
    else: #if you are using different ecological functions we create acronyms from the strings using the first letter of each word 
        dic_functions = {}
        for function in function_list:
            dic_functions[function] = acronyms(function)  
            
    return dic_functions

def dic_sp(species_list,len_species):
    
    ''' Build plant dictionary for labels using the first three letters of each word.
    
    Parameters
    ----------
    specues_list: list
        List of plant species.
    len_species: int
        Number of species.
        
    Returns
    ----------
    dic_species: dict
        Dictionary: specie: label (acronym).
    
    '''
    
    dic_species =  {}
    #Plant species labels
    for i in range(len_species): 
        plant = species_list[i]
        plant_l = plant.lower()
        words = plant_l.split(" ", 1)
        
        if len(words) >=2 :
            if len(words[0]) >= 3 and len(words[1]) >= 3:
                #dic_species[plant] = words[0][0:3].capitalize() + '.' + words[1][0:3].capitalize()
                dic_species[plant] = words[0][0:3].capitalize() + '.' + words[1][0:3]
            elif len(words[0]) >= 3 and len(words[1]) == 2:
                 #dic_species[plant] = words[0][0:3].capitalize() + '.' + words[1].capitalize() 
                 dic_species[plant] = words[0][0:3].capitalize() + '.' + words[1] 
            if species_list[i][-1]== '.': 
                #dic_species[plant] = species_list[i][:-1].capitalize() 
                dic_species[plant] = species_list[i][:-1].capitalize()     
                
    return dic_species

def p_matrix(df_matrix, functions):
    
    ''' Build plant dictionary for labels using the first three letters of each word.
    
    Parameters
    ----------
    df_matrix: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'abundance', 'cover', 'interaction_1', 'interaction_2', ...], 
        element (i,alpha) represent the probability that plant species i participates in interaction_alpha 
        ie. P_{i,alpha} = 1 - \prod_{j=1}^{j} (1 - f_{i,alpha,j}).
    functions: list
        List of interaction types.
        
    Returns
    ----------
    P_matrix_df: pandas DataFrame
        Equivalent to df_matrix without abundance and cover columns. 
        Rows and columns are sorted separetly such that higher probabilities appear first 
    P_matrix: numpy array
        Equivalent to P_matrix_df without abundance and cover columns in array form. Each component is an array corresponding to a plant specie in the order of df_matrix.
    P_matrix_t: numpy array
        Transpose of P_matrix.
    
    '''
    
    #construct P_matrix
    P_matrix_df = df_matrix
    P_matrix_df = P_matrix_df.drop([x for x in P_matrix_df.columns.to_list() if x not in functions +['plant_sp']], axis=1) #drop columns of cover and abundance
    #sort by rows and columns
    P_matrix_df['Sum'] = P_matrix_df[functions].sum(axis=1) #sum by rows
    P_matrix_df        = P_matrix_df.sort_values(by=['Sum'], ascending=False) #sort by rows. Those plants whose total contribution is higher appear first
    P_matrix_df        = P_matrix_df.drop(['Sum'], axis=1) #drop auxiliar column
    
    order = P_matrix_df.sum()[1:].sort_values(ascending=False).index.values #sort by columns (get order) #cambiar a false para el principio 
    order = np.insert(order, 0, P_matrix_df.columns[0], axis=0) #insert column of plant species first
    
    P_matrix_df = P_matrix_df[order].reset_index(drop=True) #sort
    P_matrix    = P_matrix_df.values[:, 1:].astype(float)
    P_matrix_t  = np.transpose(P_matrix_df.values[:, 1:].astype(float))
    
    return P_matrix_df, P_matrix, P_matrix_t



def rank(df_matrix, df_matrix_prod, df_dir, species_list, len_species, len_functions):
    
    columns_to_use = df_matrix_prod.columns.difference(df_matrix.columns)
    pentagones = pd.concat([df_matrix, df_matrix_prod[columns_to_use]], axis = 1)
    
    ## we now calculate from the dataframe the total percentage of contribution of each node and edge from the total
    ## that is the contribution of plant i in function j is given by the probability of i in j divided by the sum of j to all plants
    
    for column in pentagones.columns[3:]:
        total = pentagones[column].to_numpy().sum()
        pentagones[column] = pentagones[column].div(total)

    pentagone_rank = pd.DataFrame()
    pentagone_rank['plant_sp'] = species_list
    percentage_node = np.zeros(len_species)
    percentage_edge = np.zeros(len_species)
    columns_nodes = pentagones.columns[3:len_functions+3].to_numpy()
    columns_edges = pentagones.columns[3+len_functions:].to_numpy()
    for plant in pentagone_rank['plant_sp']:
         
        percentage = pentagones[columns_nodes].iloc[pentagone_rank[pentagone_rank['plant_sp'] ==plant].index.to_numpy()[0]].to_numpy().mean()
        percentage_node[pentagone_rank[pentagone_rank['plant_sp'] ==plant].index.to_numpy()[0]] = percentage  
        
        percentage = pentagones[columns_edges].iloc[pentagone_rank[pentagone_rank['plant_sp'] ==plant].index.to_numpy()[0]].to_numpy().mean()
        percentage_edge[pentagone_rank[pentagone_rank['plant_sp'] ==plant].index.to_numpy()[0]] = percentage  
        
        
    pentagone_rank['percentage_node'] = percentage_node
    pentagone_rank['node rank'] = pentagone_rank[['percentage_node']].rank(ascending=False, method='first')
     
    pentagone_rank['percentage_edge'] = percentage_edge
    pentagone_rank['edge rank'] = pentagone_rank[['percentage_edge']].rank(ascending=False, method='first')
    
    pentagone_rank.to_csv(df_dir + '/node_and_edge_ranks.csv',index=False) 
    
    return(pentagone_rank)


# %%
