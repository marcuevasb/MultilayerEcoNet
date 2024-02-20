#%%
print('Importing libraries...')
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
import distinctipy as dsp
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.cluster import hierarchy
from itertools import permutations
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def creat_dir(path, saving=False):
    
    ''' Create folder for saving.
    
    Parameters
    ----------
    path: str
        Directoy path.
    
    Returns
    ----------
    result_dir: str
        Path to the output folder.  
        
    '''
    
    directory  = os.path.dirname(path)        # directory of file
    result_dir = "/".join(directory.split('/')[:-1])+ '/output' # First folder
    os.makedirs(result_dir, exist_ok=True)    # works also if the dir already exists
    
    return result_dir

def createSinAbundance(abundance_path, dataDF, data_path):
    
    """
    Create a synthetic abundance file if it is not provided.
    abundance_path: str or None
        Path to plant abundance and cover dataset. Default None if abundance and cover are not sampled.
        Columns must be ['plant_sp','Estimated_number_of_individuals', 'species_cover (cm)'].
    dataDF: pandas DataFrame
        Pandas DataFrame containing the data.
    data_path: str
        string containing the path to the data.
    Returns
    ----------
    abundance_path: str
        if abundance_path is None, it returns the path to the synthetic abundance file. Otherwise, it returns the same path.

    """
   
    if abundance_path == None:
        unique_Species = dataDF['plant_sp'].unique()
        abundance = np.ones(len(unique_Species))
        cover = np.ones(len(unique_Species))
        abundanceDF = pd.DataFrame({'plant_species':unique_Species, 'Estimated_number_of_individuals':abundance, 'species_cover (cm)':cover})
        # save the abundance file
        abundance_path = data_path[:-4]+'_sinthetyc_abundance.csv'
        abundanceDF.to_csv(abundance_path, index=False)
        print('Synthetic abundance file created in: ', abundance_path)  
   
    return abundance_path

def RFmap_prob(head, separator, data_path, save_path, abundance_path = None):
    
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

    df = pd.read_csv(data_path,  header= head, sep = separator) # original dataset with frequencies #str(argv[0])
    # check if there is a file with abundance and cover
    abundance_path = createSinAbundance(abundance_path, df, data_path)

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
    
    return df

def RFmap_mat(df):
    
    ''' Build df in matrix mode.
    
    Parameters
    ----------
    df: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'interaction_type', 'probability', 'abundance', 'cover'], 
    
    Returns
    ----------
    RFmap_matrix: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'abundance', 'cover', 'interaction_1', 'interaction_2', ...], 
        element (i,alpha) represent the probability that plant species i participates in interaction_alpha 
        ie. P_{i,alpha} = 1 - \prod_{j=1}^{j} (1 - f_{i,alpha,j}).
    species: list
        List of plant species.
    functions: list
        List of interaction types.
    
    '''
    
    # initialize matrix dataframe
    RFmap_matrix = pd.DataFrame()

    #additionally we add the plant abundance and the cover
    species = np.unique(df['plant_sp']).tolist() #we will need this also later
    RFmap_matrix['plant_sp']  = species
    RFmap_matrix['abundance'] = [df['abundance'][df['plant_sp']== plant].mean() for plant in RFmap_matrix['plant_sp']]
    RFmap_matrix['cover']     = [df['cover'][df['plant_sp']== plant].mean() for plant in RFmap_matrix['plant_sp']]
    #construc matrix
    functions = np.unique(df['interaction_type'].to_numpy()).tolist() # array of ecological functions
    
    for function in functions: # we run the functions one by one 
        df2 = df[df['interaction_type'] == function]  # we consider the reduced dataframe of the selected function 
        # for each function we build a column in the RFmap_matrix dataframe
        array = []
        for i in RFmap_matrix['plant_sp']:# we run every possible plant species (all plant species that participate in any function)
            try:
                value = df2[df2.plant_sp == i]['probability'].to_numpy()[0]  # if the plant i participates in the function we take the probability  
            except: 
                value = 0 #if it does nor participate, the probability is assigned to 0
            array.append(value)
        RFmap_matrix[function] = array
    
    return RFmap_matrix, species, functions

def acronyms(stng):
    
    '''
    Given an input string, returns another string with its acronym, 
    considering spaces and hyphens as separators.
    '''

    oupt = stng[0]
    for i in range(1, len(stng)):
        if stng[i-1] == ' ' or stng[i-1] =='-':
            oupt += stng[i]        
    oupt = oupt.upper()
    return oupt

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

    # Some default acronyms for our ecological functions
    if Counter(function_list) == Counter(['decomposition','fungal pathogenicity','herbivory','nutrient uptake','pollination','seed dispersal']): #if you are using our ecological functions we have predefine acronyms
        dic_functions = {'seed dispersal':'SD',
                       'herbivory':'H',
                       'pollination':"P",
                       'decomposition':"D",
                       'nutrient uptake':"NU",
                       'fungal pathogenicity':'FP'}
    else: #if you are using your own ecological functions we have to define the acronyms
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
    for plant in species_list:
        plant_l = plant.lower()
        words = plant_l.split(" ", 1)
        dic_species[plant] = words[0][0:3].capitalize() + '.' + words[1][0:3]

    return dic_species

def p_matrix(RFmap_matrix, functions, save_path):
    
    ''' Build plant dictionary for labels using the first three letters of each word.
    
    Parameters
    ----------
    RFmap_matrix: pandas DataFrame
        Pandas DataFrame with columns ['plant_sp', 'abundance', 'cover', 'interaction_1', 'interaction_2', ...], 
        element (i,alpha) represent the probability that plant species i participates in interaction_alpha 
        ie. P_{i,alpha} = 1 - \prod_{j=1}^{j} (1 - f_{i,alpha,j}).
    functions: list
        List of interaction types.
    save_path: str
        Path to save output df.
        
    Returns
    ----------
    P_matrix_df: pandas DataFrame
        Equivalent to RFmap_matrix without abundance and cover columns. 
        Rows and columns are sorted separetly such that higher probabilities appear first 
    P_matrix: numpy array
        Equivalent to P_matrix_df without abundance and cover columns in array form. Each component is an array corresponding to a plant specie in the order of RFmap_matrix.
    P_matrix_t: numpy array
        Transpose of P_matrix.
    
    '''

    #construct P_matrix by sorting rows and columns
    P_matrix_df = RFmap_matrix.drop(["plant_sp", "cover", "abundance"], axis=1).iloc[
        RFmap_matrix[functions].sum(axis=1).argsort()[::-1],
        RFmap_matrix[functions].sum(axis=0).argsort()[::-1]]
    # add a column at the beginning with the plant species sorted as in P_matrix_df
    P_matrix_df.insert(0, "plant_sp", 
                       RFmap_matrix["plant_sp"].iloc[P_matrix_df.index.to_numpy()])
    
    P_matrix_df = P_matrix_df.reset_index(drop=True) 
    P_matrix    = P_matrix_df.values[:, 1:].astype(float)
    P_matrix_t  = np.transpose(P_matrix_df.values[:, 1:].astype(float))

    P_matrix_df.to_csv(save_path + '/P_matrix.csv',index=False)
    P_matrix.to_csv(save_path + '/P_matrix_matrix.csv',index=False)
    return P_matrix_df, P_matrix, P_matrix_t

def sort_RFmap_matrix(RFmap_matrix, axis_1_sort, axis_0_sort):
    
    """
    Sort the matrix by the ecological function and plant species

    Parameters
    ----------
    RFmap_matrix : pandas dataframe
        dataframe with the matrix
    axis_1_sort : list
        list with the ecological functions in the order they should be sorted
    axis_0_sort : list
        list with the plant species in the order they should be sorted

    Returns
    -------
    RFmap_matrix : pandas dataframe
        dataframe with the matrix sorted and columns "cover" and "abundance" removed
    """

    RFmap_matrix = RFmap_matrix.loc[:, ["plant_sp"] + list(axis_1_sort)]
    RFmap_matrix['plant_sp'] = RFmap_matrix['plant_sp'].astype("category")
    RFmap_matrix['plant_sp'] = RFmap_matrix['plant_sp'].cat.set_categories(axis_0_sort)
    RFmap_matrix = RFmap_matrix.sort_values(['plant_sp'])
    RFmap_matrix = RFmap_matrix.reset_index(drop=True)
    
    return RFmap_matrix

def item_words(strg, line_sep):
    
    '''
    Return the input string correctly formatted for plotting
    '''

    sep = "\n" if line_sep==True else ""
    words = strg.split(" ", 1)
    # Case A: for all ways to write this particular species
    if strg in [ 'Olea europaea var. sylvestris', 'Olea Europaea Var. sylvestris','Olea europaea Var. sylvestris','Olea Europaea Var. sylvestris','Olea Europaea Var. Sylvestris']:
        strg = r'%s $\mathit{%s}$' % ('','Olea europaea') + sep 
    # Case B: the species has more than two words
    elif len(words)> 2:
        strg = r'%s $\mathit{%s}$' % ('', words[0].capitalize()+'\,'+ words[1:][0] + sep) 
    # Case C: it has 2 or more words and some of them are longer than 12 characters
    elif any([len(word)>12 for word in words]) and len(words)>1:
        strg = rf'$\mathit{words[0].capitalize()}$' + ' \n ' + rf'$\mathit{words[1]}$'
    # Case D: the species has two words and none of them is longer than 12 characters
    elif len(words)>1:
        strg = r'%s $\mathit{%s}$' % ('', words[0].capitalize() +'\,'+ words[1:][0]) + sep
    # any other case
    else:
        strg =r'%s $\mathit{%s}$' % ('', words[0].capitalize()) 
    return strg

def titles(list_, line_sep=True):
    
    '''
    write titles for plots
    '''

    return [item_words(i,line_sep) for i in list_]

       


def plot_shp(length, ncols):
    
    """
    Given the number of elements to plot, and the number of columns wanted
    return the strings in a format that can be passed to plt.mosaic_plot

    Parameters
    ----------
    length : int
        Number of elements to plot.
    ncols : int
        Number of columns wanted.

    Returns
    -------
    plot_str : str
        String with the letters to plot.
    plot_shape : str
        String with the shape of the plot.
    nrows : int
        Number of rows of the plot. 
    """

    plot_str =string.ascii_uppercase[:(length+1)]
    plot_shape = '\n'.join(plot_str[i:i + ncols] for i in range(0, len(plot_str), ncols))
    plot_str = plot_str[:-1]
    plot_shape = plot_shape[:-1] +'.'
   
    last  = plot_shape.split('\n')[-1]
    nrows = math.ceil((length+1)/ncols)
    if len(last)<ncols:
        plot_shape = plot_shape +'.'*(ncols - len(last))
    
    return plot_str, plot_shape, nrows

def p_matrix_elem(RFmap_matrix, elem, axis):
    
    """
    Remove all but elem from RFmap_matrix and return the corresponding P_matrix and P_matrix_t

    Parameters
    ----------
    RFmap_matrix : pandas dataframe
        Dataframe with the probabilities of participation of each plant sp in each function.
    elem : list
        List of elements to keep in RFmap_matrix.
    axis : int
        Axis to keep the elements. 0 for rows and 1 for columns.

    Returns
    -------
    P_matrix : numpy array
        Probability matrix.
    P_matrix_t : numpy array
        Transpose of the probability matrix.
    """

    if axis ==1:    
        cols_to_remove = RFmap_matrix.columns.to_list()
        for el in elem:
            cols_to_remove.remove(el)
        cols_to_remove.remove('plant_sp')
        P_matrix_df = RFmap_matrix.drop(cols_to_remove, axis=1)
       
    elif axis == 0:
        rows_to_remove = RFmap_matrix['plant_sp'].to_list()
        for el in elem:
            rows_to_remove.remove(el)
        P_matrix_df = RFmap_matrix[~RFmap_matrix['plant_sp'].isin(rows_to_remove)]
       
    P_matrix    = P_matrix_df.values[:, 1:].astype(float)
    P_matrix_t  = np.transpose(P_matrix_df.values[:, 1:].astype(float))
   
    return  P_matrix, P_matrix_t



def rank(strength, elem_order, axis =1):

    """
    sort the elements of "elem_order" according to the sum of the elements of "strength" along the axis "axis"

    Parameters
    ----------
    strength : array
        array of strength of each element.
    elem_order : array
        array of elements.
    axis : int, optional

    Returns
    -------
    sums : array
        array of sum of strength of each element.
    strength_rank : array
        array of strength of each element sorted.
    species_rank : array
        array of elements sorted.

    """

    sums = np.sum(strength, axis=axis)
    sorter = np.argsort(sums)[::-1]
    species_rank = np.array(elem_order)[sorter]
    strength_rank = np.array(strength)[sorter]
   
    return sums, strength_rank,species_rank


def func_color(att_list):

    '''
    Create a dictionary that maps each element of a list to a color.

    If the list is the list of functions, the colors are the ones used in the paper.

    Otherwise, the colors are randomly generated using the function get_colors from the module dsp. 
    
    Parameters
    ----------
    att_list: attributes list
    
    Returns
    ----------
    func_to_color: dict
        Dict object attr: color.
        
    '''

    # Some default colors for our ecological functions
    if set(att_list) == set(['decomposition','fungal pathogenicity','herbivory','nutrient uptake','pollination','seed dispersal']):
        func_to_color= {'herbivory': 'tab:purple',
         'plant': 'tab:green',
         'pollination': 'tab:orange',
         'decomposition': 'tab:brown',
         'fungal pathogenicity': 'tab:olive',
         'seed dispersal': 'tab:blue',
         'nutrient uptake': 'tab:pink'}
    
    else:#if you are using your own ecological functions we have to define the colors
        colors = dsp.get_colors(len(att_list), pastel_factor=0.7, rng = 123)
        func_to_color = {att_list[item]: colors[item] for item in range(len(att_list))}
    return func_to_color

def bipartite_pos(groupL, groupR, ratioL=4/3, ratioR = 4/3):
    
    """
    Compute bipartite node positions for a horizontally aligned bipartite graph.
    Parameters
    ----------
    groupL : list
        List of nodes in the left group.
    groupR : list
        List of nodes in the right group.
    ratioL : float
        Ratio of horizontal to vertical spacing for the left group.
    ratioR : float
        Ratio of horizontal to vertical spacing for the right group.

    Returns    
    -------
    pos : dict
        Dictionary of positions keyed by node.
    pos_labels : dict
        Dictionary of label positions keyed by node.
    """

    pos = {}
    pos_labels = {}
    gapR = ratioR/ len(groupR)
    ycord = ratioL / 2 - gapR/2
    for r in groupR:
        pos[r] = (1, ycord)
        pos_labels[r] = (1.1, ycord)
        ycord -= gapR
    gapL = ratioL/ len(groupL)
    ycord = ratioL / 2 - gapL/2
    for l in groupL:
        pos[l] = (-1, ycord)
        pos_labels[l] = (-1.1, ycord)
        ycord -= gapL

    return pos, pos_labels

def layer_layout(G, layout='spring', k=2, seed=1234):
   
    """
    Compute the position dictionary of G using the specified layout.
    Parameters
    ----------
    G : graph
        A networkx graph
    layout : string
        The layout algorithm to use. Available are "spring", "circular", "shell" or "tree"
    seed : int
        Seed for random layout
    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node
    """

    if len(G.nodes) == 5: # Some default positions for our ecological functions

        pos_arr = [np.array([1., 0.]), 
                   np.array([0.30901695, 0.95105657]), 
                   np.array([-0.80901706,  0.58778526]), 
                   np.array([-0.809017  , -0.58778532]), 
                   np.array([ 0.3090171 , -0.95105651])]
        pos = {np.array(G.nodes)[i]: pos_arr[i] for i in range(len(G.nodes))}
    
    elif layout == 'spring':
        pos = nx.spring_layout(G, weight = "weight", seed=seed, k=k)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    elif layout == "tree":
        pos = nx.nx_agraph.graphviz_layout(G,prog="twopi")
    else:
        pos = nx.random_layout(G)
    
    return pos

def func_cmap(att_list,function,dic_colors_fn={},cmp = 'Reds'):
    
    '''Returns a colormap dic,
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

    corr = {'tab:purple': (148/256,103/256,189/256),
            'tab:orange': (255/256,127/256,14/256),
            'tab:green':  (44/256,160/256,44/256),
            'tab:brown':  (140/256,86/256,75/256),
            'tab:olive':  (188/256,189/256,34/256),
            'tab:blue':   (31/256,119/256,180/256),
            'tab:pink':   (227/256,119/256,194/256),
            'tab:grey':   (127/256,127/256,127/256)}
    
    # Some default colors for our ecological functions
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

    # Build a sequential cmap for each function color that goes from the function color to white
    if len(dic_colors_fn)!=0:
        func_to_cmp={}
        for function in function:
            color = dic_colors_fn[function]
            vals = np.ones((N, 3))
            vals[:, 0] = np.linspace(color[0], 1, N)
            vals[:, 1] = np.linspace(color[1], 1, N)
            vals[:, 2] = np.linspace(color[2], 1, N)
            cmp  = ListedColormap(vals)
            func_to_cmp[function]= cmp

    else:
        func_to_cmp    = {i:'Reds_9' for i in att_list}
   
    return func_to_cmp

def threedplot_sort_speces(head, separator, data_path, matrix_order):
   
    """
    Returns the species, functions and individals sorted in a way that 
        the plot of the tensor is clear and reduce overlaping

    Parameters
    ----------
    head : str
        Header of datasets.
    separator : str
        Separator of datasets.
    data_path : str
        Path to dataset with frequencies of occurrence by sampling_effort.
        Columns must be ['plant_sp', 'animal_sp','frequency_of_occurrence_weighted_by_sampling_effort']. Additionally ['plant_sp',abundance, 'cover'].
    matrix_order : pandas dataframe
        Dataframe with the order of species and functions.

    Returns
    -------
    species_array : array
        Array with the order of species.
    species_axis : array
        Array with the species names (used in plot labels).
    functions_array : array
        Array with the order of functions.
    functions_axis : array  
        Array with the functions names (used in plot labels).
    animals_array : array
        Array with the order of animals.
    animals_axis : array
        Array with the animals names (used in plot labels).
    interaction_array : array
        Array with the order of interactions.

    """

    # add order columns for species and interaction types for each row
    # based on their order in the matrix_order dataframe
    df = pd.read_csv(data_path,  header= head, sep = separator) # original dataset with frequencies #str(argv[0])
    species = matrix_order['plant_sp'].to_list()
    species_order = {s: len(species) - i for i, s in enumerate(species)}
    df['plant_sp order'] = [species_order[i] for i in df['plant_sp']]
    
    functions = matrix_order.columns[1:].to_list()
    functions_order = {f: len(functions) - i for i, f in enumerate(functions)}
    df['interaction_type order'] = [functions_order[i] for i in df['interaction_type']]
   
    animals_order_arr = []
    animals_order = {}
    last = 0
    for function in functions:
        
        # take only animals that interact with the function
        animals_fun = list(set(df['animal_sp'][df['interaction_type']  == function].to_list()))
        animals_fun = [x for x in animals_fun if x not in animals_order_arr]
        animals_order_arr.append(animals_fun)

        animals_av_freq = []
   
        for animals_fun_i in animals_fun:
            average_freq = np.sum(df['frequency_of_occurrence_weighted_by_sampling_effort'][df['animal_sp'] == animals_fun_i].to_numpy())
            animals_av_freq.append(average_freq)
           
        animals_df = pd.DataFrame()
        animals_df['animals'] = animals_fun
        animals_df['average freq'] = animals_av_freq
        animals_df = animals_df.sort_values(by=['average freq'], ascending =False)
        animals_df['order'] = np.arange(last+1,len(animals_df)+1+last)
       
        for i in animals_df['animals'].to_list():
            animals_order[i] = animals_df['order'][animals_df['animals'] == i].to_list()[0]
           
        last += len(animals_df)
    
    df['animal_sp order'] = [animals_order[i] for i in df['animal_sp']]
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




def legend_cmap(att_list, dic_colors_fn, functions_axis, func_to_cmp):
    
    '''
    colormap for legendS
    '''

    legend_elements= []
    if Counter(att_list) == Counter(['decomposition','fungal pathogenicity','herbivory','nutrient uptake','pollination','seed dispersal']):
        for fun in att_list:
            legend_elements.append(Line2D([0], [0], 
                                          marker='o', 
                                          color='w', 
                                          label=fun.capitalize(), 
                                          markerfacecolor=dic_colors_fn[fun], 
                                          markersize=15))
    else:
        for i in range(len(functions_axis))[::-1]:
            cmap = func_to_cmp[functions_axis[i]]
            cmap = mpl.cm.get_cmap(cmap)
            color = cmap(0)
           
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=functions_axis[i].capitalize(), markerfacecolor=list(color)[:3], markersize=15))

    return legend_elements



####
# ________ HIERARCHICAL CLUSTERING ________
####


def plot_dendrogram_with_names(dissimilarity_matrix, leaf_names):
    # Perform hierarchical clustering
    linkage_matrix = hierarchy.linkage(dissimilarity_matrix, method='complete')
    # Plot the dendrogram with leaf names
    # Override the default linewidth.
    plt.rcParams['lines.linewidth'] = 2.5
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(linkage_matrix, labels=leaf_names, leaf_rotation=90,leaf_font_size=15)
    #plt.xlabel('')
    plt.ylabel('Scale', fontsize=18)
    #plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

def plot_heatmap(array, x_labels, y_labels,heatmap_name):
    plt.figure(figsize=(15, 4))
    plt.imshow(array, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90, fontsize=15)
    plt.yticks(np.arange(len(y_labels)), y_labels, fontsize=15)
    #plt.xlabel('X Label')
    #plt.ylabel('Y Label')
    plt.title(heatmap_name)
    plt.show()
    
def largest_connected_component(adj_matrix):
    # Convert adjacency matrix to a sparse matrix
    adj_sparse = csr_matrix(adj_matrix)
    # Compute connected components
    num_components, labels = connected_components(adj_sparse)
    # Find the largest connected component
    largest_component_label = np.argmax(np.bincount(labels))
    # Get the indices of the nodes in the largest component
    largest_component_indices = np.where(labels == largest_component_label)[0]
    return largest_component_indices
    
def percolacion_funciones(matriz_p,cual):
    #cual 0 --> funciones
    #cual 1 --> plantas
    #lista de funciones
    lista_funciones = ['decomposition','fungal pathogenicity','nutrient uptake','herbivory','pollination','seed dispersal']
    lista_plantas = ['Withania frutescens', 'Lavatera maritima', 'Olea europaea', 'Euphorbia dendroides', 'Medicago arborea', 'Suaeda vera', 'Limonium sp', 'Diplotaxis ibicensis', 'Arisarum vulgare', 'Geranium molle', 'Narcissus tazetta', 'Ephedra fragilis', 'Fagonia cretica', 'Asparagus horridus', 'Chenopodium murale', 'Heliotropium europaeum']
    function_dict = {"decomposition":0, "fungal pathogenicity":1,"nutrient uptake":2, "herbivory":3,"pollination": 4,"seed dispersal":5}
    plant_dict ={'Withania frutescens':0, 'Lavatera maritima':1, 'Olea europaea':2, 'Euphorbia dendroides':3, 'Medicago arborea':4, 'Suaeda vera':5, 'Limonium sp':6, 'Diplotaxis ibicensis':7, 'Arisarum vulgare':8, 'Geranium molle':9, 'Narcissus tazetta':10, 'Ephedra fragilis':11, 'Fagonia cretica':12, 'Asparagus horridus':13, 'Chenopodium murale':14, 'Heliotropium europaeum':15}
    #hago permutacion en la lista
    vector_AUC_w=[]
    vector_AUC=[]
    if(cual==0):
        lista=lista_funciones
        AUC_max=81
        AUC_w_max=0
        AUC_min=66
        AUC_w_min=10000
        for perm in permutations(lista):
            matriz_p_actualizada = matriz_p.copy()
            AUC=0
            AUC_w=0
            LCC=[16]
            for funcion in perm:
                row_to_remove=function_dict[funcion]
                #calculo el peso quitado
                peso_quitado=0
                vector=matriz_p_actualizada[:,row_to_remove].copy()
                for i in range(len(vector)):
                    for j in range(len(vector)):
                        peso_quitado=peso_quitado + vector[i]*vector[j]                       
                #peso_quitado=sum(np.matmul(matriz_p_actualizada[:,row_to_remove],np.transpose(matriz_p_actualizada[:,row_to_remove]))
                #quitar de la matriz las entradas adecuadas
                matriz_p_actualizada[:,row_to_remove]=0
                LCCnew=len(largest_connected_component(np.matmul(matriz_p_actualizada, np.transpose(matriz_p_actualizada))))
                print(LCCnew)
                LCC.append(LCCnew)
                #calcular la LCC y sumarla a la AUC
                AUC_w=AUC_w+(peso_quitado)*LCC[-1] + peso_quitado*(LCC[-2]-LCC[-1])*0.5 #trapecio, para AUC weighted
                AUC=AUC+ LCC[-1] #AUC sin pesar
            print('orden:',perm)
            print('AUC=',AUC,'AUC_w',AUC_w)
            print()
            vector_AUC_w.append(AUC_w)
            vector_AUC.append(AUC)
            if(AUC==AUC_max):
                if(AUC_w>AUC_w_max):
                    max_AUC_list=perm
                    AUC_w_max=AUC_w
                    max_LCC_list=LCC
            if(AUC==AUC_min):
                if(AUC_w<AUC_w_min):
                    min_AUC_list=perm
                    AUC_w_min=AUC_w
                    min_LCC_list=LCC
    elif(cual==1):
        print('INCOMPLEEEEETE')
                
    return AUC_w_max,max_AUC_list,AUC_w_min,min_AUC_list,vector_AUC_w,max_LCC_list,min_LCC_list