#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.lines import Line2D
from infomap import Infomap 
from collections import Counter
import os



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



def build_G(head, separator,data_path, abundance_path= []):
    
    ''' Build networkx object from data.
    
    Parameters
    ----------
    head: str
        Header of datasets.
    separator: str
        Separator of datasets.
    data_path: str
        Path to dataset with frequencies of occurrence by sampling_effort.
        Columns must be ['plant_sp', 'animal_sp','frequency_of_occurrence_weighted_by_sampling_effort']. Additionally ['plant_sp',abundance, 'cover'].
    abundance_path: str
        Optional. Path to plant abundance and cover dataset. Default empty if abundance already included in data.
        Columns must be ['plant_sp',abundance, 'cover'].
    
    Returns
    ----------
    G: networkx Multigraph
        Networkx object where nodes represent plant, animal and fungi species and edges quantify ecological interactions frequencies.
    node_label: dict
        Dictionary of plant species labels.
    plant_label_color: dict
        Dictionary of plant species labels color. Plant species contributing in all ecological functions are labelled in blue
    
    '''
    
    df = pd.read_csv(data_path,  header= head, sep = separator) # original dataset with frequencies #str(argv[0])
    
    #sort df in alphabetical order of ecological functions
    df = df.sort_values('interaction_type').reset_index(drop= True)

    # check if abundance file is provided and if not, create a synthetic one
    abundance_path = createSinAbundance(abundance_path, df, data_path)

        

    #We add two columns with the corresponding abundance and species cover of each plant sp
    abundance = pd.read_csv(abundance_path, header= head,  index_col=0) # abundance dataset 
    #initialization of columns
    abundance_col = np.zeros(len(df), dtype = int)
    cover_col     =  np.zeros(len(df), dtype = float)
    #fill column
    i = 0
    if 'Estimated_number_of_individuals' or 'species_cover (cm)' in abundance.columns:
        abundance = abundance.rename(columns={"Estimated_number_of_individuals": "abundance","species_cover (cm)": "cover"})
        
    for plant in df.plant_sp: #for each plant species obtain the corresponding abundance and cover
        abundance_col[i] = abundance[abundance.index == plant]['abundance'].to_numpy()[0]
        cover_col[i]     = abundance[abundance.index == plant]['cover'].to_numpy()[0] 
        i += 1
    df['abundance'] = abundance_col #add column 
    df['cover']              = cover_col #add column
    
    df = df.rename(columns={'frequency_of_occurrence_weighted_by_sampling_effort': 'width'})
    
    # rescale widths by 3*maximum 
    df['width'] = (df['width'] /df['width'].max())*3 
    
    # Create networkx networl
    G = nx.from_pandas_edgelist(df, 'plant_sp', 'animal_sp', edge_attr=['interaction_type', 'width'], create_using=nx.MultiGraph)
    nodes = list(G.nodes())
    
    # Assign a minimum abundance of 0.5 (used for animal species)
    node_att_dict = {node:{'type': [], 'abundance':0.5} for node in nodes}
    a_max = max(df['abundance'].to_numpy())
    
    for i, row in df.iterrows(): #add atributes
        plant = row.iloc[0]
        abundance = row['abundance']
        animal  = row['animal_sp']
        int_type  = row['interaction_type']
        node_att_dict[plant]['type'] = ['plant']
        node_att_dict[plant]['abundance'] = (abundance /a_max)*3 +0.5 #rescale abundance of plant species
        
        if int_type not in node_att_dict[animal]['type']:
            node_att_dict[animal]['type'].append(int_type)
    
    nx.set_node_attributes(G, node_att_dict)
    
    #build node_label: dict of plant species label
    plants = np.unique(df['plant_sp'].to_numpy())
    node_label = {plant: index for plant, index in zip(plants, range(1,len(plants)+1))}
    #build plant_label_color: dict of node label color. Plant species contributing in all ecological functions are labeled in blue
    max_connections = len(df['interaction_type'].unique())
    plant_label_color = {plant: 'k' for plant in node_label}
    
    for plant in node_label: 
        if len(df[df.plant_sp==plant]['interaction_type'].unique()) == max_connections:
            plant_label_color[plant] = 'b'
    node_label_name = {plant: plant.title() for plant in plants}
    
    return G, node_label, plant_label_color, node_label_name

def Multi_to_Graph(G,save_path, saving = False):
    
    ''' Build networkx graph from networkx multigraph. 
    
    Parameters
    ----------
    G: networkx Multigraph object
        Networkx object where nodes represent plant, animal and fungi species and edges quantify ecological interactions frequencies.
    
    save_path: str
        Path to save multiedges found.
    
    Returns
    ----------
    H: networkx graph
        Edges weights is obtained: f_{i,k} = 1 - \prod_{j=1}^{j}.
        Edge interaction type is assigned by default to the latest in alphabetical order.
    
    '''
    multi_edges = pd.DataFrame(columns=['plant', 'animal', 'interaction_1', 'interaction_2'])
    multi_edges_tuple = []
    multi_edges_width = []
    multi_edges_save  = []
    H = nx.Graph()
    for edge in np.unique(np.array(list(G.edges())), return_counts=True, axis = 0)[0]:  
        u, v = edge[0], edge[1]
        interaction_type = G[u][v][0]["interaction_type"]
        width = [G[u][v][0]["width"]]
        if len(G[u][v])>1:
            
            for j in G[u][v]: # if the same edge appears more than once calculate width using equation
                if j==0: 
                    multi_edges_save.append(tuple((u,v,{'interaction_type':G[u][v][j]["interaction_type"], 'width':G[u][v][j]["width"]})))
                else:
                    #interaction_type = max(interaction_type, G[u][v][j]["interaction_type"])
                    width.append(G[u][v][j]["width"]) 
                    multi_edges_tuple.append(tuple((u,v,{'interaction_type':G[u][v][j]["interaction_type"], 'width':G[u][v][j]["width"]})))
                    multi_edges_width.append(G[u][v][j]["width"])
        if len(width)>1:
            multi_edges.loc[len(multi_edges.index)] = [u, v,G[u][v][0]["interaction_type"],G[u][v][1]["interaction_type"]]
            
        width =1 - np.prod(1 - np.array(width))
        # set same attributes
        att_dict = {'interaction_type': interaction_type, 'width': width}
        H.add_edge(u,v)
        nx.set_edge_attributes(H, {(u,v): att_dict})
    nx.set_node_attributes(H, dict(G.nodes(data=True)))
    multi_edges_width = np.array(multi_edges_width)
    
    if saving:
        multi_edges.to_csv(save_path + '/multi_edges.csv',index=False) 
    
    return H, multi_edges_tuple, multi_edges_save 




def lou_communities(G, seed=1234):
    
    '''Computes modular hierarchy. 
    
    Parameters
    ----------
    G: networkx Multigraph object
        Networkx object where nodes represent plant, animal and fungi species and edges quantify ecological interactions frequencies.
        
    seed: int
        Optional. Default value: 1234.
    
    Returns
    ----------
    G: networkx Multigraph object
        Adds as note attr the community id.
    
    community_dict_lou: dict
        node id: community id.
        
    '''
    
    lou_comm = nx.algorithms.community.louvain_communities(G,weight='width', seed=seed)
    Ncom = len(lou_comm)
    community_dict_lou = {}
    for com in range(Ncom):
        for node in lou_comm[com]: 
            community_dict_lou[node] = com 

    nx.set_node_attributes(G, community_dict_lou, "community lou")

    return G, community_dict_lou

def inf_communities(H, seed = 1234):
    
    '''Computes modular hierarchy. 
    
    Parameters
    ----------
    H: networkx Graph object 
       networkx Graph.
    
    Returns
    ----------
    H: networkx Graph object
        Adds as note attr the community id.
    
    community_dict_inf: dict
        node id: community id.
    '''
    
    node_label_dict = {}
    i=0
    for node in H.nodes():
        node_label_dict[node] = i
        i+=1
       
    label_node_dict = {}
    for node in node_label_dict:
        label_node_dict[int(node_label_dict[node])]  = node

    im = Infomap(silent=True, seed=seed)   
    for u,v,w in H.edges(data=True):
        im.add_link(int(node_label_dict[u]), int(node_label_dict[v]), weight=w['width'])
    im.run()
    community_dict_inf = dict()
    for node in im.tree:
        if node.is_leaf:
            a = int(node.node_id)
            community_dict_inf[label_node_dict[a]] = node.module_id
        
    nx.set_node_attributes(H, community_dict_inf, "community inf")

    return H, community_dict_inf
   
def color_of_edges(G, edge_att = 'interaction_type',cmap = plt.cm.rainbow):# run if you want to use different colors for nodes and edges
    
    '''Returns a colormap dic based on the palette 'plt.cm.rainbow',
       discretized into the number of (different) colors
       found in the attribute.
    
    Parameters
    ----------
    G: networkx MultiGraph or Graph object
        networkx object.
    
    edge_att: str
        Edge attribute to use. Default value: 'interaction_type'
        
    cmap: matplotlib cmap
        Colormal to use. Default value: plt.cm.rainbow
        
    Returns
    ----------
    cmp_edges: cmap
        Cmap of edges.
    
    colors: array
        array of edges attributs in G.edges order.
        
    dict_str_to_int: dict
        Dict object edge_attribute: int.
        Use later to obtain colors from cmp_edges.
        
    '''
     
    colors = [d[edge_att] for u,v,d in G.edges(data = True)]
    colors = np.array(colors)
    Ncolors = len(np.unique(colors))
    cmap_edges = cmap(np.linspace(0,1,Ncolors))
    cmp_edges = mpl.colors.ListedColormap(cmap_edges)

    different_colors = np.unique(colors)
    dict_str_to_int = {color: i for i, color in zip(range(len(different_colors)), different_colors)}
    
    if type(G) == nx.classes.multigraph.MultiGraph:
        weigt_dict = {edge: {"weight" : dict_str_to_int[color]} for edge,color in zip(G.edges(keys=True), colors)}
        
    else: 
        weigt_dict = {edge: {"weight" : dict_str_to_int[color]} for edge,color in zip(G.edges(), colors)}
    nx.set_edge_attributes(G, weigt_dict)
    return cmp_edges, colors, dict_str_to_int

def color_of_nodes_edges(G,node_att, cmap = plt.cm.rainbow):
    
    '''Returns a colormap dic based on the palette 'plt.cm.rainbow',
       discretized into the number of (different) colors
       found in the attribute.
    
    Parameters
    ----------
    G: networkx MultiGraph or Graph object
        networkx object.
    
    node_att: str
        Node attribute to use. 
        
    cmap: matplotlib cmap
        Colormal to use for nodes and edges. Default value: plt.cm.rainbow
        
    Returns
    ----------
    node_colors_dict: dict
        Node color dictionary. Node:color
    
    edge_to_color_dict: dict
        Edges set color dict according to node att. Edge: color. Used to plot animal/fungi species and corresponding edges in same color.
        
        
    '''

    if node_att == 'type':
        node_colors      = [d[1][node_att] for d in G.nodes(data=True)]
        node_colors      = np.array([item for sublist in node_colors for item in sublist])
    else: 
        node_colors      =np.array([d[1][node_att] for d in G.nodes(data=True)])
    Ncolors          = len(np.unique(node_colors))
    edge_to_int      = {f:i for i, f in enumerate(np.unique(node_colors))}
    node_edge_color  = {}
    
    #If data is default dataset or has the same ecological functions uses pre-established colors)
    if Counter([a for a in edge_to_int]) == Counter(['herbivory','pollination','decomposition','seed dispersal', 'nutrient uptake', 'plant','fungal pathogenicity']):
        edge_to_color_dict= {'herbivory': (0.5780392156862744, 0.4545098039215687, 0.6905882352941175),
        'plant': (0.33203125, 0.55078125, 0.25),
        'pollination': (0.8582352941176472, 0.5068627450980392, 0.19666666666666655),
        'decomposition': (0.5107843137254903,
        0.3625490196078432,
        0.33235294117647063),
        'fungal pathogenicity': (0.6472549019607846,
        0.6500000000000001,
        0.2245098039215685),
        'seed dispersal': (0.20921568627450982,
        0.45078431372549005,
        0.6182352941176471),
        'nutrient uptake': (0.8266666666666667,
        0.5301960784313726,
        0.7360784313725488)}
        node_colors_dict = {u: edge_to_color_dict[d[node_att][0]] for u,d in G.nodes(data=True)}
        node_edge_color = {u: edge_to_color_dict[d[node_att][0]] for u,d in G.nodes(data=True)}
        
        for u,d in G.nodes(data=True):
            if len(d['type'])>1:
                node_edge_color[u] = edge_to_color_dict[d[node_att][1]]
    # If not, used the input colormap
    else:
        cmap_nodes       = cmap(np.linspace(0,1,Ncolors))
        cmp_nodes        = mpl.colors.ListedColormap(cmap_nodes)
        node_colors_dict = {u: cmp_nodes.colors[edge_to_int[d[node_att][0]]] for u,d in G.nodes(data=True)}
        node_edge_color  = {u: cmp_nodes.colors[edge_to_int[d[node_att][0]]] for u,d in G.nodes(data=True)}
        for u,d in G.nodes(data=True):
            if len(d['type'])>1:
                node_edge_color[u] = cmp_nodes.colors[edge_to_int[d[node_att][1]]]
        
        edge_to_color_dict    = {k:cmp_nodes.colors[i] for k,i in edge_to_int.items()}
        
    
    return node_colors_dict, edge_to_color_dict, node_edge_color


####voy por aqui 
def size_of_nodes(G, node_att = 'abundance'):
    node_size_dict = {u: d[node_att] for u,d in G.nodes(data=True)}
    return node_size_dict



def node_edge(G, node_colors, node_label):
    node_edge_color = {i[0]: i[1] for i in node_colors.items()}
    node_edge_width = {i:0 for i in G.nodes()}
    for i in node_label.keys():
        node_edge_color[i]=(0.,0.,0.,1.)
        node_edge_width[i]= 0.1
            
    return node_edge_color, node_edge_width

def edge_color_width(H, cmp_edges):
    
    dict_edge_color_rgba = {(u,v): cmp_edges.colors[d['weight']] for u,v,d in H.edges(data=True)}
    edge_width = {(u,v): d["width"] for u,v,d in H.edges(data=True)}
    
    return dict_edge_color_rgba, edge_width 


def legend_plot(node_label, plant_label_color, dict_att_to_color, H, func_to_color, node_att="type", label= False):
    legend_elements = []
    if node_att =="type": 
        legend_elements.append(Line2D([0], [0], marker='o', label= "$\\bf{Nodes}$", markerfacecolor='w', markersize=0,color='w') )    
    for elem in node_label.items():
        if node_att=="type" and label ==True:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{elem[1]}: {elem[0]}', markerfacecolor='g', markersize=0))

      
    if node_att=="type":
        legend_elements.append(Line2D([0], [0], marker='o', color="w", markerfacecolor=dict_att_to_color['plant'], label='Plant species', markersize=15))
        for rest in set(dict_att_to_color.keys())-{'plant'}:

            if rest == 'herbivory':
                legend_elements.append(Line2D([0], [0], marker='o', color="w", label='Herbivores', markerfacecolor=dict_att_to_color[rest], markersize=15))
            elif rest == 'pollination':
                legend_elements.append(Line2D([0], [0], marker='o', color="w", label='Pollinators', markerfacecolor=dict_att_to_color[rest], markersize=15))
            elif rest == 'seed dispersal':
                legend_elements.append(Line2D([0], [0], marker='o', color="w", label='Seed dispersers', markerfacecolor=dict_att_to_color[rest], markersize=15))
            elif rest == 'decomposition':
                legend_elements.append(Line2D([0], [0], marker='o', color="w", label='Saprotrophic-fungi', markerfacecolor=dict_att_to_color[rest], markersize=15))
            elif rest == 'fungal pathogenicity':
                legend_elements.append(Line2D([0], [0], marker='o', color="w", label='Pathogenic fungi', markerfacecolor=dict_att_to_color[rest], markersize=15))
            elif rest == 'nutrient uptake':
                legend_elements.append(Line2D([0], [0], marker='o', color="w", label='Symbiotic fungi', markerfacecolor=dict_att_to_color[rest], markersize=15))
            else:
                legend_elements.append(Line2D([0], [0], marker='o', color="w", label=rest.capitalize(), markerfacecolor=dict_att_to_color[rest], markersize=15))
    legend_elements.append(Line2D([0], [0], marker='o', label= "$\\bf{Edges}$", markerfacecolor='g', markersize=0,color='w') )  
    for i in set(np.unique([d['interaction_type'] for u,v,d in H.edges(data=True)])):
        color = func_to_color[i]
        legend_elements.append(Line2D([0], [0], color=color, lw=4, label=i.capitalize()))

    if node_att !="type" or label==False: 
       legend_elements.append(Line2D([0], [0], marker='o', color='w', label='19: Withania frutescens', markerfacecolor='g', markersize=0))

    return legend_elements
