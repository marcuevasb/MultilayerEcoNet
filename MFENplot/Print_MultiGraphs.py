#%%
import matplotlib.pyplot as plt
import sys
from Plot_Mcomm_Lib import *
from netgraph import Graph, get_community_layout
import networkx as nx
import numpy as np

'''data'''
data_path = '../data/NaRedona_observed_6layers.csv' 
abundance_path = '../data/NaRedona_plant_abundance.csv' 


# '''TESTING DATA'''
data_path = '../data/data_test.csv'
abundance_path = None


head = 0 #header
sep=','


'''saving'''
#returns saving directory a new folder in the dataser folder
net_dir, plt_dir = creat_dir(data_path)


'''Multigraph'''
# Builds the networkx object from the dataset. Returns the network G, a dict object to label plant species nodes and a dict to
# highlight plant species that participate in all ecological functions.
# G has as node attribute the plant abundance (animal species are set to the minimal abundance), 
# and the intrection type (plant species and animal/fungi within function). As node attributes it has the interaction type and the width (frequency_of_occurrence).
G, node_label, plant_label_color, node_label_name= build_G(head, sep,data_path, abundance_path)
#%%

'''Graph'''

# If the dataset is a multigraph, builds a graph  using Eq.(1) from the main manuscript.
H, multi_edges_over, multi_edges_save  = Multi_to_Graph(G,net_dir)
# Calculates communities using Infomap setting them as node attributes and a dic object of every node and their community.
H, community_dict_inf = inf_communities(H) # we obtain communities using the converted multiedge
H.add_edges_from(multi_edges_save) #we change the edges to the correct width of one multiedge and plot the other appart


"""Colors from types"""
# returns a dic of nodes colors according to species, Louvain communities and Infomap communities.
# additionally returns a dic object of function interaction type: color or community: color.
node_colors_dict_type, func_to_color, node_edge_color_type = color_of_nodes_edges(H, node_att = 'type')
# node_colors_dict_comm_inf, comm_to_color_inf,node_edge_color_inf  = color_of_nodes_edges(H, node_att = "community inf")


"""nodes"""
# Dict object with the size of the nodes. Since abundacnes are very heterogeneous they are plotted using:
# f_{i,x}^alpha / (f.max()*3) .
node_size_dict =  size_of_nodes(H, node_att = 'abundance')


""" edges """
# Edge color accounts for the ecological function involved and width to the frequency of occurrence.
edge_color_type = {(u,v): func_to_color[d['interaction_type']] for u,v,d in H.edges(data=True)}
edge_width = {(u,v): d['width'] for u,v,d in H.edges(data=True)}


fontsize =  14
for real in range(1):
    print(real)
    alpha = 0.5
    
    '''layout'''
    # The network layout is obtained using Infomap communities
    node_layout_inf =get_community_layout(list(H.edges()), node_to_community=community_dict_inf, origin=(0, 0), scale=(1, 1))

    '''multiedges'''
    multi_edges_tuple = [(u,v) for u,v,d in multi_edges_over]
    multi_edge_color= {(u,v):func_to_color[d['interaction_type']] for u,v,d in multi_edges_over}
    multi_edge_width = {(u,v):d['width'] for u,v,d in multi_edges_over}

    for u,v,d in multi_edges_over:
        multi_edge_color[(v,u)] = func_to_color[d['interaction_type']]
    for u,v,d in multi_edges_over:
        multi_edge_width[(v,u)] = d['width']

    '''Plots'''       

    '''
    Infomap Layout
    ---------------------------------------------
    '''

    '''Functional species plot layout Infomap'''

    ''' Node label in node'''

    # Node colors account for plants and animal animal/fungi species within ecological functions. 
    # Edge color accounts for different functional connections and edge width is proportional to the frequency of occurrence.
    # Layout using Infomap communities

    # legend withut labels. node plant species are labeled with plant species name
    legend_elements_type = legend_plot(node_label,plant_label_color, func_to_color, H, func_to_color, node_att="type")

    ### Plot
    f, ax = plt.subplots(dpi = 300, figsize=(12,12))

    #place nodes such that nodes belonging to the same community are grouped together
    Graph(H,
        node_color=node_colors_dict_type, 
        edge_alpha=alpha,
        node_layout=node_layout_inf, 
        node_layout_kwargs=dict(node_to_community=community_dict_inf),
        edge_layout='bundled', 
        edge_layout_kwargs=dict(k=2000),
        node_size=node_size_dict, 
        edge_width=edge_width,
        ax=ax, 
        node_labels =node_label_name, 
        ode_label_fontdict = {'size':fontsize},
        node_label_offset =0.08, 
        node_edge_color= node_edge_color_type,  
        node_edge_width =0.2,
        edge_color = edge_color_type,
    )
    #add multiledges
    M = H.edge_subgraph(multi_edges_tuple)
    Graph(M,
        node_color=node_colors_dict_type, 
        edge_alpha=alpha,
        node_layout=node_layout_inf, 
        node_layout_kwargs=dict(node_to_community=community_dict_inf),
        edge_layout='bundled', 
        edge_layout_kwargs=dict(k=2000),
        node_size=node_size_dict,
        node_edge_width =0.2,
        node_edge_color= node_edge_color_type,
        edge_width=multi_edge_width,
        ax=ax,
        edge_color = multi_edge_color, 
        node_alpha = 0,
    )

    l= ax.legend(handles=legend_elements_type,  loc='upper right', frameon=False, fontsize=fontsize,bbox_to_anchor=(0.0, 1))
    # we added a string at the end of the legend for consistency in the layouts between plots
    # this last element is plotted in white
    for n, text in enumerate( l.get_texts()):
        if n == len(legend_elements_type)-1:
            text.set_color("w")

    plt.savefig(plt_dir+ f'/real_{real}.pdf', bbox_inches='tight' )


# %%
