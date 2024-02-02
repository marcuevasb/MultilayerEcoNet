# MultilayerEcoNet
The architecture of Multilayer Ecological Networks

All computational analysis was made using Python 3.10.8 running in  a Linux desktop (version 22). Dependencies are listed at the begining of each file.
python3_enviroment contains the requirements.txt and enviroment_droplet.yml files needed to recreate a python environment with all libraries adn dependencies needed.

1- FullCode.ipynb is a notebook that contains most of the results shown in the main manuscript and in the Suplementary Information. The functions used are stored in FullCodeFunctions.py.
2- The folder *data* contains the data used and a small testing dataset that can be used to run tests, created using create_syntetic_data.ipynb.
3- Two very resource consuming plots have been separated from the rest. They are in the folder MFENplot, wchich contains two python scripts to depict the Multifunctional Ecological Network and a third one with the functions used in both.

# Null models

In Null_models.ipynb:

1- Import the resource-function matrix $\bf P$, computed from the RCF tensor of the Na Redona dataset based on Eq.1 of the paper.

2- Visualize $\bf P$ along with some randomizations and compute NODF for binarizations.

3- Build nestedness-based rankings (ordering species and functions in terms of their participation strength in $\bf P$), along with suitably defined null models.

4- Build and plot $\Phi$ and $\Pi$

5- Look into hierarchical clustering properties of $\bf P$.

6- Compute the conditioned $\Phi|_i$ and $\Pi|^\alpha$, and the keystonness scores

7- Build a null model for the keystonness scores


# Nestedness 

NODF_WNODF contains all the necesary material to reproduce NODF and WNODF results from the main text using fortran90. 

For NODF:

$ f90 -O3 read_n.f. nest_nodf.f dranxor.f -o nest.x

For WNODF:

$ f90 -O3 read_w.f. nest_wnodf.f dranxor.f -o nest_w.x

The random number generator used is:

https://ifisc.uib-csic.es//raul/CURSOS/Stochastic_Simulation_Methods/dranxor.f90