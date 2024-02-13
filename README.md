# MultilayerEcoNet

GitHub repository for the manuscript "On the structure of species-function participation in multilayer ecological networks" and its supplementary information. Here, you'll find all the necessary materials to reproduce all results contained in the manuscript, including input data stored in the 'data/input' folder and output results stored in the 'data/output' folder.

# P_processing

All computational analysis was made using Python 3.10.8 running in  a Linux desktop (version 22). Dependencies are listed at the beginning of each file.
python3_enviroment contains the requirements.txt and enviroment_droplet.yml files needed to recreate a python environment with all libraries adn dependencies needed. To install the environment with conda, run conda env create -f environment_droplet.yml.

1- `P_processing` is a folder that contains the necessary material to reproduce most of the results shown in the main manuscript and in the Suplementary Information.
2- `FullCode.ipynb` is the notebook, and functions used are stored in `FullCodeFunctions.py`.
3- Two very resource consuming plots have been separated from the rest. They are in the folder MFENplot, wchich contains two python scripts to depict the Multifunctional Ecological Network (MFEN) and a third one with the functions used in both.

# P analytics

In `P_analytics.ipynb`:

1- Import the resource-function matrix $\mathbf{P}$, computed from MFEN of the Na Redona dataset based on Eq.1 of the paper generated in `FullCode.ipynb`.
2- Visualize $\bf P$ along with some randomizations and compute NODF for binarizations.
3- Build nestedness-based rankings (ordering species and functions in terms of their participation strength in $\bf P$), along with suitably defined null models.
4- Build and plot $\Phi$ and $\Pi$.
5- Look into hierarchical clustering properties of $\mathbf{P}$.
6- Compute the conditioned $\Phi|_i$ and $\Pi|^\alpha$, and the keystonness scores.
7- Build a null model for the keystonness scores.

# Nestedness 

`NODF_WNODF` contains all the necessary material to reproduce NODF and WNODF results from the main text using fortran90 using the resource-function matrix $\mathbf{P}$ of the paper generated in `FullCode.ipynb`. 

For NODF:
$ f90 -O3 read_n.f. nest_nodf.f dranxor.f -o nest.x

For WNODF:
$ f90 -O3 read_w.f. nest_wnodf.f dranxor.f -o nest_w.x

The random number generator used is:
https://ifisc.uib-csic.es//raul/CURSOS/Stochastic_Simulation_Methods/dranxor.f90
