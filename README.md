# MultilayerEcoNet

GitHub repository for the manuscript **On the structure of species-function participation in multilayer ecological networks** and its supplementary information. Here, you'll find all the necessary materials to reproduce all results contained in the manuscript (Hervias-Parejo, S., Cuevas-Blanco, M., Lacasa, L., Traveset, A., Donoso, I., Heleno, R., ... & Eguiluz, V. M. (2023). The architecture of multifunctional ecological networks. bioRxiv, 2023-07.), including input data stored in the 'data/input' folder and output results stored in the 'data/output' folder.


# P processing

All computational analysis was made using Python 3.10.8 running in  a Linux desktop (version 22). Dependencies are listed at the beginning of each file.
python3_enviroment contains the requirements.txt and enviroment_droplet.yml files needed to recreate a python environment with all libraries adn dependencies needed. To install the environment with conda, run conda env create -f environment_droplet.yml.

1- `P_processing` is a folder that contains the necessary material to reproduce most of the results shown in the main manuscript and in the Suplementary Information.

2- `FullCode.ipynb` is the notebook, and functions used are stored in `FullCodeFunctions.py`.

3- Two very resource consuming plots have been separated from the rest. They are in the folder `MFENplot`, wchich contains two python scripts to depict the Multifunctional Ecological Network (MFEN) (`Print_MultiGraphs.py` generates Figure 2 from the main manusccripy and `Print_MultiGraphs_info_multi.py` Figure 1a from SI) and a third one with the functions used in both (`Plot_Mcomm_Lib.py`) using Netgraph.


# P analytics

In `P_analytics.ipynb`:

1- Import the resource-function matrix $\mathbf{P}$, computed from MFEN of the Na Redona dataset based on Eq.1 of the paper generated in `FullCode.ipynb`.

2- Visualize $\bf P$ along with some randomizations and compute NODF for binarizations.

3- Build nestedness-based rankings (ordering species and functions in terms of their participation strength in $\mathbf{P}$), along with suitably defined null models.

4- Build and plot $\mathbf{\Phi}$ and $\mathbf{\Pi}$.

5- Look into hierarchical clustering properties of $\mathbf{P}$.

6- Compute the conditioned $\mathbf{\Phi}|_i$ and $\mathbf{\Pi}|^\alpha$, and the keystonness scores.

7- Build a null model for the keystonness scores.


# Nestedness 

`NODF_WNODF` contains all the necessary material to reproduce NODF and WNODF results and null models from the main text using fortran90 using the resource-function matrix $\mathbf{P}$ of the paper generated in `FullCode.ipynb`. 

For NODF compile using:
```
f95 -O3 read_n.f nest_nodf.f dranxor.f90 -o nest_n.x
```
and run using:
```
./nest_n.x
```

For WNODF compile using:
```
f95 -O3 read_w.f nest_wnodf.f dranxor.f90 -o nest_w.x
```
and run using:
```
./nest_w.x
```

The random number generator used is:
https://ifisc.uib-csic.es//raul/CURSOS/Stochastic_Simulation_Methods/dranxor.f90

# Authors

Study site, field sampling and data curation:
- Sandra Hervı́as-Parejo
- Anna Traveset
- Isabel Donoso
- Ruben Heleno
- Manuel Nogales
- Susana Rodrı́guez-Echeverrı́a
  
Mathematical modelling, Network analysis, Data analysis and Simulations:
- Mar Cuevas-Blanco
- Lucas Lacasa
- Carlos J. Melian
- Victor M. Eguiluz
