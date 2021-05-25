import os
import sys
import argparse

import json
import pandas as pd
import numpy as np

sys.path.append('../../')

import Maranga.utils.parse_multismiles as mutils
import Maranga.scripts.fingerprints as fings
import aizynthfinder.context.scoring as scoring
import aizynthfinder.context.config as con

from aizynthfinder.analysis import ReactionTree
from rdkit.Chem import rdChemReactions
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean


#convert input to list of rxns
data_e = mutils.read_hdf('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/explore/results/tg1/tg1_explore_3.hdf5')
solved_data_e = data_e.loc[(data_e.is_solved==True)]
solved_trees_e = mutils.collect_trees(solved_data_e)
#convert input to list of rxns
data_s = mutils.read_hdf('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/explore/results/tg1/tg1_std_1.hdf5')
solved_data_s = data_s.loc[(data_s.is_solved==True)]
solved_trees_s = mutils.collect_trees(solved_data_s)

#parse imported reaction trees
reactions_e = fings.parse_input(solved_trees_e, 'explore')
reactions_s = fings.parse_input(solved_trees_s, 'normal')
reactions = reactions_e + reactions_s

r_e = [i.get('reaction') for i in reactions_e]
r_s = [i.get('reaction') for i in reactions_s]
rs = [ReactionTree.from_dict(i) for i in r_s]
re = [ReactionTree.from_dict(i) for i in r_e]

all_distances, min_distances = fings.split_aiz_distance(re, rs)

#plot SCATTER of all distances
import matplotlib.pyplot as plt

plt.figure(1)
plt.hist(all_distances)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/explore/results/images/all_distances.png')
plt.show()

plt.figure(2)
plt.hist(min_distances)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/explore/results/images/min_distances.png')
plt.show()

largest, lengths = fings.split_similarity(reactions_e, reactions_s)

plt.figure(3)
plt.scatter(min_distances, lengths)
plt.xlabel('Distance')
plt.ylabel('Length')
#plt.ylim(0,100)
plt.savefig('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/explore/results/images/min_distances_length.png')
plt.show()