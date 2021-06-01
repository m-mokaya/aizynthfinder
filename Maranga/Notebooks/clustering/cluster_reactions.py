import os 
import sys

sys.path.append('../../')

import pandas as pd
import numpy as np
import sklearn
import json
import argparse

from rdkit.Chem import rdChemReactions
from rdkit import DataStructs

import aizynthfinder.analysis
from aizynthfinder.mcts.node import Node
from aizynthfinder.analysis import ReactionTree
from aizynthfinder.mcts.state import State
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.context.stock import StockException

import Maranga.utils.parse_multismiles as mutils
from Maranga.utils.butina_clustering import SmartsCluster

#input_filename = '/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/explore/results/tg1_explore_1.hdf5'
#config_file = '/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/explore/config.yml'

#setup up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=False, help='filename/location of results file')
parser.add_argument('--config', required=True, help='file location of config file')
parser.add_argument('--input', required=True, help='input .hdf5 file location')
args = parser.parse_args()

print('Reading In data')

#read in data file
data = mutils.read_hdf(args.input)
solved_data = data.loc[(data.is_solved==True)]
solved_trees = mutils.collect_trees(solved_data)

'''#create list of all pathways scores => Route cost scorer
top_scores = []
for int, i in enumerate(solved_trees):
    val = mutils.calculate_cost_from_trees(i, args.config)
    top_scores.append(val)

# all scores
all_scores = []
for i in top_scores:
    for p in i:
        all_scores.append()'''

#create list containing all pathway 'top_scores'
top_scores = solved_data.top_scores.values
all_scores = [float(s) for i in top_scores for s in i.split(',')]
print(len(all_scores))


print('Calculated routes')

#pull out smile/smarts for each => list of lists
smiles_smarts_gen = [[index, mutils.findkeys(pathway, 'smiles')] for index, molecule in enumerate(solved_trees) for pathway in molecule]
smiles_smarts = [[i[0], list(i[1])] for i in smiles_smarts_gen]
smarts = [[i[0], mutils.remove_non_smarts(i[1])] for i in smiles_smarts]

#convert list of list of smarts to list of rxns => [[rnx1, rnx2, ..], [rxn2, rxn2, ...]]
pathway_rxns = []
for pathway in smarts:
    rxns = [pathway[0]]
    actual_rxns = [mutils.generate_rxns(i) for i in pathway[1]]
    rxns.append(actual_rxns)
    pathway_rxns.append(rxns)


num_reactions = len(pathway_rxns)
print('Num: Reactions', num_reactions)


#generate reaction finderprint for eac hreaction => [[rnx1fp, rnx2fp, ..], [rxn2fp, rxn2fp, ...]]
pathway_fps = []
all_fingerprints= []

count = 0
previous_length = 0
for index, pathway in enumerate(pathway_rxns):
    fingerprints = [[pathway[0], index, rindex+previous_length, rdChemReactions.CreateStructuralFingerprintForReaction(reaction)] for rindex, reaction in enumerate(pathway[1])]
    all_fingerprints.extend(fingerprints)
    previous_length += len(pathway[1])


#butina cluster all the reactions
from rdkit.ML.Cluster import Butina

cutoff = 0.2

dists = []

for i in range(len(all_fingerprints)):
    sims = DataStructs.BulkTanimotoSimilarity(all_fingerprints[i][3], [x[3] for x in all_fingerprints[:i]])
    dists.extend([1-x for x in sims])

clusters = Butina.ClusterData(dists, len(all_fingerprints), cutoff, isDistData=True)

print('clustered reactions')

print('all_fingerprints: ', all_fingerprints[:3])
print(len(all_fingerprints))

#add cluster to each reaction info list (in all_fingerprints => [[mol, pathway, rxn index, fingerprint, cluster]])
for index in range(len(clusters)):
    for val in clusters[index]:
        all_fingerprints[val].append(index)
        all_fp_dict[val]['cluster'] = index


#convert each fingperint to a dict representing each pathway 
all_vectors = []
for r in range(num_reactions):
    checklist = [i for i in all_fingerprints if i[1] == r]
    vector_dict = dict.fromkeys(range(len(clusters)), 0)
    for reaction in checklist:
        vector_dict[reaction[4]] += 1
    all_vectors.append(vector_dict)

#convert dict to vector representing each pathway
rxn_vectors = [[indx, list(i.values())] for indx, i in enumerate(all_vectors)]



print('Complete, saving json')

with open(args.output, 'w') as f:
    json.dump(all_fp_dict, f)

print('Complete')