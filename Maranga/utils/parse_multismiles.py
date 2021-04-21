import os 
import sys

sys.path.append('../../')

import pandas as pd
import numpy as np

from rdkit.Chem import rdChemReactions

import aizynthfinder.analysis
from aizynthfinder.mcts.node import Node
from aizynthfinder.analysis import ReactionTree
from aizynthfinder.mcts.state import State
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.context.stock import StockException

#read in hdf file from aizynthfinder output
def read_hdf(filename):
    data = pd.read_hdf(filename, 'table')
    return data

#return dataframe only contained solved molecules
def collect_solved(data_df):
    solved_data = data_df.loc[(data_df.solved==True)]
    return solved_data

#collect trees from dataframe (returns a list of dicts)
def collect_trees(data_df):
    trees = data_df.trees.values
    return trees

#pull out list of smarts for each reaction
def collect_smarts(tree_dict):
    return None

def findkeys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x

#remove all reactants
def remove_non_smarts(smiles_list):
    rxns = []
    for i in smiles_list:
        if '>>' in i:
            rxns.append(i)
    return rxns

#generate reactions
def generate_rxns(rxn):
    return rdChemReactions.ReactionFromSmarts(rxn)



