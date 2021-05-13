# takes in a list[rxns] or list of list[rxns], returns a list of fingerprints for each reactions.

import os
import sys
import argparse

import json
import pandas as pd
import numpy as np
import scipy

sys.path.append('../../')

import Maranga.utils.parse_multismiles as mutils
import aizynthfinder.context.scoring as scoring
import aizynthfinder.context.config as con

from aizynthfinder.analysis import ReactionTree
from rdkit.Chem import rdChemReactions
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import euclidean


#determine whether input is reactions for several molecules or single molecule. 
def parse_input(input, type):
    """
    Returns list of dictionaries representing each reaction.
    
    :param: list or list of list reactions as dictionaries. 
    :return: list of dictionaries.
    """
    if isinstance(input[0], list):
        print('Multiple Molecules')
        #reactions for several molecules
        all_dict = []
        prev = 0
        for ind, mol in enumerate(input):
            dict = [{'molecule': ind, 'index': num+prev, 'reaction': i, 'type': type} for num, i in enumerate(mol)]
            all_dict.extend(dict)
            prev+=len(mol)
        return all_dict
    else:
        print('Single Molecule')
        #reactions for single molecule
        dict = [{'index': ind, 'reaction': i} for i in enumerate(input)]
        print('Parse input (dict example): ', dict[0])
        return dict

    
#calcualte reaction fingerprints
def generate_fingerprints(rxn_dict):
    """
    Adds fingerprints to list of dictionaries.
    
    :param: list of reaction informations (in dict)
    :return: list of reactions informations with finger added. 
    """

    #generates a list of smarts for each individual reaction
    smarts_str = [list(mutils.findkeys(i.get('reaction'), 'smiles')) for i in rxn_dict]  
    smarts = [mutils.remove_non_smarts(i) for i in smarts_str]

    #flatten smarts list.
    all_smarts = []
    for i in smarts:
        all_smarts.extend(i)
    
    #generate set of unique smarts. basis for fingerprints
    smarts_set = sorted(set(all_smarts), key=all_smarts.index)
    uni_smarts = list(smarts_set)
    print('Print lengh of reaction vector: ', len(uni_smarts))
    
    
    #generate fingerprints from unique smarts and reacion dictionary
    for index, item in enumerate(smarts):
        vector = []

        for i in uni_smarts:
            vector.append(item.count(i))
        
        rxn_dict[index]['fingerprint'] = vector
    
    return rxn_dict


#calculate euclidian distances between vectors
def calculate_distances(rxn_dict):
    """
    Calculates distances between every pair of routes in a given list of reactions. 

    :param: list of dictionaries with 'fingerprint' key
    :return: list of distances, list of dict(with max dist key for each route), list of largest distances
    """

    #calculate distances between all pairs and return list
    fingerprints = [i.get('fingerprint') for i in rxn_dict]
    npvec = np.array(fingerprints)
    distances = euclidean_distances(npvec, npvec)
    
    #determine the largest distance to all other routes and add to rxn_dict
    largest_distances = []
    for index1, item1 in enumerate(fingerprints):
        dist = []
        for index2, item2 in enumerate(fingerprints):
            if index1 != index2:
                dist.append(euclidean(item1, item2))
        largest_distances.append(min(dist))
        rxn_dict[index1]['max distance'] = min(dist)
    
    return distances, largest_distances, rxn_dict

def jaccard_binary(x,y):
    """A function for finding the similarity between two binary vectors"""
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = intersection.sum() / float(union.sum())
    return similarity

def new_jaccard_similarity(x,y):
    """Function that determines the similarity between who vectors

    :param: orderedvectors, x and y
    :return: similarity score between 0 and 1
    """
    a = 0
    b = 0
    for index, item in enumerate(x):
        if (item == y[index]):
            a += 1
        else:
            b += 1
    return a/(a+b)




def calculate_similarity(rxn_dict):
    """
    Calculate pairwise Jaccard distances

    :param: list of dictionaries with 'fingerprint' key
    :return: list of distances, list of dict(with max dist key for each route), list of largest distances
    """
    
    fingerprints = [i.get('fingerprint') for i in rxn_dict]
    npvec = np.array(fingerprints)

    dists = []
    nfps = len(fingerprints)

    #conver the fingerprints to binary fingerprints
    binary_fingerprints = []
    for i in fingerprints:
        f = []
        for x in i:
            if (x != 0):
                f.append(1)
            else:
                f.append(0)
        binary_fingerprints.append(f)

    #calculate pairwise jaccard distances between all pairs
    sims = []
    for index1, item1 in enumerate(fingerprints):
        for index2, item2 in enumerate(fingerprints[index1:]):
            sims.append(new_jaccard_similarity(item1, item2))

    #add the smallest similarity value for each fingerprint
    smallest_sim = []
    largest_sim = []
    for index1, item1 in enumerate(fingerprints):
        sim = []
        for index2, item2 in enumerate(fingerprints):
            if index1 != index2:
                sim.append(new_jaccard_similarity(item1, i))
        m = max(sim)
        mi = min(sim)
        largest_sim.append(m)
        smallest_sim.append(mi)
    rxn_dict[index1]['max sim'] = m

    #determine the number of steps in each reaction return as list
    lengths = [sum(1 if i != 0 else 0 for i in items) for items in fingerprints]

    return sims, smallest_sim, largest_sim, rxn_dict, lengths 

#calcaute largest similarity of each explore reaction pathway to all normal reactions.
def split_similarity(e, s):
    largest = []
    lengths = []
    for rxn in e:
        rxn_fp = rxn.get('fingerprint')
        sims = [new_jaccard_similarity(rxn_fp, i.get('fingerprint')) for i in s]
        largest.append(max(sims))
        lengths.append(sum(1 if i != 0 else 0 for i in rxn_fp))
    return largest, lengths
        

        



    