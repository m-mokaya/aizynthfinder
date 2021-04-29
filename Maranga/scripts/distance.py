import os
import sys
import argparse

import json
import pandas as pd
import numpy as np


sys.path.append('../../')

import Maranga.utils.parse_multismiles as mutils
import aizynthfinder.context.scoring as scoring
import aizynthfinder.context.config as con

from aizynthfinder.analysis import ReactionTree
from rdkit.Chem import rdChemReactions
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from sklearn.metrics.pairwise import euclidean_distances


#import file from either hdf file or json file
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=False, help='filename/location of results file')
parser.add_argument('--config', required=True, help='file location of config file')
parser.add_argument('--input', required=True, help='input .hdf5 file location')
parser.add_argument('--multi', required=False, help='is input data for single or mutli smiles')
parser.add_argument('--cutoff', required=True, help='cutoff for reactions clustering (Butina clustering)')
args = parser.parse_args()

#parses file to generate either 1. [[{}, {}], [{}, {}]] or list of dictionaries (single molecule)
def parse_input(input):
    if '.hdf5' in input:
        print('passed')
        return input
    else:
        return mutils.read_json(input)

def parse_multi(input):
    data = mutils.read_hdf(input)

    solved_data = data.loc[(data.is_solved==True)]
    unsolved_data = data.loc[(data.is_solved==False)]
    all_solved = data.is_solved.values

    #all_trees = data.trees.values
    all_solved_trees = solved_data.trees.values
    #all_unsolved_trees = unsolved_data.trees.values

    mols = []
    for mol in all_solved_trees:
        dicts = [i for i in mol]
        mols.append(dicts)
    return mols


#calculates the state score for each reaction
def collect_states_scores(data, conf):
    config = con.Configuration().from_file(conf)
    state_scorer = scoring.StateScorer(config=config)
    rxns = [ReactionTree.from_dict(tree) for tree in data]
    all_scores  = state_scorer(rxns)
    return all_scores


#collects returns list of list of reactants => [[rxn, rxn], [rxn, rxn]] from list of dict => [{}, {}]
def collect_rxns(data):
    smiles_smarts_gen = [mutils.findkeys(i, 'smiles') for i in data]
    smiles_smarts = [list(i) for i in smiles_smarts_gen]
    smarts = [mutils.remove_non_smarts(i)for i in smiles_smarts]

    #convert smarts to rxns
    rxns = []
    for i in smarts:
        rxn = [mutils.generate_rxns(r) for r in i]
        rxns.append(rxn)
    return rxns


#takes in lsit of llist of reactions => lsitof list of fingerprints
def generate_fingerprints(data):
    fingerprints = []
    all_fingerprints = []

    previous_len = 0
    for index, pathway in enumerate(data):
        fngr = [[index, rindex+previous_len, rdChemReactions.CreateStructuralFingerprintForReaction(r)] for rindex, r in enumerate(pathway)]
        fingerprints.append(fngr)
        all_fingerprints.extend(fngr)
        previous_len += len(pathway)

    return fingerprints, all_fingerprints


#clusters fingerprints given user defined cutoff => 
def cluster_reactions(all_fingerprints, cutoff):
    dists = []

    for i in range(len(all_fingerprints)):
        sims = DataStructs.BulkTanimotoSimilarity(all_fingerprints[i][2], [x[2] for x in all_fingerprints[:i]])
        dists.extend([1-x for x in sims])
    clusters = Butina.ClusterData(dists, len(all_fingerprints), float(cutoff), isDistData=True)
    return clusters


#add cluster to each reaction info list (in all_fingerprints => [[pathway indx, rxn index, fingerprint, cluster]])
def add_cluster_info(clusters, all_fingerprints):
    for index in range(len(clusters)):
        for val in clusters[index]:
            all_fingerprints[val].append(index)
    return all_fingerprints

#generate vectors for each reaction => [[vector], [vector]]
def generate_vectors(nreactions, all_fingerprints, clusters):
    all_vectors = []
    for r in range(nreactions):
        rxns = [i for i in all_fingerprints if i[0] == r]
        vector_dict = dict.fromkeys(range(len(clusters)), 0)
        for rxn in rxns:
            vector_dict[rxn[-1]] += 1
        all_vectors.append(vector_dict)
    rxn_vectors = [[indx, list(i.values())] for indx, i in enumerate(all_vectors)]
    return rxn_vectors

def calculate_distances(vectors):
    vectors = [i[1] for i in vectors]
    npvec = np.array(vectors, dtype=object)
    return euclidean_distances(npvec, npvec)


def plot_dist_hist(distances, output, ind):
    fdist = [item for sublist in distances for item in sublist]

    import matplotlib.pyplot as plt

    plt.hist(fdist, bins=15)
    plt.savefig(os.path.join(output, 'hist_'+str(ind)+'.png'))
    #plt.show()

def plot_heatmap(distances, output, ind):
    import matplotlib.pyplot as plt

    plt.imshow(distances, cmap='hot')
    plt.colorbar()
    plt.savefig(os.path.join(output, 'heatmap_'+str(ind)+'.png'))
    #plt.show()




def main(input, output, cutoff, ind):
    data = parse_input(input)

    reactions = collect_rxns(data)

    nreactions = len(reactions)

    fingerprints, all_fingerprints = generate_fingerprints(reactions)
    clusters = cluster_reactions(all_fingerprints=all_fingerprints, cutoff=cutoff)
    new_fingerprints = add_cluster_info(clusters, all_fingerprints)
    vectors = generate_vectors(nreactions, new_fingerprints, clusters)
    distances = calculate_distances(vectors)
    
    print('Num Reactions: ', nreactions)
    print('Mean Distance: ', np.mean(distances))
    print('Standard Deviation', np.std(distances))

    if os.path.exists(output) == False:
        os.mkdir(output)


def main_single(input, output, cutoff, ind):

    reactions = collect_rxns(input)

    nreactions = len(reactions)

    fingerprints, all_fingerprints = generate_fingerprints(reactions)
    clusters = cluster_reactions(all_fingerprints=all_fingerprints, cutoff=cutoff)
    new_fingerprints = add_cluster_info(clusters, all_fingerprints)
    vectors = generate_vectors(nreactions, new_fingerprints, clusters)
    distances = calculate_distances(vectors)
    
    print('Num Reactions: ', nreactions)
    print('Mean Distance: ', np.mean(distances))
    print('Standard Deviation', np.std(distances))

    if os.path.exists(output) == False:
        os.mkdir(output)

    return nreactions, np.mean(distances), np.std(distances)


    plot_dist_hist(distances, output, ind)
    plot_heatmap(distances, output, ind)
    
def main_mutli(input, output, cutoff):
    data = parse_multi(input)

    n = []
    m = []
    s = []

    for ind, i in enumerate(data):
        num, mean, std = main_single(i, output, cutoff, ind)
        n.append(num)
        m.append(mean),
        s.append(std)

    d = {
        'nreactions': n, 
        'mean': m,
        'deviation': s
    }

    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(output, 'results.csv'))

    print(mean)
    import matplotlib.pyplot as plt

    plt.hist(m, bins=15)
    plt.savefig(os.path.join(output, 'all_hist_'+str(ind)+'.png'))
    



if __name__ == "__main__":
    print('Starting')
    if args.multi == False:
        print('Single')
        main(args.input, args.output, args.cutoff, 0)
    else:
        print('multi')
        main_mutli(args.input, args.output, args.cutoff)
    print('Complete')