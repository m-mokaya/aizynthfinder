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
from scipy.spatial.distance import euclidean

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


def get_top_scores(input):
    data = mutils.read_hdf(input)

    solved_data = data.loc[(data.is_solved==True)]

    top_scores = solved_data.top_scores.values
    topscore = solved_data.top_score.values

    all = []
    for i in top_scores:
        new = [float(s) for s in i.split(',')]
        all.append(new)


    means = [np.mean(i) for i in all]

    return  topscore 


#calculates the state score for each reaction
def collect_states_scores(data, conf):
    config = con.Configuration().from_file(conf)
    state_scorer = scoring.StateScorer(config=config)
    #rxns = [ReactionTree.from_dict(tree) for tree in data]
    all_scores = state_scorer(data)
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


#clusters fingerprints given user defined cutoff => list of clusters : [[cluster 1 indexes], [cluster 2 indexes]]
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

#from list of vectors determine those with distance greater than the  cutoff. Returns list of novel reactions
def find_novel(vector_list, cutoff):
    novel = []
    for i in vector_list:
        for r in vector_list:
            if euclidean(i[2], r[2]) >= cutoff:
                novel.append(i)
    return novel

def main_json(input, output, cutoff, ind):
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


def main_all_hdf(input, output, cutoff):
    reactions = collect_rxns(input)

    nreactions = len(reactions)

    fingerprints, all_fingerprints = generate_fingerprints(reactions)
    clusters = cluster_reactions(all_fingerprints=all_fingerprints, cutoff=cutoff)
    new_fingerprints = add_cluster_info(clusters, all_fingerprints)
    vectors = generate_vectors(nreactions, new_fingerprints, clusters)
    #distances = calculate_distances(vectors)
    
    print('Num Reactions: ', nreactions)
    #print('Mean Distance: ', np.mean(distances))
    #print('Standard Deviation', np.std(distances))

    if os.path.exists(output) == False:
        os.mkdir(output)

    return nreactions, vectors


def main_single_hdf(input, output, cutoff, ind):

    reactions = collect_rxns(input)

    nreactions = len(reactions)

    fingerprints, all_fingerprints = generate_fingerprints(reactions)
    clusters = cluster_reactions(all_fingerprints=all_fingerprints, cutoff=cutoff)
    new_fingerprints = add_cluster_info(clusters, all_fingerprints)
    vectors = generate_vectors(nreactions, new_fingerprints, clusters)
    distances = calculate_distances(vectors)
    
    #print('Num Reactions: ', nreactions)
    #print('Mean Distance: ', np.mean(distances))
    #print('Standard Deviation', np.std(distances))

    if os.path.exists(output) == False:
        os.mkdir(output)

    return nreactions, np.mean(distances), np.std(distances), vectors, distances


    #plot_dist_hist(distances, output, ind)
    #plot_heatmap(distances, output, ind)



def main_multi_all(input, output, cutoff):
    import math
    data = parse_multi(input)

    c = get_top_scores(input)
    


    flat_data = []
    for i in data:
        flat_data.append(i)

    nreaction, vectors = main_all_hdf(flat_data, output, cutoff)

    distances = []
    count =0
    for ind, i in enumerate(vectors):
        euc = []
        for j in vectors[ind:]:
            euc.append(euclidean(i[1], j[1]))
        euc_ = [i for i in euc if i >= 35]
        if len(euc_) != 0:
            print('Novel: ',i[0])
            distances.append(i)
    
    print(len(distances))
    print('example: ', distances[:15])


    v1 = [v[1] for v in vectors]


    d = {
        'cost': c,
        'vectors': v1
    }

    '''df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(output, 'results_2.csv'))'''


    import matplotlib.pyplot as plt



    ''' 
    plt.hist(distances, bins=15)
    plt.savefig(os.path.join(output, 'all_dist.png'))'''


    
def main_mutli(input, output, cutoff):
    data = parse_multi(input)

    c = get_top_scores(input)
    #print(c)

    n = []
    m = []
    s = []
    v = []
    v2 = []

    prev = 0

    nov = []

    for ind, i in enumerate(data):
        num, mean, std, vectors, distances = main_single_hdf(i, output, cutoff, ind)
        n.append(num)
        m.append(mean),
        s.append(std)
        v1 = [[ind, prev+v[0], v[1]] for v in vectors]
        v.extend(v1)
        v2.append(v1)
        prev+= len(vectors)



    
    novel = []
    for i in v2: 
        molv = []
        for indx, r1 in enumerate(i):
            euc = []
            for r2 in i[indx:]:
                euc.append(euclidean(r1[-1], r2[-1]))
            euc_ = [i for i in euc[1:] if i >= 3.0]
            if (len(euc_) == (len(euc[1:])) and len(euc_) != 0):
                novel.append(r1)

    

    print('Nov len: ', len(novel))
    print('Nov: ', novel)




    d = {
        'nreactions': n, 
        'mean': m,
        'deviation': s,
        'cost': c,
        'vectors': v2
    }

    #df = pd.DataFrame(data=d)
    #df.to_csv(os.path.join(output, 'results.csv'))

    print(mean)
    import matplotlib.pyplot as plt

    #plt.plot(distances)
    #plt.show()
    #plt.savefig(os.path.join(output, 'all_hist_'+str(ind)+'.png'))
    
    #novel = find_novel(v, 2.5)
    #print('Len Novel: ', len(novel))


    


if __name__ == "__main__":
    print('Starting')
    if args.multi == 's':
        print('Single')
        main_json(args.input, args.output, args.cutoff, 0)
    elif args.multi == 'm':
        print('multi')
        main_mutli(args.input, args.output, args.cutoff)
    else:
        main_multi_all(args.input, args.output, args.cutoff)
        print('multi all')
    print('Complete')
    