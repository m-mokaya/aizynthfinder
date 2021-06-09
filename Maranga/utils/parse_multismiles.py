import os 
import sys

sys.path.append('../../')

import pandas as pd
import numpy as np
import json

from rdkit.Chem import rdChemReactions

import aizynthfinder.analysis
from aizynthfinder.mcts.node import Node
from aizynthfinder.analysis import ReactionTree
from aizynthfinder.mcts.state import State
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.context.stock import StockException
import aizynthfinder.context.scoring as scoring

import aizynthfinder.context.config as con

#read in .json file => 
def read_json(filename):
    with open(filename, 'r') as input:
        data = json.load(input)
    reaction_trees = data['reaction trees']

    json_reaction_trees = [json.loads(i) for i in reaction_trees]
    return json_reaction_trees

#read in hdf file from aizynthfinder output
def read_hdf(filename):
    data = pd.read_hdf(filename, 'table')
    return data

#return dataframe only contained solved molecules
def collect_solved(data_df):
    solved_data = data_df.loc[(data_df.is_solved==True)]
    return solved_data

#collect trees from dataframe (returns a list of dicts)
def collect_trees(data_df):
    trees = data_df.trees.values
    return trees

def collect_scores(data_df):
    top_scores = data_df.top_scores.values
    return top_scores

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


def generate_solved_dicts_from_hdf(filename):
    data = read_hdf(filename)
    solved_data = collect_solved(data)
    solved_trees = collect_trees(solved_data)

    dict_list  = []
    for i in solved_trees:
        for itree, tree in enumerate(i):
            solved_reaction = ReactionTree.from_dict(tree)
            dict_list.append(solved_reaction)
    return dict_list

def dict_list_to_json(dict_list, filename):
    json_list = [dict_list[i].to_json() for i in dict_list]

    output = {
        'reaction trees': json_list
    }
    
    with open(filename, 'w') as outfile:
        json.dump(output, outfile)

    return json_list

def generate_images_from_hdf(filname, output_folder):
    dict_list = generate_solved_dicts_from_hdf(filname)
    try:
        for i, tree in enumerate(dict_list):
            imagefile = os.path.join(output_folder, 'route_'+str(i)+'.png')
            tree.to_image().save(imagefile)
        return True
    except:
        print('Error saving images')
        return False

def calculate_cost_from_trees(tree_list, configfile):
    #initalise config 
    config = con.Configuration()
    config = config.from_file(configfile)

    policy_scorer = scoring.USPTOModelPolicyProbabilityScorer()
    cost_scorer = scoring.PriceSumScorer(config=config)

    scores = []
    for ind, i in enumerate(tree_list):
        tree = ReactionTree.from_dict(i)
        cost = cost_scorer(tree)
        policy = policy_scorer(tree)
        if (policy == 0):
            print('Zero policy')
            #scores.append(cost/0.001)
        else:
            scores.append(cost/policy)

    #scores = [cost_scorer(ReactionTree.from_dict(i))/policy_scorer(ReactionTree.from_dict(i)) for i in tree_list]
    return scores

def calculate_route_cost(reactions, stock_inchis, stock):
    not_in_stock_multiplier = 10

    leaves = [list(i.leafs()) for i in reactions]
    costs = []
    for leaf in leaves:
        prices = []
        for mol in leaf:
            inchi = mol.inchi_key
            # check if mol in stock
            if inchi in stock_inchis:
                #print(str(mol)+' in stock')
                # get index of item
                index = stock[stock['inchi_key']==inchi].index.values
                # print('Index: ', index)
                # get price value
                try:
                    price = stock.iloc[index[0]]['price']
                except:
                    print('Index: ', index)
                    print('Did not work. Default Cost.')
                    price = 1.8 * not_in_stock_multiplier
                # print('Price: ', price)
                prices.append(price)
            else:
                #print(str(mol)+' NOT in stock')
                price = 1.8 * not_in_stock_multiplier
                #print('Price: ', price)
                prices.append(price)
        costs.append(prices)
    
    prices = [sum(i) for i in costs]

    policy_scorer = scoring.USPTOModelPolicyProbabilityScorer()
    policies = [policy_scorer(i) for i in reactions]

    overal_costs = [] 
    for price, policy in zip(prices, policies):
        if policy == 0:
            continue
        else:
            overal_costs.append(price/policy)

    return overal_costs




