# File to calculate the cost of a series of route 

import sys
import json
import argparse
import pandas as pd
import numpy as np

sys.path.append('../')

import aizynthfinder.analysis
import aizynthfinder.chem
import aizynthfinder.context.config as con


from aizynthfinder.mcts.node import Node
from aizynthfinder.analysis import ReactionTree
from aizynthfinder.chem import TreeMolecule, Molecule, UniqueMolecule
from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.context.stock import StockException
import aizynthfinder.context.scoring as scoring
import Maranga.utils.parse_multismiles as mutils

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='results file location')
parser.add_argument('--output', required=True, help='output file location')
parser.add_argument('--type', required=True, help='type of input file')
args = parser.parse_args()


stock_file = '/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/aizynthfinder/data/molport_in_stock.hdf5'
stock = pd.read_hdf(stock_file, 'table')
stock_inchis = stock['inchi_key'].tolist()

# collect the reaction treees as list of dicts
if args.type == 'json':
    with open(args.input) as f: 
        data = json.load(f)

    data_t = data.get('reaction trees')
    trees = [json.loads(i) for i in data_t]
elif args.type == 'hdf5':
    data = mutils.read_hdf(args.input)
    solved_data = data.loc[(data.is_solved==True)]
    trees_str = mutils.collect_trees(solved_data)
    trees= []
    for i in trees_str:
        trees.extend(i)

print('Loaded trees, calculating cost ...')
    
rxns = [ReactionTree.from_dict(i) for i in trees]


#calculate the cost of each tree and return a list
costs = mutils.calculate_route_cost(rxns, stock_inchis, stock)
invalid_routes = [i for i in costs if i == 1.8*10]

with open(args.output, 'w') as b:
    b.write('Cost for: \n')
    b.write(args.input+'\n')
    b.write('Number of reactions: '+str(len(costs))+'\n')
    b.write('Mean cost: '+str(np.mean(costs))+'\n')
    b.write('Standard Deviation: '+str(np.std(costs))+'\n')
    b.write('Invalid routes: '+str(len(invalid_routes)))
    b.close()

print('Cost for: \n')
print(args.input+'\n')
print('Number of reactions: '+str(len(costs))+'\n')
print('Mean cost: '+str(np.mean(costs))+'\n')
print('Standard Deviation: '+str(np.std(costs))+'\n')
print('Invalid routes: '+str(len(invalid_routes))+'\n')
print('Complete')











