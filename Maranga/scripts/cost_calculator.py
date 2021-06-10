# File to calculate the cost of a series of route 

import sys
import json
import argparse
import pandas as pd
import numpy as np
import os
from statistics import mean, stdev
import math

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
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='results file location')
parser.add_argument('--output', required=True, help='output file location')
parser.add_argument('--type', required=True, help='type of input file')
parser.add_argument('--name', required=True, help='name of job')
args = parser.parse_args()


stock_file = '/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/aizynthfinder/data/molport_blocks_stock.hdf5'
stock = pd.read_hdf(stock_file, 'table')
stock_inchis = stock['inchi_key'].tolist()

print('\n')
print('Number of stock items: ', len(stock_inchis))

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

print('Number of trees: ', len(trees))
print('\n')
print('Loaded trees, calculating cost ...')
    
rxns = [ReactionTree.from_dict(i) for i in trees]


#calculate the cost of each tree and return a list
costs = mutils.calculate_route_cost(rxns, stock_inchis, stock)
invalid_routes = [i for i in costs if isinstance(i, float) == False]

print('\n')
nans = [i for i in costs if math.isnan(i)]
print('NaNs: ', len(nans))
nan_mean = np.nanmean(costs)

costs = [nan_mean if math.isnan(i) else i for i in costs]

text_filename = "tg2_"+args.name+"_cost.txt"
with open(os.path.join(args.output, text_filename), 'w') as b:
    b.write('Cost for: \n')
    b.write(args.input+'\n')
    b.write('Number of reactions: '+str(len(costs))+'\n')
    b.write('Mean cost: '+str(np.nanmean(costs))+'\n')
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

# write costs to file image
image_filename = "tg2_"+args.name+"_costs.png"
plt.hist(costs)
plt.xlabel('Cost')
plt.ylabel('Freq')
plt.savefig(os.path.join(args.output, image_filename))
plt.show()










