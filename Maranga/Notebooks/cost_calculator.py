import sys
import json
import argparse
import pandas as pd
import numpy as np

sys.path.append('../../')

import aizynthfinder.analysis
import aizynthfinder.chem
import aizynthfinder.context.config as con

from aizynthfinder.mcts.node import Node
from aizynthfinder.analysis import ReactionTree
from aizynthfinder.mcts.state import State
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.context.stock import StockException
import aizynthfinder.context.scoring as scoring

#setup up command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=False, help='filename/location of results file')
parser.add_argument('--config', required=True, help='file location of config file')
parser.add_argument('--input', required=True, help='input .hdf5 file location')
args = parser.parse_args()

f = open(args.output, 'a')
f.write('Results file for: '+args.input)

#import data from .hdf5 file
data = pd.read_hdf(args.input, 'table')

solved_data = data.loc[(data.is_solved==True)]
unsolved_data = data.loc[(data.is_solved==False)]
all_solved = data.is_solved.values

true = []
false = []

for i in range(len(all_solved)):
    if all_solved[i] == True:
        true.append(i)
    else:
        false.append(i)

print('True:', (len(true)/len(all_solved))*100)
print('False: ', (len(false)/len(all_solved))*100)

f.write('True: '+str((len(true)/len(all_solved))*100))
f.write('False: '+str((len(false)/len(all_solved))*100))

all_trees = data.trees.values
all_solved_trees = solved_data.trees.values
all_unsolved_trees = unsolved_data.trees.values

json_results = []
solved_json_results = []
print('length of solved trees: ', len(all_solved_trees))
print('length of unsolved trees: ', len(all_unsolved_trees))
unsolved_json_results = []

for i in all_trees:
    for itree, tree in enumerate(i):
        reaction_json = ReactionTree.from_dict(tree).to_json()
        json_results.append(reaction_json)

for i in all_unsolved_trees:
    for itree, tree in enumerate(i):
        unsolved_reaction_json = ReactionTree.from_dict(tree).to_json()
        unsolved_json_results.append(unsolved_reaction_json)

for i in all_solved_trees:
    for itree, tree in enumerate(i):
        solved_reaction_json = ReactionTree.from_dict(tree).to_json()
        solved_json_results.append(solved_reaction_json)


solved_json_results = [json.loads(i) for i in solved_json_results]

#solved reaction parse rxn tree
solved_reaction_data = []

for i in solved_json_results:
    if 'children' in i:
        child = i.get('children')[0]
        if child.get('type') == 'reaction':
            metadata = child.get('metadata')
            solved_reaction_data.append((child.get('smiles'), metadata.get('classification'), metadata.get('library_occurence'), metadata.get('policy_probability')))
        else:
            pass
    else:
        pass

smiles = [i[0] for i in solved_reaction_data]
classification = [i[1] for i in solved_reaction_data]
uspto_freq = [i[2] for i in solved_reaction_data]
policy = [i[3] for i in solved_reaction_data]


reaction_class = {}

for i in classification:
    if (i in reaction_class):
        reaction_class[i]['Frequency'] += 1
    else:
        reaction_class[i] = {}
        reaction_class[i]['Frequency'] = 1

reaction_class = dict(sorted(reaction_class.items(), key=lambda item: item[1]['Frequency'], reverse=True))

#calculate average policy predictions for each grouped reaction

reaction_class_policy = {}
reaction_class_freq = {}

for i in solved_reaction_data: 
    if i[1] in reaction_class_policy:
        reaction_class_policy[i[1]] += i[3]
        reaction_class_freq[i[1]] += 1
    else:
        reaction_class_policy[i[1]] = i[3]
        reaction_class_freq[i[1]] = 1

reaction_class_policy_mean = {}

example = 0
for i in reaction_class_policy:
    reaction_class_policy_mean[i] = (reaction_class_policy.get(i)/reaction_class_freq.get(i))

    if example < 3:
        print(str(i)+', '+str(reaction_class_policy.get(i)))
        print(str(i)+', '+str(reaction_class_freq.get(i)))
        print(reaction_class_policy_mean[i])
        print('\n')
        
    example += 1

#reaction_class_policy_mean = dict(sorted(reaction_class_policy_mean.items(), key=lambda item: item[1], reverse=True))

print('Rank, Reaction, Frequency, Policy value')

'''for x in reaction_class_policy_mean:
    print (x + '    '+ str(reaction_class_policy_mean[x]))'''

for i in reaction_class:
    reaction_class[i]['Policy'] = reaction_class_policy_mean.get(i)

count = 0
for i in reaction_class:
    print(str(count)+','+i+', '+str(reaction_class[i]['Frequency'])+', '+str(reaction_class[i]['Policy']))
    count+=1

reaction_class_lit = {}
reaction_class_freq = {}

for i in solved_reaction_data: 
    if i[1] in reaction_class_lit:
        reaction_class_lit[i[1]] += i[2]
        reaction_class_freq[i[1]] += 1
    else:
        reaction_class_lit[i[1]] = i[2]
        reaction_class_freq[i[1]] = 1

reaction_class_lit_mean = {}

example = 0
for i in reaction_class_lit:
    reaction_class_lit_mean[i] = (reaction_class_lit.get(i)/reaction_class_freq.get(i))

    if example < 3:
        print(str(i)+', '+str(reaction_class_lit.get(i)))
        print(str(i)+', '+str(reaction_class_freq.get(i)))
        print(reaction_class_lit_mean[i])
        print('\n')
        
    example += 1

reaction_class_lit_mean = dict(sorted(reaction_class_lit_mean.items(), key=lambda item: item[1], reverse=True))

print('Rank, Reaction, Frequency, Policy, Literature')
f.write('\n\nRank, Reaction, Frequency, Policy, Literature\n')

'''for x in reaction_class_lit_mean:
    print (x + '    '+ str(reaction_class_lit_mean[x]))'''


for i in reaction_class:
    reaction_class[i]['Literature'] = reaction_class_lit_mean.get(i)

count = 1
for i in reaction_class:
    print(str(count)+','+i+', '+str(reaction_class[i]['Frequency'])+', '+str(round(reaction_class[i]['Policy'], 4))+', '+str(round(reaction_class[i]['Literature'], 0)))
    f.write(str(count)+','+i+', '+str(reaction_class[i]['Frequency'])+', '+str(round(reaction_class[i]['Policy'], 4))+', '+str(round(reaction_class[i]['Literature'], 0))+'\n')
    count+=1


# route cost calculator
config = con.Configuration()
config = config.from_file(args.config)

rxns = [ReactionTree.from_dict(tree) for i in all_solved_trees for x, tree in enumerate(i)]


policy_scorer = scoring.USPTOModelPolicyProbabilityScorer()
policy_scores = policy_scorer(rxns)
policy_mean = np.mean(policy_scores)

cost_scorer = scoring.RouteCostScorer(config=config)
cost_scores = cost_scorer(rxns)
cost_mean = np.mean(cost_scores)

num_reactions_scorer = scoring.NumberOfReactionsScorer()
num_reactions_scores = num_reactions_scorer(rxns)
num_reactions_mean = np.mean(num_reactions_scores)

overall_cost = (0.2*100 - (1/10)*100)/num_reactions_mean

print('policy: ', policy_mean)
f.write('\n\npolicy: '+ str(policy_mean)+'\n')

print('Cost: ', cost_mean)
f.write('Cost: '+ str(cost_mean)+'\n')

print('Cost Term: ', cost_mean/100)

print('Num reactions: ', num_reactions_mean)
f.write('Num Reactions: '+ str(num_reactions_mean)+'\n')

print('Target Library Synthesis Cost: ', cost_mean)
f.write('Target Lib cost: '+ str(cost_mean)+'\n')
f.close()