"""
Module to run entire aizynth reaction/transformation assessment pipeline given a .txt file with SMILES. 
"""

from multiprocessing import process
import os
import sys
import argparse
import yaml

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import multiprocessing as mp

sys.path.append('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder')
sys.path.append('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/utils')

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.aizynthfinder import AiZynthExpander

import utils.parse_multismiles as mutils
import utils.fingerprints as fings 

from aizynthfinder.reactiontree import ReactionTree
from collections import defaultdict, Counter


"""
1. collect arguments
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='folder for input files')
parser.add_argument('--output', required=True, help='folder for output files')
parser.add_argument('--threshold', required=False, help='cost thresgold for "cheap" reactions')
parser.add_argument('--run', required=True, help='job iteration')
parser.add_argument('--nproc', required=False, help='number of cores to run job')
args = parser.parse_args()


"""
2. Run AiZynthFinder in explore and standard configurations
"""

def run_aiz(input_loc, config_name, output_loc, filename, nproc):

    n = int(nproc)

    smiles_file = os.path.join(input_loc, "target_smiles.txt")
    
    config_file = os.path.join(input_loc, config_name)


    smiles = extract_smiles_from_file(smiles_file)
    split_smiles = split_smiles_list(smiles, n)


    if len(smiles) > 1:
        pool = mp.Pool(n)
        smiles_input = [(config_file, i) for i in split_smiles]
        results = pool.starmap(worker, smiles_input)
        # results = pool.map(worker(config_file, split_smiles), split_smiles)
        results = [pd.DataFrame.from_dict(i) for i in results]
        data = pd.DataFrame.from_dict(pd.concat(results))
    else:
        finder = AiZynthFinder(configfile=config_file)
        finder.stock.select('molport')
        finder.expansion_policy.select('uspto')
        finder.filter_policy.select('uspto')

        results = defaultdict(list)
        for smi in smiles:
            finder.target_smiles = smi
            finder.prepare_tree()
            search_time = finder.tree_search()
            finder.build_routes()
            stats = finder.extract_statistics()

        solved_str = 'is_solved' if stats['is_solved'] else 'is not solved'
        print(f'Done with {smi} in {search_time:.3} s and {solved_str}')

        for key, value in stats.items():
            results[key].append(value)
        results['top scores'].append(", ".join("%.4f" % score for score in finder.routes.scores))
        results['trees'].append(finder.routes.dicts)
        data = pd.DataFrame.from_dict(results)
    
    # save data to .hdf5 file
    file_output = os.path.join(output_loc, filename)
    data.to_hdf(file_output, key="table", mode="w")

    print(f'Output saved to {file_output}')
    return data

def worker(config_file, smiles):
    finder = AiZynthFinder(configfile=config_file)
    finder.stock.select('molport')
    finder.expansion_policy.select('uspto')
    finder.filter_policy.select('uspto')

    results = defaultdict(list)
    for smi in smiles:
        print(f'Starting with {smi}')
        finder.target_smiles = smi
        finder.prepare_tree()
        search_time = finder.tree_search()
        finder.build_routes()
        stats = finder.extract_statistics()

        solved_str = 'is_solved' if stats['is_solved'] else 'is not solved'
        print(f'Done with {smi} in {search_time:.3} s and {solved_str}')

        for key, value in stats.items():
            results[key].append(value)
        results['top scores'].append(", ".join("%.4f" % score for score in finder.routes.scores))
        results['trees'].append(finder.routes.dicts)
    return results

"""
3. Find novel & cheap pathways => transformations to optimise
"""

def novel_pathways(std_results, exp_results, output_loc, threshold):
    """
    Find all novel routes generated in explore mode relative to all routes in std mode and explore mode.

    :param std_results: dataframe metadata from std routes
    :param exp_results: dataframe metadata from exp routes
    :return: list of novel routes as list[reaction dicts]
    """
    ### INPUT DATA
    # collect succesful routes to a new df
    std_solved_routes = mutils.collect_trees(std_results.loc[(std_results.is_solved==True)])
    exp_solved_routes = mutils.collect_trees(exp_results.loc[(exp_results.is_solved==True)])
    
    # separate all reactions
    reactions_std = fings.parse_input(std_solved_routes, 'normal')
    reactions_exp = fings.parse_input(exp_solved_routes, 'explore')
    
    all_reactions = reactions_std + reactions_exp

    r_e = [i.get('reaction') for i in reactions_exp]
    r_s = [i.get('reaction') for i in reactions_std]
    rs = [ReactionTree.from_dict(i) for i in r_s]
    re = [ReactionTree.from_dict(i) for i in r_e]

    ### GENERATE FINGERPRINTS
    fingerprints = fings.generate_fingerprints(all_reactions)

    # calculate reaction similarity from fingerprints
    largest, lengths, all_distances, explore_reactions = fings.split_sim(fingerprints)

    num_novel = sum(1 for i in largest if i == 1.0)

    print('Proportion Novel Routes: ', (num_novel/len(largest))*100)

    combos = list(zip(largest, lengths))
    weight_counter = Counter(combos)

    weights = [weight_counter[(largest[i], lengths[i])] for i, _ in enumerate(largest)]

    print('Plotting novelty..')
    fig1 = plt.figure()
    plt.scatter(largest, lengths, s=weights)
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Route Length')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'images/reaction_novelty.png'))
    plt.show()

    fig1 = plt.figure()
    plt.hist(all_distances)
    plt.xlabel('Route distance')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'images/route_distances.png'))
    plt.show()

    print('\n')


    ### CALCULATE COSTS OF EXPLORE ROUTES
    # import stock file to calculate novel route costs
    stock_file = '/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/aizynthfinder/data/molport_blocks_stock.hdf5'
    stock = pd.read_hdf(stock_file, 'table')
    stock_inchis = stock['inchi_key'].tolist()

    # get all the reaction dicts for all explore reactions
    explore_rxn_trees = [ReactionTree.from_dict(i.get('reaction')) for i in explore_reactions]

    # calculate the cost of all routes
    costs = mutils.calculate_route_cost(explore_rxn_trees, stock_inchis, stock)
    nan_mean = np.nanmean(costs)
    costs = [nan_mean if np.isnan(i) else i for i in costs]
    
    # add cost value to explore reactions in dict
    for index, i in enumerate(explore_reactions):
        i['cost'] = costs[index]

    # number of novel reactions + cost
    novel_explore = [i for i in explore_reactions if i.get('largest') != 1.0] 
    novel_costs = [i.get('cost') for i in novel_explore]

    # get all reactions under cost of 70
    under_threshold_explore = [i for i in explore_reactions if (i.get('cost')) < float(threshold) and i.get('largest') != 1.0]
    transformations = find_novel_reaction_templates(under_threshold_explore)
    return transformations


def find_novel_reaction_templates(rxns):
    """
    Function to find all the transformations used from a set of reaction routes
    :param rxns: list of reaction routes
    :return: list of transformations used
    """
    transformations = []
    for i in rxns:
        templates = list(mutils.findkeys(i, 'classification'))
        transformations.extend(templates)
    
    s_transformations = list(set(transformations))

    return s_transformations


def count_reactions(rxns):
    """
    Function to count all the transformations used for a list of pathways

    :param rxns: list of reactions
    :return: dict of transfortions and frequencies
    """

    templates = []
    templates_dict = {}
    for i in rxns:
        templates.extend(list(mutils.findkeys(i, 'classification')))

    print('# Templates: ', len(templates))

    for i in templates:
        if i in templates_dict:
            templates_dict[i] += 1
        else:
            templates_dict[i] = 1
    return templates_dict


def find_best_transformations(std_result, opt_result, trans):
    """
    Function to determine the change in % transformation usage in standard and optimised routes. 

    :param std_result: dataframe of std result routes
    :param opt_result: dataframe of opt result routes
    :param trans: all the transformations optimised
    :return: dict with all opt and used transformations with % change (opt - std)
    """
    std_solved = std_result.loc[(std_result.is_solved==True)]
    std_reactions = std_solved.trees.values
    opt_solved = opt_result.loc[(opt_result.is_solved==True)]
    opt_reactions = opt_solved.trees.values

    std_freq = count_reactions(std_reactions)
    opt_freq = count_reactions(opt_reactions)

    std_sum = sum(std_freq.values())
    opt_sum = sum(opt_freq.values())

    std_per = {}
    opt_per = {}

    difference = {}
    
    for i in trans:
        o = opt_freq.get(i)
        s = std_freq.get(i)

        if o != None and s != None:
            difference[i] = (o/opt_sum)*100 - (s/std_sum)*100
    return difference



"""
General helper functions
"""

def extract_smiles_from_file(filename):
    """
    Extracts a list of smiles from file with new smile on every line
    :param filename: location of .txt file with smiles
    :return: list of smiles strings
    """
    with open(filename, 'r') as f:
        smiles = [i for i in f.read().splitlines()]
    print('SMILES file open targets extracted.')
    return smiles

def extract_templates():
    """
    Function to extract all the templates from input reaction data file
    :return: list of all tempaltes available
    """
    f = '/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/aizynthfinder/data/uspto_templates.hdf5'
    data = pd.read_hdf(f, 'table')

    classes = data['classification'].tolist()
    return list(set(classes))

def save_opt_dict(dict, output, filename):
    """
    Save list of transformations in optimisation dict.

    :param dict: dictionary to save
    :param output: locations of result directory
    :param filename: filename to save file in output location
    """
    with open(os.path.join(output, filename), 'w') as f:
        json.dump(dict, f)

def split_smiles_list(data, chunk_size):
    return (data[i::chunk_size] for i in range(chunk_size))

def reaction_costs(hdf):
    stock_file = '/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/aizynthfinder/data/molport_blocks_stock.hdf5'
    stock = pd.read_hdf(stock_file, 'table')
    stock_inchis = stock['inchi_key'].tolist()

    solved_data = hdf.loc[(hdf.is_solved==True)]
    print('Number of solved molecues: ', len(solved_data))
    reactions = solved_data.trees.values
    r = []
    for i in reactions:
        r.extend([ReactionTree.from_dict(p) for p in i])
    costs = mutils.calculate_route_cost(r, stock_inchis, stock)

    nan_mean = np.nanmean(costs)
    costs = [nan_mean if np.isnan(i) else i for i in costs]

    print('Mean cost: ', np.mean(costs))
    print('SD: ', np.std(costs))
    return costs

"""
Run Top vs Random
"""
def run_top_reactions(args, trans, random, threshold):

    o_mean = []
    o_sd = []
    for index in range(1, len(trans)+1):
        opt_l = trans[:index]
    
        opt_dict = {}
        for i in opt_l:
            opt_dict[i] = 10.0

        loc = os.path.join(args.output, f'opts/opt{args.run}_class_{int(threshold)}_o{index}.json')

        with open(loc, 'w') as f:
            json.dump(opt_dict, f)
        
        #create new config file
        with open(os.path.join(args.input, 'config_std.yml'), 'r') as f:
            data = yaml.safe_load(f)
        
        # change policy values 
        data["properties"]["policy_values"] = loc

        with open(os.path.join(args.input, f'results/{args.run}/configs/config_opt{args.run}_{int(threshold)}_o{index}.yml'), 'w') as f:
            yaml.safe_dump(data, f)
        
        print('Starting "opt" run..')
        opt_df = run_aiz(args.input, f'results/{args.run}/configs/config_opt{args.run}_{int(threshold)}_o{index}.yml', args.output, f'hdfs/opt_results_{args.run}_{int(threshold)}_o{index}.hdf5', args.nproc)
        print('Done.')

        o_costs = reaction_costs(opt_df)
        o_mean.append(np.mean(o_costs))
        o_sd.append(np.std(o_costs))
        print(f'Run {index} of threshold {int(threshold)}. Mean: {np.mean(o_costs)} and SD: {np.std(o_costs)}.')

    r_av_mean = []
    r_av_sd = []

    for y, item in emumerate(random):
        print(f'Random iteration {y}.')
        r_mean = []
        r_sd = []
        for index2 in range(1, len(item)+1):
            opt_l = item[:index2]
        
            opt_dict = {}
            for i in opt_l:
                opt_dict[i] = 10.0

            loc = os.path.join(args.output, f'opts/opt{args.run}_class_{int(threshold)}_r{index2}.json')

            with open(loc, 'w') as f:
                json.dump(opt_dict, f)
            
            #create new config file
            with open(os.path.join(args.input, 'config_std.yml'), 'r') as f:
                data = yaml.safe_load(f)
            
            # change policy values 
            data["properties"]["policy_values"] = str(loc)

            with open(os.path.join(args.input, f'results/{args.run}/configs/config_opt{args.run}_{int(threshold)}_r{index2}.yml'), 'w') as f:
                yaml.safe_dump(data, f)
            
            print('Starting "opt" run..')
            opt_df = run_aiz(args.input, f'results/{args.run}/configs/config_opt{args.run}_{int(threshold)}_r{index2}.yml', args.output, f'hdfs/opt_results_{args.run}_{int(threshold)}_r{index2}.hdf5', args.nproc)
            print('Done.')

            r_costs = reaction_costs(opt_df)
            r_mean.append(np.mean(r_costs))
            r_sd.append(np.std(r_costs))
            print(f'Run {index2}  of threshold {int(threshold)}. Mean: {np.mean(r_costs)} and SD: {np.std(r_costs)}.')
        
        r_av_mean.append(r_mean)
        r_av_sd.append(r_sd)

        r_out_mean = np.mean(np.array(r_av_mean), axis=0)
        r_out_sd = np.mean(np.array(r_av_sd), axis=0)

    print('done calculting means')
    return o_mean, o_sd, r_out_mean, r_out_sd

"""
Main Function
"""

def main(args):

    ofile = open(os.path.join(args.output, f'output_{args.run}.txt'), 'w')
    ofile.write('STD & EXP runs.')


    """
    1. Run AiZ in explore and normal modes
    """
    '''
    print('Starting "std" run..')
    # Run AiZ in Std mode
    std_df = run_aiz(args.input, 'config_std.yml', args.output, f'hdfs/std_results_{args.run}.hdf5', args.nproc)
    print('done.')

    print('Starting "exp" run..')
    # run AiZ in exp mode
    exp_df = run_aiz(args.input, 'config_exp.yml', args.output, f'hdfs/exp_results_{args.run}.hdf5', args.nproc)
    print('done.')
    '''
    std_df = pd.read_hdf(os.path.join(args.output, f'hdfs/std_results_{args.run}.hdf5'), 'table')
    print('std route costs: ')
    std_costs = reaction_costs(std_df)

    print('\n')
    exp_df = pd.read_hdf(os.path.join(args.output, f'hdfs/exp_results_{args.run}.hdf5'), 'table')
    print('exp route costs: ')
    exp_costs = reaction_costs(exp_df)


    ofile.write('STD Route:\n')
    ofile.write(f'Mean: {np.mean(std_costs)}\n')
    ofile.write(f'SD: {np.std(std_costs)}\n')

    ofile.write('\n')
    ofile.write('EXP Route:\n')
    ofile.write(f'Mean: {np.mean(exp_costs)}\n')
    ofile.write(f'SD: {np.std(exp_costs)}\n')


    """
    2. Determine which transformations given cost threshold.
    """
    ofile.write('\n')
    ofile.write('Transformations & Thresholds\n')

    threshold_means = []
    threshold_sd = []

    for ind, i in enumerate(np.arange(5, np.mean(std_costs), 5)):
        print('\n')
        print(f'Threshold range: (5-{np.mean(std_costs)})')

        ofile.write(f'Threshold range: (5-{np.mean(std_costs)})\n')
        
        threshold = i

        print('\n')
        print('Cost threshold: ', threshold)
        ofile.write(f'Cost threshold: {int(threshold)}\n')

        print('\n')
        print('Calculating transformations to optimise.')

        # transformations to optimise
        transformations = novel_pathways(std_df, exp_df, args.output, threshold)
        print('Transformations: ', transformations)
        print('Number of transformations: ', len(transformations))
        
        ofile.write(f'Transformations to optimise: {transformations}\n')
        ofile.write(f'Number of transformations: {len(transformations)}\n')

        if len(transformations) == 0:
            print('0 transformations optimised, increasing threshold.')
            ofile.write('0 transformations optimised, increasing threshold.\n')
            continue

        #make new optimisation dict
        opt_dict = {}
        for i in transformations:
            opt_dict[i] = 10.0
        
        save_opt_dict(opt_dict, args.output, f'opt{args.run}_class_{int(threshold)}.json')
        print('\n')
        print('Optimisation dict saved to: ', os.path.join(args.output, f'opts/opt'+args.run+'_class_{int(threshold)}.json'))

        with open(os.path.join(args.input, 'config_std.yml'), 'r') as f:
            data = yaml.safe_load(f)
        
        # change policy values 
        data["properties"]["policy_values"] = str(os.path.join(args.output,f'opts/opt{args.run}_class_{int(threshold)}.json'))

        with open(os.path.join(args.input, f'results/{args.run}/configs/config_opt{args.run}_{int(threshold)}.yml'), 'w') as f:
            yaml.safe_dump(data, f)
        
        print('\n')
        print(f'Starting "opt" ({args.run}_{int(threshold)}) run..')
        opt_df = run_aiz(args.input, f'results/{args.run}/configs/config_opt{args.run}_{int(threshold)}.yml', args.output, f'hdfs/opt{args.run}_results_{int(threshold)}.hdf5', args.nproc)
        print('Done.')

        print('\n')
        print('Calculating opt costs...')
        c = reaction_costs(opt_df)


        ofile.write('\n')
        ofile.write(f'OPT (threshold: {int(threshold)}) costs\n')
        ofile.write(f'Mean: {np.mean(opt_costs)}\n')
        ofile.write(f'SD: {np.std(opt_costs)}\n')

        threshold_means.append(np.mean(c))
        threshold_sd.append(np.std(c))

        print('\n')
        print('Sorting trns...')
        # sort the transformations by use in std vs. opt
        difference = find_best_transformations(std_df, opt_df, opt_dict.keys())
        s_difference = dict(sorted(difference.items(), key=lambda kv: kv[1], reverse=True))

        print('Plotting trns..')
        fig1 = plt.figure()

        x_vals = s_difference.keys()
        y_vals = [s_difference.get(i) for i in x_vals]
        plt.plot(x_vals, y_vals)
        plt.xticks(rotation=20)
        plt.xlabel('Template')
        plt.ylabel('Difference in template useage (opt - std) / %')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f'images/trans_%_difference_{int(threshold)}.png'))
        plt.show()

        print('\n')
        print('Ordered transformations:')
        for k, v in s_difference.items():
            print(f'{k}: {v}')

        # how many transformations to test
        c = sum(1 if i > 0 else 0 for i in s_difference.values())
        counts = len(s_difference)
        print('Number of transformations in random test: ', len(s_difference))

        all_classes = extract_templates()
        print('Total # transformations: ', len(all_classes))
        
        all_random_opt = []
        for i in range(3):
            rnd_opt = rnd.sample(all_classes, counts)
            all_random_opt.append(rnd_opt)
        top_opt = list(s_difference.keys())[:counts]
        
        print('\n')
        print('Transformations to optimse: ', top_opt)
        print('Random Trans to opt: ', rnd_opt)

        o_m, o_sd, r_m, r_sd = run_top_reactions(args, top_opt, all_rnd_opt, threshold)
        
        #plot figure
        fig4 = plt.figure()
        x = np.arange(1, counts+1, 1)
        plt.plot(x, o_m, label='OPT mean')
        plt.plot(x, o_sd, label='OPT SD')
        plt.plot(x, r_m, label='RND mean')
        plt.plot(x, r_sd, label='RND SD')
        plt.xlabel('# transformations optimised')
        plt.ylabel('Route Cost')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(args.output, f'images/random_vs_opt_{int(threshold)}.png'))
        plt.show()

    fig5 = plt.figure()
    x = np.arange(5, np.mean(std_costs), 5)
    plt.plot(x, threshold_means, label='Mean')
    plt.plot(x, threshold_sd, label='SD')
    plt.xlabel('Threshold')
    plt.ylabel('Route Cost')
    plt.legend()
    plt.savefig(os.path.join(args.output, 'images/threshold_results.png'))
    plt.show()

    ofile.close()

    print('JOB COMPLETE')

if __name__ == "__main__":
    main(args)




    





    



