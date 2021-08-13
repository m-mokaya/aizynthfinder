"""
Module to run entire aizynth reaction/transformation assessment pipeline given a .txt file with SMILES. 
"""

import os
import sys
import argparse

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./utils')
sys.path.append('/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder')

from aizynthfinder.aizynthfinder import AiZynthFinder

"""
1. collect arguments
"""

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='folder for input files')
parser.add_argument('--output', required=True, help='folder for output files')
args = parser.parse_args()


"""
2. Run AiZynthFinder in explore and standard configurations
"""

def extract_smiles_from_file(filename):
    """
    Extracts a list of smiles from file with new smile on every line
    :param filename: location of .txt file with smiles
    :return: list of smiles strings
    """
    with open(filename, 'r') as f:
        smiles = [i for i in f.readlines()]
    print('SMILES file open targets extracted.')
    return smiles

def run_aiz(input_loc, config_name, output_loc, smiles):
    smiles_file = os.path.join(input_loc, "target_smiles.txt")
    config_file = os.path.join(input_loc, config_name)

    smiles = extract_smiles_from_file(smiles_file)

    expander = AiZynth(configfile=config_file)
    finder.stock.select('molport')
    finder.expansion_policy.select('uspto')
    finder.filter_policy.select('uspto')

    results = []
    for smi in smiles:
        finder.target_smiles = smi
        finder.tree_search()
        finder.build_routes()
        stats = finder.extract_statistics()
    results.append(stats)
    return results, AiZynthFinder.routes()


print(run_aiz(args.input, 'config_std.yml', args.output, 'target_smiles.txt')[0])
print('ROUTES: ')
print(print(run_aiz(args.input, 'config_std.yml', args.output, 'target_smiles.txt')[1]))
