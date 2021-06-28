import os 
import sys

sys.path.append('../../')

import pandas as pd
import numpy as np
import json


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions 
from rdkit import DataStructs



# calculate bulk similarity of a series of molecules inputted as smiles
def bulk_tanimoto(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    fps = [generate_fingerprints(i) for i in mols]
    
    
    nfps = len(fps)
    print('Number of molecules: ', nfps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    
    return sims


def generate_fingerprints(smile):
    return Chem.RDKFingerprint(smile)
    