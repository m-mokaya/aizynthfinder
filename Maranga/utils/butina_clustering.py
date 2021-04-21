import os
import sys

sys.path.append('../../')

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions 

class SmartsCluster:

    def __init__(self, smarts: list):
        self.smarts_list = smarts
        self.reaction_finderprints = None

    def clusterFps(self, cutoff):
        from rdkit import DataStructs
        from rdkit.ML.Cluster import Butina

        #generate distance matrix
        dists = []
        nfps = len(self.reaction_finderprints)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(self.reaction_finderprints[i], self.reaction_finderprints[:i])
            dists.extend([1-x for x in sims])
        
        #cluster the data
        cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
        return cs

    #generate list of reaction fingerprints from list of smarts
    def generateFpsList(self):
        rxns = [rdChemReactions.ReactionFromSmarts(i) for i in self.smarts_list]
        rxn_fps = [rdChemReactions.CreateStructuralFingerprintForReaction(i) for i in rxns]
        self.reaction_finderprints = rxn_fps
        return self.reaction_finderprints