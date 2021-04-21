import os 
import sys

sys.path.append('../../')

import pandas as pd
import numpy as np

import aizynthfinder.analysis
from aizynthfinder.mcts.node import Node
from aizynthfinder.analysis import ReactionTree
from aizynthfinder.mcts.state import State
from aizynthfinder.chem import TreeMolecule
from aizynthfinder.context.collection import ContextCollection
from aizynthfinder.context.stock import StockException

from parse_multismiles import findkeys


test_dict = {
    "type": "mol",
    "hide": False,
    "smiles": "O=S(=O)(NCC(c1ccccc1)N1CCCCCC1)c1ccccc1",
    "is_chemical": True,
    "in_stock": False,
    "children": [
      {
        "type": "reaction",
        "hide": False,
        "smiles": "[C:1]([NH2:2])[CH3:3]>>O=[C:1]([NH2:2])[CH3:3]",
        "is_reaction": True,
        "metadata": {
          "template_hash": "164bcd5926595fb14a6a9bdfa6263711f1e81657ce47f1a9ea439ccb",
          "classification": "Amide to amine reduction",
          "library_occurence": 1676,
          "policy_probability": 0.014999999664723873,
          "policy_name": "uspto",
          "template_code": 4087
        },
        "children": [
          {
            "type": "mol",
            "hide": False,
            "smiles": "O=C(NS(=O)(=O)c1ccccc1)C(c1ccccc1)N1CCCCCC1",
            "is_chemical": True,
            "in_stock": False,
            "children": [
              {
                "type": "reaction",
                "hide": False,
                "smiles": "[C:1]([cH3:2])([C:3]([NH2:4])=[O:5])[N:7]([CH3:6])[CH3:8]>>Br[C:1]([cH3:2])[C:3]([NH2:4])=[O:5].[CH3:6][N:7][CH3:8]",
                "is_reaction": True,
                "metadata": {
                  "template_hash": "1c206a60c6688c0f23290fc0b129e7b035ee1d777a3da755bf80901c",
                  "classification": "Heteroaryl N-alkylation",
                  "library_occurence": 17,
                  "policy_probability": 0.014499999582767487,
                  "policy_name": "uspto",
                  "template_code": 5139
                },
                "children": [
                  {
                    "type": "mol",
                    "hide": False,
                    "smiles": "C1CCCNCC1",
                    "is_chemical": True,
                    "in_stock": True
                  },
                  {
                    "type": "mol",
                    "hide": False,
                    "smiles": "O=C(NS(=O)(=O)c1ccccc1)C(Br)c1ccccc1",
                    "is_chemical": True,
                    "in_stock": False,
                    "children": [
                      {
                        "type": "reaction",
                        "hide": False,
                        "smiles": "[C:1]([CH3:2])(=[O:3])[N:4][S:5](=[O:6])(=[O:7])[cH3:8]>>O[C:1]([CH3:2])=[O:3].[N:4][S:5](=[O:6])(=[O:7])[cH3:8]",
                        "is_reaction": True,
                        "metadata": {
                          "template_hash": "4706e6cfa0d924430570646678e66958822140d1963f55fba6317d0e",
                          "classification": "N-acylation to amide",
                          "library_occurence": 151,
                          "policy_probability": 0.4880000054836273,
                          "policy_name": "uspto",
                          "template_code": 13000
                        },
                        "children": [
                          {
                            "type": "mol",
                            "hide": False,
                            "smiles": "NS(=O)(=O)c1ccccc1",
                            "is_chemical": True,
                            "in_stock": True
                          },
                          {
                            "type": "mol",
                            "hide": False,
                            "smiles": "O=C(O)C(Br)c1ccccc1",
                            "is_chemical": True,
                            "in_stock": False,
                            "children": [
                              {
                                "type": "reaction",
                                "hide": False,
                                "smiles": "[Br:1][C:5]([C:3](=[O:2])[OH:4])[cH3:6]>>O=C1CCC(=O)N1[Br:1].[O:2]=[C:3]([OH:4])[C:5][cH3:6]",
                                "is_reaction": True,
                                "metadata": {
                                  "template_hash": "d657e39736321b514ec1b19399aef5f74d4ad26a8d8c75b998b45919",
                                  "classification": "Halogenation",
                                  "library_occurence": 12,
                                  "policy_probability": 0.7285000085830688,
                                  "policy_name": "uspto",
                                  "template_code": 39213
                                },
                                "children": [
                                  {
                                    "type": "mol",
                                    "hide": False,
                                    "smiles": "O=C(O)Cc1ccccc1",
                                    "is_chemical": True,
                                    "in_stock": True
                                  },
                                  {
                                    "type": "mol",
                                    "hide": False,
                                    "smiles": "O=C1CCC(=O)N1Br",
                                    "is_chemical": True,
                                    "in_stock": True
                                  }
                                ]
                              }
                            ]
                          }
                        ]
                      }
                    ]
                  }
                ]
              }
            ]
          }
        ]
      }
    ]
  }
