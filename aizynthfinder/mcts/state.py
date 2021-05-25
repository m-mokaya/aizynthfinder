""" Module contain a class that encapsulate the state of search tree node.
"""
from __future__ import annotations
from collections import defaultdict
from aizynthfinder.context.stock import StockException
import os
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import Draw

from aizynthfinder.chem import TreeMolecule

if TYPE_CHECKING:
    from aizynthfinder.utils.type_utils import StrDict, Any, List, Optional, Sequence
    from aizynthfinder.utils.serialization import (
        MoleculeDeserializer,
        MoleculeSerializer,
    )
    from aizynthfinder.context.config import Configuration

os.environ["PYTHONHASHSEED"] = "42"


class State:
    """
    Encapsulation of the molecular state of a node.

    A state consists of an immutable list of molecules that are either solved
    (can be found in stock) or that potentially can be expanded to new molecules
    by applying a reaction on them.

    The class is hashable and comparable by the inchi keys of all the molecules.

    :ivar mols: the list of molecules
    :ivar stock: the configured stock
    :ivar in_stock_list: for each molecule if they are in stock
    :ivar is_solved: is true if all molecules are in stock:
    :ivar max_transforms: the maximum of the transforms of the molecule
    :ivar is_terminal: is true if the all molecules are in stock or if the maximum transforms has been reached

    :param mols: the molecules of the state
    :param config: settings of the tree search algorithm
    """

    def __init__(self, mols: Sequence[TreeMolecule], config: Configuration) -> None:
        self.mols = mols
        self.stock = config.stock
        self._config = config
        self.in_stock_list = [mol in self.stock for mol in self.mols]
        self._stock_availability: Optional[List[str]] = None
        self.is_solved = all(self.in_stock_list)
        self.max_transforms = max(mol.transform for mol in self.mols)
        self.is_terminal = (
            self.max_transforms > config.max_transforms
        ) or self.is_solved
        self._score: Optional[float] = None

        inchis = [mol.inchi_key for mol in self.mols]
        inchis.sort()
        self._hash = hash(tuple(inchis))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return False
        return self.__hash__() == other.__hash__()

    def __str__(self) -> str:
        """A string representation of the state (for print(state))"""
        string = "%s\n%s\n%s\n%s\nScore: %0.3F Solved: %s" % (
            str([mol.smiles for mol in self.mols]),
            str([mol.transform for mol in self.mols]),
            str([mol.parent.smiles if mol.parent else "-" for mol in self.mols]),
            str(self.in_stock_list),
            self.score,
            self.is_solved,
        )
        return string

    @classmethod
    def from_dict(
        cls, dict_: StrDict, config: Configuration, molecules: MoleculeDeserializer
    ) -> "State":
        """
        Create a new state from a dictionary, i.e. deserialization

        :param dict_: the serialized state
        :type dict_: dict
        :param config: settings of the tree search algorithm
        :type config: Configuration
        :param molecules: the deserialized molecules
        :type molecules: MoleculeDeserializer
        :return: a deserialized state
        :rtype: State
        """
        mols = molecules.get_tree_molecules(dict_["mols"])
        return State(mols, config)

    @property
    def score(self) -> float:
        """
        Returns the score of the state

        :return: the score
        :rtype: float
        """
        if not self._score:
            self._score = self._calc_score()
        return self._score

    @property
    def stock_availability(self) -> List[str]:
        """
        Returns a list of availabilities for all molecules

        :return: the list
        :rtype: list of str
        """
        if not self._stock_availability:
            self._stock_availability = [
                self.stock.availability_string(mol) for mol in self.mols
            ]
        return self._stock_availability

    def serialize(self, molecule_store: MoleculeSerializer) -> StrDict:
        """
        Serialize the state object to a dictionary

        :param molecule_store: the serialized molecules
        :type molecule_store: MolecularSerializer
        :return: the serialized state
        :rtype: dict
        """
        return {"mols": [molecule_store[mol] for mol in self.mols]}

    def to_image(self, ncolumns: int = 6) -> Any:
        """
        Constructs an image representation of the state

        :param ncolumns: number of molecules per row, defaults to 6
        :type ncolumns: int, optional
        :return: the image representation
        :rtype: a PIL image
        """
        for mol in self.mols:
            mol.sanitize()
        legends = self.stock_availability
        mols = [mol.rd_mol for mol in self.mols]
        return Draw.MolsToGridImage(mols, molsPerRow=ncolumns, legends=legends)

    def _calc_score(self) -> float:
        # How many is in stock (number between 0 and 1)
        num_in_stock = np.sum(self.in_stock_list)
        # This fraction in stock, makes the algorithm artificially add stock compounds by cyclic addition/removal
        fraction_in_stock = num_in_stock / len(self.mols)

        # Max_transforms should be low
        max_transforms = self.max_transforms
        # Squash function, 1 to 0, 0.5 around 4.
        max_transforms_score = self._squash_function(max_transforms, -1, 0, 4)

        # NB weights should sum to 1, to ensure that all
        score4 = 0.95 * fraction_in_stock + 0.05 * max_transforms_score


        # get the price of each molecule if in stock
        default_cost = 1.0
        not_in_stock_multiplier = 10

        print('Stock: ', self._config.stock)

        costs = {}
        #iterate through molecules
        for mol in self.mols:
            print(mol)
            print('Type: ', type(mol))
            print(getattr(mol, 'price'))
            #check if mol in stock (if not, that doct position is skipped).
            if mol not in self.in_stock_list:
                print('not in stock')
                continue
            try:
                values = []
                values.append(getattr(mol, 'price'))
                cost = min(values)
                print('Cost: ', cost)
            except StockException:
                print('no cost => 1.0')
                costs[mol] = 1.0
            else:
                costs[mol] = cost
        
        #determine the most expensive molecule
        max_cost = max(costs.values()) if costs else default_cost
        #fills missing molecules (those not in stock with not in cost score).
        costs_score = defaultdict(lambda: max_cost * not_in_stock_multiplier, costs)
        largest_cost = max_cost * not_in_stock_multiplier
        normalised_costs = [i/largest_cost for i in list(costs_score.values())]

        #output_score = (0.95*fraction_in_stock)+(0.03*(1-np.mean(normalised_costs)))+(0.05*max_transforms_score)
        output_score = (0.95*fraction_in_stock)+(0.04*max_transforms_score)+(0.01*(1-np.mean(normalised_costs)))
        #print('FIStock: ', 0.95*(fraction_in_stock))
        #print('transforms: ', 0.04*(max_transforms_score))
        #print('costs: ', 0.01*(1-np.mean(normalised_costs)))
        return output_score

        #return score4

    @staticmethod
    def _squash_function(
        r: float, slope: float, yoffset: float, xoffset: float
    ) -> float:
        """Squash function loosely adapted from a sigmoid function with parameters
        to modify and offset the shape

        :param r: the sign of the function, if the function goes from 1 to 0 or from 0 to 1
        :param slope: the slope of the midpoint
        :param xoffset: the offset of the midpoint along the x-axis
        :param yoffset: the offset of the curve along the y-axis"""
        return 1 / (1 + np.exp(slope * -(r - xoffset))) - yoffset
