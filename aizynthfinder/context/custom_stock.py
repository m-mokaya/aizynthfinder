import pandas as pd

from aizynthfinder.aizynthfinder.context.stock import StockQueryMixin, StockException
from aizynthfinder.aizynthfinder.chem import Molecule



# Added to allow for lookup of prices from stock file
class InMemoryInchiKeyQueryWithPrices(StockQueryMixin):
    """
    A Stock query class that is looking up inchi keys and price data from a stock file. 

    The items in this file should have inchi key and price data

    :param filename: location of stock file
    """
    def __init__(self, filename: str) -> None:
        stock = pd.read_hdf(filename, key="table")
        self._inchi_keys = list(stock["inchi_key"].values)
        self._prices = list(stock["price"].values)
    
    def __contains__(self, mol: Molecule) -> bool:
        return mol.inchi_key in self._inchi_keys
    
    def __len__(self) -> int:
        return len(self._inchi_keys)
    
    def price(self, mol) -> float:
        try:
            return self._prices[self._inchi_keys.index(mol.inchi_key)]
        except ValueError:
            raise StockException("Cannot compute price")

stock = InMemoryInchiKeyQueryWithPrices(filename='/data/localhost/not-backed-up/mokaya/exscientia/aizynthfinder/Maranga/experiments/price/data/molport_in_stock.hdf5')