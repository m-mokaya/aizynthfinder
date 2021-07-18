import os
import sys
import json

'''
# TODOs
[] create a class that takes self and dict file locations
[] Has a method (__call__ method) that returns a dictionary.
'''

class PolicyValues:
    
    #initialises PolicyValues class
    def __init__(self, filename) -> None:
        self.filename = filename
    
    def from_dict(self):
        with open(self.filename) as infile:
            data = json.load(infile)
        return data
