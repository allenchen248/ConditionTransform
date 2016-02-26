import ast
import random
import string
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from collections import defaultdict

from plot import *
from nametransform import *

# TODO
### 
# (1) Functional - takes in default values later
# (2) In value below, b must be between a and 10

stochastic_funcs = ['chimpRand', 'chimpNorm', 'Discrete']

sto_tree = """def test_func():
	a = .5
	b = a+12+chimpRand(10)
	c = b+a
	return c"""

chimp_tree = """class TestNormal(Dist):
    def __init__(self):
        self.discrete = Discrete([1,1],size=1)
    def run(self,trace):
        t = trace.chimplify("choice", self.discrete)
        return trace.chimplify("x",Normal(0,t*10,1))"""
