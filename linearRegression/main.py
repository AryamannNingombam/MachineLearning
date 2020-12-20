import numpy as np
import pandas as pn

fullDataset = pn.read_csv('insurance.csv')

print(type(fullDataset.values))