import pickle
import pandas as pd
import numpy as np
import os




data=pickle.load(open('../data/market_data.p', 'rb'))

cleanData = {fname[:-4]: pd.read_csv('../data/CleanedData/' + fname) for fname in os.listdir('../data/CleanedData')}
del cleanData['.DS_S']


for key in cleanData.keys():
    cleanData[key].index=cleanData[key].iloc[:,0]
    cleanData[key].index.name='date'
    cleanData[key].drop(0, axis=1)









