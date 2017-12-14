import pickle
import pandas as pd
import numpy as np
import os




data=pickle.load(open('../data/market_data.p', 'rb'))

cleanData = {fname[:-4]: pd.read_csv('../data/CleanedData/' + fname) for fname in os.listdir('../data/CleanedData')}
del cleanData['.DS_S']


for key in cleanData.keys():
    df=cleanData[key]
    df.index=df.iloc[:,0]
    df.index.name='date'
    df.drop(df.columns[0], axis=1, inplace=True)
    

















