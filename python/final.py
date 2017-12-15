import pickle
import pandas as pd
import numpy as np
import os
from cvxopt import matrix, solvers



# solvers.options['show_progress'] = False


def factor_mimicking_portfolio_cvx(data, exp_factor, neu_factors, covariance, date, cutoff=0.3):
    cutoffs = np.array([cutoff, 1 - cutoff]) * 100
    
    # focus on ones we care, others are zeors
    a=data[exp_factor].loc[date, ]
    touse=a.index[~a.isnull()]
    n = len(a)
    k=len(neu_factors)
    bs=np.ones((k, n))*np.nan
    for i in range(k):
        b=data[neu_factors[i]].loc[date,]
        bs[i,]=b
        touse=touse.difference(b.index[b.isnull()])
    

    V=covariance[touse].loc[touse]
    a=a[touse]
    price_vec = data['PX.Weekly'].loc[date,touse]

    bs=pd.DataFrame(bs, columns=covariance.index)
    bs=bs[touse]


    lower, upper = np.percentile(a[touse], cutoffs)
    mid = (lower < a[touse]) & (a[touse] < upper)
    short, long = a[touse] <= lower, a[touse] >= upper
    touse=short|long

    n=touse.sum()

    




    # objective function xT P x + qT x
    P = matrix(V[touse].loc[:,touse].values*52**2)
    price_vec = price_vec[touse]
    q=matrix(np.zeros(n))

    # and subject to Ax = b
    
    A = np.stack((a[touse].values,     # unit exposure to factor
                            price_vec.values)) # dollar neutral
    
    A=matrix(np.array(np.vstack((A, bs.loc[:,touse].values)),dtype=float))  # exposure to factor b1, b2
                            
    b = matrix([1.0, 0.0]+[0]*k)

    # G x <= h
    G=pd.DataFrame(np.eye(n),columns=touse[touse].index, index=touse[touse].index)

    G.loc[long,long]=np.eye(long.sum())*-1

    G = matrix(G.values)

    h = matrix(np.zeros(n))

    
    success = False
    holdings = None
    try:
        sol = solvers.qp(P, q, G, h, A, b)

        x = np.array(sol['x']).flatten()
        success = sol['status'] == 'optimal'
        if success:
            holdings = pd.Series(x, index=touse[touse].index)
    except:
        print('cvx didn''t find solution')
    return holdings, success







# data=pickle.load(open('../data/market_data.p', 'rb'))

cleanData = {fname[:-4]: pd.read_csv('../data/CleanedData/' + fname) for fname in os.listdir('../data/CleanedData')}
del cleanData['.DS_S']


for key in cleanData.keys():
    df=cleanData[key]
    df.index=pd.PeriodIndex(df.iloc[:,0], freq='D')
    df.index.name='date'
    df.drop(df.columns[0], axis=1, inplace=True)
    try:
        df[df=='NA']=np.nan
    except:
        pass
    if key in ['beta', 'MKshare']:
        cleanData[key]=-(df-df.mean())/df.std()
    if key == 'mom':
        cleanData[key]=(df-df.mean())/df.std()






formPeriod=12


startDate='2005-01-07'

curr_ret_data=cleanData['stock.ret'][:startDate]
curr_ret_data=curr_ret_data.dropna(how='all')
V=np.cov(curr_ret_data.transpose())
V=pd.DataFrame(V, index=curr_ret_data.columns, columns=curr_ret_data.columns)





holdings1, _ = factor_mimicking_portfolio_cvx(cleanData, 'beta', ['MKshare', 'B2P', 'mom'], V, startDate, 0.1)
holdings2, _ = factor_mimicking_portfolio_cvx(cleanData, 'mom', ['MKshare', 'B2P'], V, startDate, 0.1)






