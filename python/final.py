import pickle
import pandas as pd
import numpy as np
import os
from cvxopt import matrix, solvers



solvers.options['show_progress'] = False


def factor_mimicking_portfolio_cvx(data, exp_factor, neu_factors, covariance, date):

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
    
    n=len(touse)

    

    bs=pd.DataFrame(bs, columns=a.index)



    # objective function xT P x + qT x
    P = matrix(covariance[touse].loc[touse].values)
    price_vec = cleanData['PX.Weekly'].loc[date,touse]
    q=matrix(np.zeros(n))

    # and subject to Ax = b
    A = np.stack((a[touse].values,     # unit exposure to factor
                            price_vec.values)) # dollar neutral

    A=matrix(np.array(np.vstack((A, bs[touse].values)),dtype=float))  # exposure to factor b1, b2

    b = matrix([1.0, 0.0]+[0]*k)

    # G x <= h
    G = matrix(np.zeros((1,n)))

    h = matrix(np.ones(1))

    
    success = False
    holdings = None
    import pdb; pdb.set_trace()
    try:
        sol = solvers.qp(P, q, G, h, A, b)

        x = np.array(sol['x']).flatten()
        success = sol['status'] == 'optimal'
        if success:
            holdings = pd.Series(x, index=a.index)
    except:
        print('cvx didn''t find solution')
    return success, holdings







#data=pickle.load(open('../data/market_data.p', 'rb'))

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
        cleanData[key]=(df-df.mean())/df.std()





formPeriod=12


startDate='2005-01-07'

curr_ret_data=cleanData['stock.ret'][:startDate]
curr_ret_data=curr_ret_data[~curr_ret_data.isnull().all(axis=1)]
V=np.cov(curr_ret_data.transpose())
V=pd.DataFrame(V, index=curr_ret_data.columns, columns=curr_ret_data.columns)




success, holdings = factor_mimicking_portfolio_cvx(cleanData, 'beta', ['MKshare', 'B2P'], V, startDate)






