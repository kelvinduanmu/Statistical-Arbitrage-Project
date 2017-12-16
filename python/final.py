import pickle
import pandas as pd
import numpy as np
import os
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


solvers.options['show_progress'] = False


def factor_mimicking_portfolio_cvx(data, exp_factor, neu_factors, covariance, date, cutoff=0.1):
    cutoffs = np.array([cutoff, 1 - cutoff]) * 100

    # focus on ones we care, others are zeors
    a = data[exp_factor].loc[date, ]
    touse = a.index[~a.isnull()]
    n = len(a)
    k = len(neu_factors)
    bs = np.ones((k, n)) * np.nan
    for i in range(k):
        b = data[neu_factors[i]].loc[date, ]
        bs[i, ] = b
        touse = touse.difference(b.index[b.isnull()])

    touse=touse.intersection(covariance.index)

    V = covariance[touse].loc[touse]
    a = a[touse]
    price_vec = data['PX.Weekly'].loc[date, touse]

    bs = pd.DataFrame(bs, columns=data[exp_factor].columns)
    bs = bs[touse]

    lower, upper = np.percentile(a[touse], cutoffs)
    mid = (lower < a[touse]) & (a[touse] < upper)
    short, long = a[touse] <= lower, a[touse] >= upper
    touse = short | long

    n = touse.sum()


    # objective function xT P x + qT x
    P = matrix(V[touse].loc[:, touse].values)
    price_vec = price_vec[touse]
    q = matrix(np.zeros(n))

    # and subject to Ax = b

    A = np.stack((a[touse].values,     # unit exposure to factor
                  price_vec.values))  # dollar neutral

    A = matrix(np.array(np.vstack((A, bs.loc[:, touse].values)), dtype=float))  # exposure to factor b1, b2

    b = matrix([1.0, 0.0] + [0] * k)

    # G x <= h
    G = pd.DataFrame(np.eye(n), columns=touse[touse].index, index=touse[touse].index)

    G.loc[long, long] = np.eye(long.sum()) * -1

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


def factor_premium(data, startDate, holding):
    ret=data['stock.ret'].loc[(pd.Period(startDate, freq='D')-2*365):startDate,holding.index]
    return (holding*ret).sum(axis=1).mean()
    



def combine_factors_portfolio_cvx(data, factor_premia, covariance, H_mat, trans_cost_mult, gamma, date):

    betas=data['beta'].loc[date, H_mat.index].values

    covariance = covariance[H_mat.index].loc[H_mat.index]

    n = covariance.shape[1]
    
    # objective function xT P x + qT x
    P = matrix((H_mat.T.dot((gamma*covariance+trans_cost_mult*np.eye(n)).dot(H_mat))).values)
    q = matrix(-factor_premia.reshape(-1,1))

    # G x <= h
    G = matrix(np.eye(len(factor_premia))*-1)

    h = matrix(np.zeros(2))

    success = False
    holdings = None
    try:
        sol = solvers.qp(P, q, G, h)

        x = np.array(sol['x']).flatten()
        success = sol['status'] == 'optimal'
        if success:
            holdings = x
    except:
        print('cvx didn''t find solution')
    return holdings, success




def data_preparation():
    # data=pickle.load(open('../data/market_data.p', 'rb'))

    cleanData = {fname[:-4]: pd.read_csv('../data/CleanedData/' + fname) for fname in os.listdir('../data/CleanedData')}
    del cleanData['.DS_S']

    keys=list(cleanData.keys())

    for key in keys:
        df = cleanData[key]
        df.index = pd.PeriodIndex(df.iloc[:, 0], freq='D')
        df.index.name = 'date'
        df.drop(df.columns[0], axis=1, inplace=True)
        try:
            df[df == 'NA'] = np.nan
        except:
            pass
        if key in ['beta', 'MKshare']:
            temp = -(df - df.mean()) / df.std()
            if key=='beta':
                cleanData['BAB']=temp
            else:
                cleanData[key]=temp
        if key == 'mom':
            cleanData[key] = (df - df.mean()) / df.std()

    return cleanData

def strategy_simulation(cleanData, startDate, holdingPeriod, trans_cost_mult, gamma):
    
    
    curr_ret_data = cleanData['stock.ret'][:startDate]
    curr_ret_data = curr_ret_data.dropna(how='all')
    curr_ret_data=curr_ret_data.drop(curr_ret_data.loc[:,curr_ret_data.isnull().any()].columns, axis=1)
    V = np.cov(curr_ret_data.transpose())
    V = pd.DataFrame(V, index=curr_ret_data.columns, columns=curr_ret_data.columns)

    holdings1, _ = factor_mimicking_portfolio_cvx(cleanData, 'BAB', ['MKshare', 'B2P', 'mom', 'beta'], V, startDate, 0.1)
    holdings2, _ = factor_mimicking_portfolio_cvx(cleanData, 'mom', ['MKshare', 'B2P', 'beta'], V, startDate, 0.1)

    H_mat = pd.DataFrame({'BAB': holdings1, 'MOM': holdings2}).fillna(0)
    factor_premia = np.array([factor_premium(cleanData, startDate, holding) for holding in [holdings1, holdings2]])
    
    weights, _ = combine_factors_portfolio_cvx(cleanData, factor_premia, V, H_mat, trans_cost_mult, gamma, startDate)

    holding_overall=H_mat.dot(weights)

    holdingPX=cleanData['PX.Weekly'][startDate:].head(12)
    PnL=(holding_overall*holdingPX[holding_overall.index]).sum(axis=1)

    transc=(holding_overall**2).sum()*trans_cost_mult


    return PnL[-1]-transc



cleanData=data_preparation()
startDate = '2005-01-07'
dates=cleanData['stock.ret'].loc[startDate:].index
dates=dates[:-100]
performance_means = list()

performance=pd.DataFrame({'PnL':[np.nan]}, index=dates)

ind_start = 0
for g in (np.arange(1, 11)*0.1):
    for t in (np.arange(1, 11)*0.005):
        for h in (np.arange(1, 14)*4):
            for i in range(len(dates)):
                try:
                    performance.loc[dates[i]]=strategy_simulation(cleanData, dates[i], h, t, g)
                    performance_means[ind_start] = performance.mean()
                except:
                    pass
                ind_start = ind_start + 1

#performance.plot(linestyle='None',marker='o')
#performance.mean()





