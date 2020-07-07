# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:03:15 2020

@author: pulki
"""
from pandas_datareader import data as pdr
# ! pip install quantstats --upgrade --no-cache-dir
import quantstats as qs
import strategies
import fix_yahoo_finance as yf
from tqdm import tqdm
from datetime import datetime
from datetime import date,timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
from hmmlearn import hmm
from sklearn.decomposition import PCA
import time
import regimeDetection


#Some utility functions

def Fit_RP(price,tickerEquity,N):
    ERCEquity=strategies.ERCRP()
    z=ERCEquity.get_allocations(price[tickerEquity],N)
    wEquity=pd.DataFrame(z,columns=tickerEquity,index=price.index)
    wEquity=wEquity.shift(1)
    wEquity.replace(np.nan,1/len(tickerEquity),inplace=True)
    rtnERCEquity=(wEquity*np.log(price[tickerEquity]).diff()).sum(axis=1)
    nvERCEquity=np.exp(rtnERCEquity.cumsum())
    shpERCEquity=rtnERCEquity.mean()/rtnERCEquity.std()*np.sqrt(252)
    return [rtnERCEquity,nvERCEquity,wEquity.loc[rtnERCEquity.index]]


def Fit_MSR(rf,dfMix,N):
    MVMix=strategies.MVPort(rf.loc[dfMix.index])
    o=MVMix.get_allocations(dfMix.values,N)
    wMix=pd.DataFrame(o,columns=dfMix.columns,index=dfMix.index)
    wMix=wMix.shift(1)
    wMix.replace(np.nan,1/dfMix.shape[1],inplace=True)
    rtnMVOMix=(wMix*np.log(dfMix).diff()).sum(axis=1)
    nvMVOMix=np.exp(rtnMVOMix.cumsum())
    shpMVOMix=(rtnMVOMix-rf.loc[dfMix.index]).mean()/rtnMVOMix.std()*np.sqrt(252)
    return [rtnMVOMix,nvMVOMix,wMix.loc[rtnMVOMix.index]]
