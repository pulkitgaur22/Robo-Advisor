
'''

@Authors : Pulkit Gaur, Hongyi Wu, Jiaqi Feng

'''

def goodPrint(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print (df)
########################################################################

## Import Statements, please install hmmlearn & quantstats


from pandas_datareader import data as pdr
import quantstats as qs
#import fix_yahoo_finance as yf
import yfinance as yf
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
import strategies
import utilityFuncs
import os
import regimeDetection as rgd
 
os.getcwd()
#######################################################################

"""## Data Pre Processing"""
stocks = ["SCO","SPY","GLD","VWO","IEF","EMB","lqd","VNQ","MNA","CAD=X","^IRX"]
#oil,sp500,gold,emerging_eq,us_7_10year_bonds,emerging bonds, hy_coroprate, reit etf, Hedge Fund, risk_free (13-week treasury bond), CAD
start = datetime(2010,1,1)
end = datetime(2020,6,1)

data = pdr.get_data_yahoo(stocks, start=start, end=end)
data = data["Adj Close"]

rf = data.iloc[1:,-1]/252
cad=data.iloc[1:,-2]/252
data = data.iloc[1:,:-2]
returns=data.pct_change().dropna()
clmns='Oil,SPX,Gold,EM EQ,US Bond,EMD,US HY,REIT,Hedge Fund'.split(',')
dataIdx=data.index.values
dataNamed=pd.DataFrame(data.values,index=dataIdx,columns=clmns).dropna()
rtnNamed=dataNamed.pct_change().dropna()



#######################################################################

#Portfolio Construction


tickerEquity=['XLY','XLI','XLF','XLV','XLK','XLP']
# Consumer Discritionary, Industrial, Financial, Health Care,Technology, Consumer Staples
tickerEquityCAD=['XMD.TO','XFN.TO','ZUH.TO','XIT.TO','ZDJ.TO']
# Mid/Small, Financial, Health Care, Information Technology, DJI

tickerCredit=["EMB","HYG",'LQD','MBB']
# EMD, HY, IG, MBS
tickerCreditCAD=['ZEF.TO','XHY.TO','ZCS.TO','XQB.TO']
# EMD, HY, Corp Bond, IG

tickerHedge=['IEF']
tickerHedgeCAD=['CGL.TO']

tickerPE=['PSP','IGF','VNQ','MNA']
# PE, Infra, REITs, HF
tickerPECAD=['CGR.TO','CIF.TO']
# REITs, Hedge fund HHF not long enough, Infra

<<<<<<< Updated upstream
#tickerBM=['SPY','HYG']
#stocks = tickerEquity+tickerCredit+tickerPE+tickerHedge+tickerAlternative+["SPY","CAD=X","^IRX"]
stocks = tickerEquity+tickerCredit+tickerPE+tickerHedge+["SPY","CAD=X","^IRX"]
stocksCAD = tickerEquityCAD+tickerCreditCAD+tickerPECAD+tickerHedgeCAD+["SPY","CAD=X","^IRX"]
=======
stocks = tickerEquity+tickerCredit+tickerAlts+tickerHedge+["SPY"]
stocksCAD = tickerEquityCAD+tickerCreditCAD+tickerAltsCAD+tickerHedgeCAD+["SPY"]
>>>>>>> Stashed changes

start = datetime(2010,1,1)
end = datetime(2020,6,1)

price,rtn=utilityFuncs.pull_data(stocks)
priceCAD,rtnCAD=utilityFuncs.pull_data(stocksCAD)

rtnTotal,nvTotal,wTotal=utilityFuncs.make_port(price,tickerEquity,tickerCredit,tickerPE,True)
#Results for NYSE
rtnTotalCAD,nvTotalCAD,wTotalCAD=utilityFuncs.make_port(priceCAD,tickerEquityCAD,tickerCreditCAD,tickerPECAD,True)
#Results for TSX

mutualDate=[i for i in wTotal.index if i in wTotalCAD.index]


<<<<<<< Updated upstream
weightMerged=pd.concat([wTotal.loc[mutualDate]/2,wTotalCAD.loc[mutualDate]/2],axis=1)
weightMerged.to_pickle('weights.pkl')
=======
    weightMerged=pd.concat([wTotal.loc[mutualDate]*0.556,wTotalCAD.loc[mutualDate]*0.444],axis=1)
    weightMerged.to_pickle('Data\\weights.pkl')
>>>>>>> Stashed changes


######################################################################
    
# Regime Detection
if 'Signal.pkl' in os.listdir(os.getcwd()+'\\Data'):
    signalSeries=pd.read_pickle('Data\\Signal.pkl')
else:
    dataHMM=pd.read_excel('Data\\HMM_data.xlsx',index_col=0)
    start = datetime(2008,1,1)
    end = datetime(2020,5,31)
    
    term_premium = pdr.get_data_yahoo(['^TYX','^IRX'], start=start, end=end)
    term_premium = term_premium["Adj Close"]
    term_premium = term_premium['^TYX']-term_premium['^IRX']
    
    dataHMM=dataHMM.loc[term_premium.index]
    dataHMM.iloc[:,-1]=term_premium.values
    
    dataInput=dataHMM
    dataInput_m=dataInput.resample('m').last()
    dataNormed1=rgd.percentile_data(dataInput,1)
    
    EMIndex1=(dataNormed1*[0.2,0.2,0.2,0.2,0.15,0.05]).sum(axis=1)# 1-year version
    
    model=hmm.GMMHMM(n_components=3, covariance_type="full",random_state= 0)
    
    newStates1=[]
    for i in tqdm(range(251,EMIndex1.size+1)):
        dataHMMTemp=EMIndex1.iloc[:i].values.reshape(-1,1)
        states=rgd.fix_states(model.fit(dataHMMTemp).predict(dataHMMTemp),EMIndex1.iloc[:i].values)
        newStates1.append(states[-1])
    
    dataHMMInit1=EMIndex1.iloc[:250].values.reshape(-1,1)
    modelInit1=model.fit(dataHMMInit1)
    stateInit1=rgd.fix_states(modelInit1.predict(dataHMMInit1),dataHMMInit1)
    updatedStates1=pd.Series(list(stateInit1)+newStates1,index=EMIndex1.index)
    signalOff=[i for i in range(1,updatedStates1.size) if updatedStates1[i-1]==1 and updatedStates1[i]==2]
    signalOn=[i for i in range(1,updatedStates1.size) if updatedStates1[i-1]==2 and updatedStates1[i]==1]
    
    signalSeries=pd.Series(0,index=updatedStates1.index)
    signalSeries[signalOn]=1
    signalSeries[signalOff]=-1
    
    signalSeries.to_pickle('Data\\Signal.pkl')
    
    
"""
#######################################################################

#Rebalancing and Portfolio Allocation
myMask=[]
temp=[]
x=2015

for i in range(6):
  temp.append(date(x+i,4,1))
  temp.append(date(x+i,10,1))

trialList=list(temp)
rebalancing=[]

#Getting all the rebalancing dates
for i in trialList:
  try:
    a= (weightsAll.loc[i])
    rebalancing.append(i)
  except:
    try:
       a= (weightsAll.loc[i+timedelta(days=1)])
       rebalancing.append(i+timedelta(days=1))
    except:
       try:
            a=(weightsAll.loc[i+timedelta(days=2)])
            rebalancing.append(i+timedelta(days=2))
       except:
            try:
                a=(weightsAll.loc[i+timedelta(days=3)])
                rebalancing.append(i+timedelta(days=3))
            except:
                   pass

for i in list(weightsAll.index):
    i = i.to_pydatetime().date()
    if i in (rebalancing):
        myMask.append(True)
    else:
        myMask.append(False)


#This dataframe contains all the portfolio weights
ERCWeight=weightsAll.loc[myMask]

#Performance Analysis
start=100000
portfolioValue=price.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
portfolioValue= (portfolioValue[ERCWeight.columns])
price=price[ERCWeight.columns].dropna()
price=price.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
investment=[]


for i in range(len(ERCWeight)):
  rebalanceDate=ERCWeight.index[i]

  try:
    endDate=ERCWeight.index[i+1] - timedelta(days=1)
  except:
    endDate=date(2020,6,1)

  relevantData=portfolioValue[rebalanceDate:endDate]
  rebalanceDate=relevantData.index[0]
  endDate=relevantData.index[-1]
  moneyAllocated=start*ERCWeight.iloc[i]
  noofUnits=moneyAllocated.divide(price.loc[rebalanceDate])

  portfolioValue[rebalanceDate:endDate]=portfolioValue[rebalanceDate:endDate]*list(noofUnits)
  investment.extend([100000+(i*10000)]*len(portfolioValue[rebalanceDate:endDate]))
  endvalue=portfolioValue.loc[endDate].sum()
  start=10000+endvalue


portfolioValue["Value"]=portfolioValue.sum(axis=1)
portfolioValue["Principal"]=investment


rebalancing = portfolioValue[~portfolioValue['Principal'].diff().isin([0])].index
portfolioValue["Return"]=portfolioValue["Value"].pct_change()
portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Return']=(portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value'])/((portfolioValue.shift(1).loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value'])+10000)-1

#Generate graphs for the portfolio
returnData=portfolioValue.Return.dropna()
qs.reports.full(returnData)

#Portfolio Exposures
plt.figure(figsize=(10,5))
labels=list(ERCWeight.columns)
plt.stackplot(list(ERCWeight.index),ERCWeight.values.T,labels=labels)
plt.title("Weights Rebalancing Evolution")
plt.legend()
plt.show()
plt.figure(figsize=(10,5))
temp=portfolioValue[ERCWeight.columns].div(portfolioValue.Value,axis=0)
plt.stackplot(list(portfolioValue.index),temp.T,labels=labels)
plt.title("Exposure by Asset Class")
plt.legend()

plt.figure(figsize=(14,5))
plt.plot(portfolioValue[ERCWeight.columns])
plt.legend(labels=labels)

#######################################################################

#Risk Models

df = pd.read_csv("Data/MacroData.csv", index_col='DATE')
df = df.loc[df.index>='2010-03-01'].iloc[:-2,:]
df = df.applymap(lambda x:float(x))
credit_risk_premium = (df['BAMLC0A4CBBBEY']-df['BAMLC0A1CAAAEY'])-(df['BAMLC0A4CBBBEY']-df['BAMLC0A1CAAAEY']).shift(1)

inflation = df['CPIAUCSL'].pct_change().dropna()*100*12
Industrial_prod_growth = df['INDPRO'].pct_change().dropna()*100
riskData = pd.DataFrame(inflation).join(Industrial_prod_growth).join(df.iloc[:,2:7])
riskData['CreditPremium'] = credit_risk_premium
riskData.columns = ['Inflation','IndustrialProdGrowth','T-Bill','Oil','Libor','House','Unemploy','CreditPremium']
riskData['Unexpected Inflation'] = (riskData['Inflation']-riskData['Inflation'].shift(1))-(riskData['T-Bill'].shift(1)-riskData['T-Bill'].shift(2))
riskData = riskData.dropna()
riskData = riskData[['IndustrialProdGrowth','Oil','Libor','House','Unemploy','CreditPremium','Unexpected Inflation']]


riskReturns=portfolioValue.Return.dropna()
riskReturns.index = riskReturns.index.map(lambda x:pd.to_datetime(str(x)))
monthlyReturns=riskReturns.groupby([riskReturns.index.year,riskReturns.index.month]).sum()
monthlyReturns.index.names=["Year","Month"]
monthlyReturns=monthlyReturns.reset_index(level=[0,1])
indexList=[]


for i in range(len(monthlyReturns)):
  indexList.append(date(int(monthlyReturns.iloc[i].Year),int(monthlyReturns.iloc[i].Month),1))


monthlyReturns.index=indexList
monthlyReturns.drop(["Year","Month"],axis=1,inplace=True)
monthlyReturns = monthlyReturns.set_index(pd.DatetimeIndex(monthlyReturns.index))


#Fitting the linear model
X=riskData.loc["2015-04-01":][riskData.columns]
Y=monthlyReturns.loc["2015-04-01":"2020-05-01"]
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
model.summary()

#######################################################################


#Regime Detection and Overlay strategy

dataHMM=pd.read_csv('Data/HMM_data.csv',index_col=0,parse_dates=True)
#VIX, FX Vol, IR Vol, Term Premium, Credit Spread, TED Spread
start = datetime(2008,1,1)
end = datetime(2020,5,31)

term_premium = pdr.get_data_yahoo(['^TYX','^IRX'], start=start, end=end)
term_premium = term_premium["Adj Close"]
term_premium = term_premium['^TYX']-term_premium['^IRX']

dataHMM=dataHMM.loc[term_premium.index]
dataHMM.iloc[:,-1]=term_premium.values

dataInput=dataHMM
dataInput_m=dataInput.resample('m').last()
dataNormed1=  regimeDetection.percentile_data(dataInput,1)

EMIndex1=(dataNormed1*[0.2,0.2,0.2,0.2,0.15,0.05]).sum(axis=1)# 1-year version

model=hmm.GMMHMM(n_components=3, covariance_type="full",random_state= 0)

newStates1=[]
for i in tqdm(range(251,EMIndex1.size+1)):
    dataHMMTemp=EMIndex1.iloc[:i].values.reshape(-1,1)
    states=regimeDetection.fix_states(model.fit(dataHMMTemp).predict(dataHMMTemp),EMIndex1.iloc[:i].values)
    newStates1.append(states[-1])

dataHMMInit1=EMIndex1.iloc[:250].values.reshape(-1,1)
modelInit1=model.fit(dataHMMInit1)
stateInit1=regimeDetection.fix_states(modelInit1.predict(dataHMMInit1),dataHMMInit1)
updatedStates1=pd.Series(list(stateInit1)+newStates1,index=EMIndex1.index)

rtnEquity=np.exp(price.iloc[:,:3]).diff()

idxValid1=[i for i in EMIndex1.index if i in rtnEquity.index]
rtnEquity=rtnEquity.loc[idxValid1]
rtnEquity['Risk_index']=EMIndex1.loc[rtnEquity.index]
rtnEquity['State']=updatedStates1.loc[rtnEquity.index]
rtnEquity['State_all_period']=0

hmmRiskData=rtnEquity.loc["2015-04-01":"2020-06-01"]
plt.figure(figsize=(15,8))
for i in range(len(hmmRiskData)):
    if (hmmRiskData.iloc[i].State) == 0 :
        plt.axvline(x=hmmRiskData.index[i],color='limegreen')
    elif (hmmRiskData.iloc[i].State) == 1:
        plt.axvline(x=hmmRiskData.index[i],color='khaki')
    else:
        plt.axvline(x=hmmRiskData.index[i],color='tomato')


plt.plot(100*dataNamed.loc["2015-04-01":"2020-06-01"].Gold/dataNamed.loc["2015-04-01"].Gold, color='blue',label="Gold")
plt.plot(100*dataNamed.loc["2015-04-01":"2020-06-01"].Oil/dataNamed.loc["2015-04-01"].Oil, color='brown',label="Oil")
plt.plot(100*dataNamed.loc["2015-04-01":"2020-06-01"]["US Bond"]/dataNamed.loc["2015-04-01"]["US Bond"], color='pink',label="Bond")
plt.plot(100*dataNamed.loc["2015-04-01":"2020-06-01"].SPX/dataNamed.loc["2015-04-01"].SPX, color='black',label="SP500")

plt.legend()
plt.show()

##############################################################

#Attribution


selectedIndex=1000

print ("Portfolio Value : ", round(portfolioValue["Value"].iloc[selectedIndex]))

nvEquity=portfolioValue[tickerEquity].iloc[selectedIndex]
tickerEqNames=["Consumer Discretionary", "Industrial", "Financial", "Health Care","Technology","Consumer Staples"]
nvEquity.index=tickerEqNames
print ("Equity Exposure : ",sum(round(nvEquity)))
goodPrint(round(nvEquity).to_string())

nvCredit=portfolioValue[tickerCredit].iloc[selectedIndex]
tickerCreditNames= [ "Emerging Markets", "High Yield", "Investment Grade", "Mortgage Backed Securities"]
nvCredit.index=tickerCreditNames
print ("Credit Exposure : ",sum(round(nvCredit)))
goodPrint(round(nvCredit).to_string())


nvHF=portfolioValue[tickerHedge].iloc[selectedIndex]
print ("Hedge Fund Exposure : ",sum(nvHF))
nvPE=portfolioValue[tickerPE].iloc[selectedIndex]
print ("Private Equity Exposure : ",sum(nvPE))
nvAlts=portfolioValue[tickerAlternative].iloc[selectedIndex]
print ("Merger Arb. Exposure : ",sum(nvAlts))


#Return Attribution

#Risk Attribution

"""