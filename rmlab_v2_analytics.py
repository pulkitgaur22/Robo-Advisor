'''
@Authors : Pulkit Gaur, Hongyi Wu, Jiaqi Feng

This code is written for the course MMF 2025 - Risk Management Laboratory
by Dr. Dan Rosen. 
Robo Advisor
'''
########################################################################

## Import Statements, please install hmmlearn & quantstats
    
from pandas_datareader import data as pdr
import quantstats as qs
from scipy.stats import norm 
import yfinance as yf
from tqdm import tqdm
import math
from datetime import datetime
from datetime import date,timedelta
import matplotlib.pyplot as plt
import numpy as np;np.random.seed(0)
import pandas as pd
import statsmodels.api as sm
import seaborn as sns; sns.set()
from scipy import stats
from hmmlearn import hmm
from sklearn.decomposition import PCA
import regimeDetection
import strategies
import utilityFuncs
import os
import metricsCalculator
import regimeDetection as rgd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import riskModel
os.getcwd()

#######################################################################

"""## Initial Data Pre Processing - Exploration"""

stocks = ["SCO","SPY","GLD","VWO","IEF","EMB","lqd","VNQ","MNA","CAD=X","^IRX"]
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

"""## Actual Data and Portfolio Construction"""

#Portfolio Construction
#US Equity Tickers
tickerEquity=['XLY','XLI','XLF','XLV','XLK','XLP']
tickerEqNamesUS=["Consumer Discretionary", "Industrial", "Financial", "Health Care","Technology","Consumer Staples"]

#CAD Equity Tickers
tickerEquityCAD=['XMD.TO','XFN.TO','ZUH.TO','XIT.TO','ZDJ.TO']
tickerEqNamesCAD=["Mid_Small_CAD", "Financial_CAD", "Health Care_CAD", "Information Technology_CAD", "DJI_CAD"]

#US Credit Tickers
tickerCredit=["EMB","HYG",'LQD','MBB']
tickerCreditNamesUSD= [ "Emerging Markets", "High Yield", "Investment Grade", "Mortgage Backed Securities"]

#CAD Credit Tickers
tickerCreditCAD=['ZEF.TO','XHY.TO','ZCS.TO','XQB.TO']
tickerCreditNamesCAD= [ "Emerging Markets_CAD", "High Yield_CAD", "Corporate Bonds_CAD","Investment Grade_CAD"]

#Hedge Assets- gold in CAD and us treasury- used when regime changes
tickerHedge=['IEF']
tickerHNamesUSD=["US_Treasury"]
tickerHedgeCAD=['CGL.TO']
tickerHNamesCAD=["Gold_CAD"]

#US Alternatives Tickers
tickerAlts=['PSP','IGF','VNQ','MNA']
tickerAltsNamesUSD=["PE", "Infra", "REITs", "HF"]

#CAD Alternatives Tickers
tickerAltsCAD=['CGR.TO','CIF.TO']
tickerAltsNamesCAD=["REITs_CAD", "Infra_CAD"]

#Downloading the FX and Overnight Interest Rate Data
start = datetime(2015,4,1)
end = datetime(2020,6,1)
fx = pdr.get_data_yahoo("CAD=X", start=start, end=end)
fxData = fx["Adj Close"]
oRates=pd.read_csv("Data/canadaOvernight.csv",index_col=0,parse_dates=True).sort_index()

#Downloading data from Yahoo Finance API
stocks = tickerEquity+tickerCredit+tickerAlts+tickerHedge+["SPY","CAD=X","^IRX"]
stocksCAD = tickerEquityCAD+tickerCreditCAD+tickerAltsCAD+tickerHedgeCAD+["SPY","CAD=X","^IRX"]

start = datetime(2010,1,1)
end = datetime(2020,6,1)
price,rtn=utilityFuncs.pull_data(stocks)
priceCAD,rtnCAD=utilityFuncs.pull_data(stocksCAD)
commonDate=[i for i in price.index if i in priceCAD.index]
priceMerged=pd.concat([price.loc[commonDate],priceCAD.loc[commonDate]],axis=1)

start = datetime(2015,4,1)
end = datetime(2020,6,1)
priceHedge= pdr.get_data_yahoo(tickerHedge+tickerHedgeCAD, start=start, end=end)["Adj Close"]
priceHedge= priceHedge.ffill(axis=0).dropna()

#Calculate weights for US tickers
rtnTotal,nvTotal,wTotal,rtnBreakDown=utilityFuncs.make_port(price,tickerEquity,tickerCredit,tickerAlts,True)
#Calculate weights for CAD tickers
rtnTotalCAD,nvTotalCAD,wTotalCAD,rtnBreakDownCAD=utilityFuncs.make_port(priceCAD,tickerEquityCAD,tickerCreditCAD,tickerAltsCAD,True)

#Merge the Weights in one file
mutualDate=[i for i in wTotal.index if i in wTotalCAD.index]
weightMerged=pd.concat([wTotal.loc[mutualDate]*0.556,wTotalCAD.loc[mutualDate]*0.444],axis=1)
weightMerged.to_pickle('weights.pkl')

######################################################################
    
# Regime Detection and Signals Generation
'''
Input-  
Output- 


'''

if 'Signal.pkl' in os.listdir(os.getcwd()+'\\Data'):
    
    signalSeries=pd.read_pickle('Data\\Signal.pkl')
    
else:
    
    #Using different factors like Fixed Income Vol, Equity Vol, FX Vol, Term Premium,
    #Credit Premium, Move Index etc.
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
    
    #Learning from the data
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

    #Generating Signal Series
    signalSeries=pd.Series(0,index=updatedStates1.index)
    signalSeries[signalOn]=1
    signalSeries[signalOff]=-1
    signalSeries.to_pickle('Data\\Signal.pkl')
    
#######################################################################

'''Rebalancing and Portfolio Allocation'''

    
#Renaming the weights
weightsAll=weightMerged

#Finding the dates to rebalance    
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


#Using the mask, this dataframe contains all the portfolio weights
ERCWeight=weightsAll.loc[myMask]

#Automated Backtesting for Main Portfolio

#This is the value that is used for investing in main portfolio
start=90000
portfolioValue=priceMerged.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
portfolioValue= (portfolioValue[ERCWeight.columns])
price=priceMerged[ERCWeight.columns].dropna()
price=price.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()

#This two lists will hold the principal amount and cash available.
investment=[]
cash=[]

#Backtesting Code
for i in range(len(ERCWeight)):
    
  rebalanceDate=ERCWeight.index[i]

  #finding start and end date for a rebalancing period.  
  try:
    endDate=ERCWeight.index[i+1] - timedelta(days=1)
  except:
    endDate=date(2020,6,1)

  relevantData=portfolioValue[rebalanceDate:endDate]
  rebalanceDate=relevantData.index[0]
  endDate=relevantData.index[-1]
  
  
  #Money Allocated to each of the asset in CAD
  moneyAllocated=start*ERCWeight.iloc[i]
  
  #Finding FX rate on first day and converting the USD prices to CAD
  try:
      fxConvert=fxData.loc[rebalanceDate]
  except:
      fxConvert=fxData.loc[rebalanceDate.date()-timedelta(days=1)]
      
      
  usTickers=[i for i in list(price.columns) if (i[-2:] != "TO")]
  priceinCAD=price.copy().loc[rebalanceDate]
  priceinCAD[usTickers]*=fxConvert

  # Number of Units to buy for each asset in each period
  noofUnits=moneyAllocated.divide(priceinCAD)
  
  # Adding all the values and evolution of value for each asset in a period.
  portfolioValue[rebalanceDate:endDate]=portfolioValue[rebalanceDate:endDate]*list(noofUnits)
  investment.extend([100000+(i*10000)]*len(portfolioValue[rebalanceDate:endDate]))
  cash.extend([10000+(i*1000)]*len(portfolioValue[rebalanceDate:endDate]))

  #Figuring out the value of portfolio on the last day in CAD that will be used for
  #reinvesting next period
  priceinCAD=portfolioValue.copy().loc[endDate]
  
  try:
      fxConvert=fxData.loc[endDate]
  except:
      fxConvert=fxData.loc[endDate.date()-timedelta(days=1)]
  
  priceinCAD[usTickers]*=fxConvert
  endvalue=priceinCAD.sum()
  
  #This 9000 means the amount that is added in the next rebalancing period.
  start=9000+endvalue

#Adding the column for cash in the dataframe
portfolioValue["Cash"]=cash


#Adding the regime strategy overlay

trades=signalSeries.loc[pd.to_datetime('2015-04-01'):pd.to_datetime('2020-06-01')].dropna()
moneyAccount=portfolioValue.Cash.copy()
openPos=0

#This list has all the buy and sell dates for each round trip of trade.
regimeDates=[]

for i in range(len(moneyAccount)):
    try:
        currentIndex=moneyAccount.index[i]        
        if trades[currentIndex] == 1 and openPos==0:
            buyIndex=currentIndex
            buyPrice=priceHedge.loc[(currentIndex.date())]
            openPos=1

        elif trades[moneyAccount.index[i]] == -1 and openPos==1:
            sellPrice=priceHedge.loc[(currentIndex.date())]
            openPos=0
            regimeDates.append([buyIndex,currentIndex])
    except:
            pass

# This dataframe holds the price for GOLD and US Treasury
priceHedge2=priceHedge.copy()
for i in priceHedge.index:
    if i not in portfolioValue.index:
        priceHedge2.drop(i,inplace=True)
priceHedge=priceHedge2

#This holds the price for the assets during each round trip
tradeData=[]
for i in range(len(regimeDates)):   
    buyDate= regimeDates[i][0]
    sellDate= regimeDates[i][1]
    goldData=priceHedge.loc[buyDate:sellDate]["CGL.TO"]/priceHedge.loc[buyDate]["CGL.TO"]
    treaData=priceHedge.loc[buyDate:sellDate].IEF/priceHedge.loc[buyDate].IEF
    tradeData.append([goldData,treaData])

   
# Finding the value of cash, gold and US Treasury for the 5 years
# and M2m/PL calculation    
cashValue=[moneyAccount.iloc[0]]
treaValue=[0]
goldValue=[0] 
j=0   
buyDates= [i[0] for i in regimeDates]  
sellDates= [i[1] for i in regimeDates]   
numberofDays=0
openPos=False

for i in range(len(portfolioValue)-1):
    
    currentIndex=portfolioValue.index[i]
    ORate=oRates.loc[currentIndex.date()]/36500
    
    if currentIndex in rebalancing[1:]:
        
        if openPos==True:
            cashValue[i:i+numberofDays+1]=np.add(cashValue[i:i+numberofDays+1],1000)
        else:
            cashValue[i]=cashValue[i]+1000
    
    if openPos==True:
        
        if numberofDays>0:
            numberofDays-=1
            continue
        
        elif numberofDays==0:
            cashValue[i]=goldValue[i]+treaValue[i]+cashValue[i]
            goldValue[i]=0
            treaValue[i]=0
            openPos=False
           
         
    if currentIndex in buyDates:
        
        numberofDays=len(tradeData[j][0])-2
        goldData=np.multiply(list(tradeData[j][0]),float(cashValue[i]/2))
        treaData=np.multiply(list(tradeData[j][1]),float(cashValue[i]/2))      
        goldValue[i]=(cashValue[i]/2)
        treaValue[i]=(cashValue[i]/2)
        cashValue[i]=0
        goldValue.extend(list(goldData[1:]))
        treaValue.extend(list(treaData[1:]))
        cashValue.extend(len(goldData[1:])*[0])
        j+=1
        openPos=True

    else:
         
         cashValue.append((cashValue[i])*(1+float(ORate)))
         treaValue.append(0)
         goldValue.append(0)


#Adding the value of the regime assets in the main dataframe.
portfolioValue["Cash"]=cashValue
portfolioValue["CGL.TO"]=goldValue
portfolioValue["IEF"]=treaValue

#Adding some more columns for convinience in analysis
usTickers.append("IEF")
cadTickers=list(set(portfolioValue.columns)-set(usTickers))

portfolioValue["USDTickers"]=portfolioValue[usTickers].sum(axis=1)
portfolioValue["CADTickers"]=portfolioValue[cadTickers].sum(axis=1)

portfolioValue=portfolioValue.join(fxData)
portfolioValue.ffill(axis=0,inplace=True)
portfolioValue["USDTickers_CAD"]=portfolioValue["USDTickers"].multiply(portfolioValue["Adj Close"])
# portfolioValue.drop(["Adj Close"],inplace=True,axis=1)    
    
portfolioValue["Principal"]=investment
portfolioValue["Value_CAD"]=portfolioValue["CADTickers"]+portfolioValue["USDTickers_CAD"]

rebalancing = portfolioValue[~portfolioValue['Principal'].diff().isin([0])].index
portfolioValue["Return"]=portfolioValue["Value_CAD"].pct_change()
portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Return']=(portfolioValue.loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value_CAD'])/((portfolioValue.shift(1).loc[list(portfolioValue.loc[portfolioValue.index.isin(rebalancing)][1:].index),'Value_CAD'])+10000)-1

###################################

''' Analysis and Metrics Calculations'''
print ()
print ("###################Portfolio Stats##################")
equityTickers=tickerEquity+tickerEquityCAD
creditTickers=tickerCredit+tickerCreditCAD
altsTickers=tickerAlts+tickerAltsCAD

#Generate graphs for the portfolio
returnData=portfolioValue.Return.dropna()
metricsCalculator.portfolioGraphsandStats(returnData)

#Evolution of USD/CAD exposure
metricsCalculator.usdcadExposures(portfolioValue)

#Asset Classes Weights Evolutions
metricsCalculator.weightsEvolution(portfolioValue,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD)

#Notional Value in each Asset
metricsCalculator.nvCalculator(portfolioValue,len(portfolioValue)-1,equityTickers,creditTickers,altsTickers,tickerEqNamesUS,tickerEqNamesCAD,tickerCreditNamesUSD,tickerCreditNamesCAD,tickerAltsNamesUSD,tickerAltsNamesCAD)

#Important Stats for Performance
metricsCalculator.get_stats(portfolioValue,rebalancing)

#Transaction Costs
metricsCalculator.txnCostCalc(portfolioValue,rebalancing)

#Benchmark Comparison
print ()
print ("###################Benchmark Comparison##################")
benchmarkData=metricsCalculator.benchmarkComp(portfolioValue)

#Exposure Plots
exposure = metricsCalculator.getExposure(portfolioValue,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD,tickerHedge,tickerHedgeCAD,'2020-06-01')
exposure['Weight'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))
plt.title("Exposures")
plt.show()

#Return Attribution
print ()
print ("###################Return Attribution###################")
df = metricsCalculator.getReturnAttribution(portfolioValue,rebalancing,tickerEquity,tickerEquityCAD,tickerCredit,tickerCreditCAD,tickerAlts,tickerAltsCAD)
print (round((df/df.sum())*100,2))
df.plot(kind='pie',autopct='%.2f')
plt.title("Return Attribution")
plt.show()

#Risk Attribution
print ()
print ("###################Risk Attribution###################")
riskAttribution = metricsCalculator.getRiskAttribution(portfolioValue, rtnBreakDown,rtnBreakDownCAD,exposure,'2020-06-01')
print (round(riskAttribution*100,2))
riskAttribution['Risk Attribution'].plot.pie(autopct='%.2f', fontsize=12, figsize=(8, 8))

#####################################################
''' Risk Model'''
#Linear Model and then shocking one by one and also, copula based distribution
portReturns=benchmarkData.Port_Returns.dropna()
upScenario,downScenario,simup,simdown=riskModel.getResults(benchmarkData,portfolioValue,tickerAlts,tickerCredit,tickerEquity,tickerHedge,tickerAltsCAD,tickerCreditCAD,tickerEquityCAD,tickerHedgeCAD)
upScenario=upScenario.drop(["constant"],axis=1)
upScenario["Value"]=upScenario["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
upScenario["Portfolio Estimated Return"]=upScenario["Portfolio Estimated Return"]*100

utilityFuncs.goodPrint(round(upScenario,2))

downScenario=downScenario.drop(["constant"],axis=1)
downScenario["Value"]=downScenario["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
downScenario["Portfolio Estimated Return"]=downScenario["Portfolio Estimated Return"]*100
utilityFuncs.goodPrint(round(downScenario,2))

simup["Value"]=simup["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
simup["Portfolio Estimated Return"]=simup["Portfolio Estimated Return"]*100

simdown["Value"]=simdown["Portfolio Estimated Return"]*portfolioValue.iloc[-1].Value_CAD
simdown["Portfolio Estimated Return"]=simdown["Portfolio Estimated Return"]*100

simup=simup.drop(["constant"],axis=1)
simdown=simdown.drop(["constant"],axis=1)

utilityFuncs.goodPrint(round(simup,2))
utilityFuncs.goodPrint(round(simdown,2))

##################################################

''' Stressed VaR '''















