# this is implementing Prado's book
# machine learning in finance

import csv
import pandas as pd
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt
# %matplotlib inline
from utils_4 import *
import time

# ch3 
# SNIPPET 3.1 DAILY VOLATILITY ESTIMATES
def getDailyVol(close,span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-1)
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

# SNIPPET 3.2 TRIPLE-BARRIER LABELING METHOD
def applyPtSlOnT1(close,events,ptSl,molecule):
    events_=events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    # events_=events.loc[molecule] 
    # out=events_[['t1']].copy(deep=True)
    if ptSl[0]>0:
        pt=ptSl[0]*events_['trgt']
    else:
        pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0:
        sl=-ptSl[1]*events_['trgt']
    else:
        sl=pd.Series(index=events.index) # NaNs
    for loc,t1 in events_.t1.fillna(close.index[-1]).iteritems():
        df0=close[loc:t1] # path prices
    # #     df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        df0=(df0/close[loc]-1) # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    return out


# # SNIPPET 3.3 GETTING THE TIME OF FIRST TOUCH
def getEvents(close,tEvents,ptSl,trgt,minRet,numThreads,t1=False):
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet
    #2) get t1 (max holding period)
    if t1 is False:
        t1=pd.Series(pd.NaT,index=tEvents)
    #3) form events object, apply stop loss on t1 
    side_=pd.Series(1.,index=trgt.index) 
    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_}, \
            axis=1).dropna(subset=['trgt']) 
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index), \
            numThreads=numThreads,close=close,events=events,ptSl=[ptSl,ptSl]) 
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan 
    events=events.drop('side',axis=1)
    return events
# ch7 cross_validation
# purging P106

# SNIPPET 3.5 LABELING FOR SIDE AND SIZE
def getBins(events,close):
    #1) prices aligned with events
    events_=events.dropna(subset=['t1']) 
    px=events_.index.union(events_['t1'].values).drop_duplicates() 
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index) 
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1 
    out['bin']=np.sign(out['ret'])
    return out

# # SNIPPET 3.6 EXPANDING getEvents TO INCORPORATE META-LABELING
# def getEvents(close,tEvents,ptSl,trgt,minRet,numThreads,t1=False,side=None):
#     #1) get target
#     trgt=trgt.loc[tEvents]
#     trgt=trgt[trgt>minRet] # minRet
#     #2) get t1 (max holding period)
#     if t1 is False:
#         t1=pd.Series(pd.NaT,index=tEvents)
#     #3) form events object, apply stop loss on t1
#     if side is None:
#         side_,ptSl_=pd.Series(1.,index=trgt.index),[ptSl[0],ptSl[0]] 
#     else:
#         side_,ptSl_=side.loc[trgt.index],ptSl[:2] 
#         events=pd.concat({'t1':t1,'trgt':trgt,'side':side_}, 
#                 axis=1).dropna(subset=['trgt'])
        
# # SNIPPET 3.7 EXPANDING getBins TO INCORPORATE META-LABELING
# def getBins(events,close):
#     #1) prices aligned with events
#     events_=events.dropna(subset=['t1']) 
#     px=events_.index.union(events_['t1'].values).drop_duplicates() 
#     px=close.reindex(px,method='bfill')
#     #2) create out object
#     out=pd.DataFrame(index=events_.index) 
#     out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1 
#     if 'side' in events_:
#         out['ret']*=events_['side'] # meta-labeling out['bin']=np.sign(out['ret'])
#     if 'side' in events_:
#         out.loc[out['ret']<=0,'bin']=0 # meta-labeling 
#     return out

# SNIPPET 3.8 DROPPING UNDER-POPULATED LABELS
def dropLabels(events,minPtc=.05):
    # apply weights, drop labels with insufficient examples 
    while True:
        df0=events['bin'].value_counts(normalize=True) 
        if df0.min()>minPct or df0.shape[0]<3:
            break 
        print 'dropped label',df0.argmin(),df0.min() 
        events=events[events['bin']!=df0.argmin()]
    return events


# SNIPPET 4.1 ESTIMATING THE UNIQUENESS OF A LABEL
def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed 
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count.

    '''
    #1) find events that span the period [molecule[0],molecule[-1]] 
    t1=t1.fillna(list( closeIdx )[-1]) # unclosed events still must impact other weights 
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0] 
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max() 
    #2) count events spanning a bar 
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()])) 
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.iteritems(): 
        count.loc[tIn:tOut]+=1
    return count.loc[molecule[0]:t1[molecule].max()]

# SNIPPET 4.2 ESTIMATING THE AVERAGE UNIQUENESS OF A LABEL
def mpSampleTW(t1,numCoEvents,molecule):
    # Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean() 
    return wght
# numCoEvents=mpPandasObj(mpNumCoEvents,('molecule',events.index),numThreads,
#                         closeIdx=close.index,t1=events['t1'])
# numCoEvents=numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')] 
# numCoEvents=numCoEvents.reindex(close.index).fillna(0) 
# out['tW']=mpPandasObj(mpSampleTW,('molecule',events.index),numThreads,
# t1=events['t1'],numCoEvents=numCoEvents)

# SNIPPET 4.3 BUILD AN INDICATOR MATRIX
def getIndMatrix(barIx,t1):
    # Get indicator matrix
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0])) 
    for i,(t0,t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1,i]=1
    return indM

# SNIPPET 4.4 COMPUTE AVERAGE UNIQUENESS
def getAvgUniqueness(indM):
    # Average uniqueness from indicator matrix 
    c=indM.sum(axis=1) # concurrency 
    u=indM.div(c,axis=0) # uniqueness 
    avgU=u[u>0].mean() # average uniqueness 
    return avgU

# SNIPPET 4.5 RETURN SAMPLE FROM SEQUENTIAL BOOTSTRAP
def seqBootstrap(indM,sLength=None):
    # Generate a sample via sequential bootstrap 
    if sLength is None:
        sLength=indM.shape[1] 
    phi=[]
    while len(phi)<sLength:
        avgU=pd.Series() 
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1] 
        prob=avgU/avgU.sum() # draw prob 
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi

# # SNIPPET 4.6 EXAMPLE OF SEQUENTIAL BOOTSTRAP
# def main():
#     t1=pd.Series([2,3,5],index=[0,2,4]) # t0,t1 for each feature obs 
#     barIx=range(t1.max()+1) # index of bars 
#     indM=getIndMatrix(barIx,t1) 
#     phi=np.random.choice(indM.columns,size=indM.shape[1])
#     print phi
#     print 'Standard uniqueness:',getAvgUniqueness(indM[phi]).mean() 
#     phi=seqBootstrap(indM)
#     print phi
#     print 'Sequential uniqueness:',getAvgUniqueness(indM[phi]).mean() 
#     return

# SNIPPET 4.7 GENERATING A RANDOM T1 SERIES
def getRndT1(numObs,numBars,maxH):
# random t1 Series
    t1=pd.Series()
    for i in xrange(numObs): 
        ix=np.random.randint(0,numBars) 
        val=ix+np.random.randint(1,maxH) 
        t1.loc[ix]=val
    return t1.sort_index()

# SNIPPET 4.8 UNIQUENESS FROM STANDARD AND SEQUENTIAL BOOTSTRAPS
def auxMC(numObs,numBars,maxH):
    # Parallelized auxiliary function
    t1=getRndT1(numObs,numBars,maxH) 
    barIx=range(t1.max()+1)
    indM=getIndMatrix(barIx,t1) 
    phi=np.random.choice(indM.columns,size=indM.shape[1]) 
    stdU=getAvgUniqueness(indM[phi]).mean() 
    phi=seqBootstrap(indM) 
    seqU=getAvgUniqueness(indM[phi]).mean()
    return {'stdU':stdU,'seqU':seqU}

# SNIPPET 5.1 WEIGHTING FUNCTION
def getWeights(d,size):
    # thres>0 drops insignificant weights
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_) 
    w=np.array(w[::-1]).reshape(-1,1) 
    return w

def plotWeights(dRange,nPlots,size):
    w=pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=getWeights(d,size=size) 
        w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d]) 
        w=w.join(w_,how='outer')
    ax=w.plot()
    ax.legend(loc='upper left');plt.show() 
    return

# SNIPPET 5.2 STANDARD FRACDIFF (EXPANDING WINDOW)
def fracDiff(series,d,thres=.01): 
    #1) Compute weights for the longest series 
    w=getWeights(d,series.shape[0])
    #2) Determine initial calcs to be skipped based on weight-loss threshold 
    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]
    #3) Apply weights to values
    df={}
    for name in series.columns:
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_ = pd.Series([0] * len(seriesF)) 
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]
            if not np.isfinite(series.loc[loc,name]):continue # exclude NAs 
            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True) 
    df=pd.concat(df,axis=1)
    return df

def getWeights_FFD(d, thres, lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_)<thres:
            break
        w.append(w_)
        k+=1
        ctr += 1
        if ctr == lim:
            break
    w = np.array(w[::-1]).reshape(-1,1)
    return w

# SNIPPET 5.3 THE NEW FIXED-WIDTH WINDOW FRACDIFF METHOD
def fracDiff_FFD(series,d,thres=1e-5): 
    #1) Compute weights for the longest series
    w=getWeights_FFD(d,thres,len(series))
    width=len(w)-1
    #2) Apply weights to values
    df={}
    for name in series.columns: 
        seriesF = series[[name]].fillna(method='ffill').dropna()
        df_=pd.Series([0.0]*len(seriesF)) 
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs 
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True) 
    df=pd.concat(df,axis=1)
    return df

# SNIPPET 5.4 FINDING THE MINIMUM D VALUE THAT PASSES THE ADF TEST
def plotMinFFD():
    from statsmodels.tsa.stattools import adfuller
    path,instName='/Users/jun/Documents/py/pairs_trading/record/','test'
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
#     df0=pd.read_csv(path+instName+'.csv',index_col=0,parse_dates=True)
    df0 = temp
    for d in np.linspace(0,1,11):
#         df1=np.log(df0[['ClosePrice']]).resample('1D').last() # downcast to daily obs
        df1=np.log(df0[['ClosePrice']])
        df2=fracDiff_FFD(df1,d,thres=.01)
        corr=np.corrcoef(df1.loc[df2.index,'ClosePrice'],df2['ClosePrice'])[0,1]
        df2=adfuller(df2['ClosePrice'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value
    out.to_csv(path+instName+'_testMinFFD.csv')
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    plt.savefig(path+instName+'_testMinFFD.png')
    return


# chapter 6, no snippet

# SNIPPET 7.1 PURGING OBSERVATION IN THE TRAINING SET
def getTrainTimes(t1,testTimes):
    trn=t1.copy(deep=True)
    for i,j in testTimes.iteritems():
        df0=trn[(i<=trn.index)&(trn.index<=j)].index # train starts within test '
        df1=trn[(i<=trn)&(trn<=j)].index # train ends within test 
        df2=trn[(trn.index<=i)&(j<=trn)].index # train envelops test 
        trn=trn.drop(df0.union(df1).union(df2))
    return trn

# getTrainTimes( pd.Series(range(200)), pd.Series([(100)]))

# SNIPPET 7.2 EMBARGO ON TRAINING OBSERVATIONS
def getEmbargoTimes(times,pctEmbargo):
    # Get embargo time for each bar
    step=int(times.shape[0]*pctEmbargo) 
    if step==0:
        mbrg=pd.Series(times,index=times) 
    else:
        mbrg=pd.Series(times[step:],index=times[:-step])
        mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:])) 
    return mbrg

# SNIPPET 7.3 CROSS-VALIDATION CLASS WHEN OBSERVATIONS OVERLAP
# class PurgedKFold(_BaseKFold):
import sklearn
# sklearn.model_selection.KFold
from sklearn.model_selection import KFold
class PurgedKFold(KFold):
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1,pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(n_splits,shuffle=False,random_state=None) 
        self.t1=t1
        self.pctEmbargo=pctEmbargo
        
    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index') 
        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[(i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),self.n_splits)]
        for i,j in test_starts:
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j] 
            maxT1Idx=self.t1.index.searchsorted(self.t1[test_indices].max()) 
            train_indices=self.t1.index.searchsorted(self.t1[self.t1<=t0].index) 
            if maxT1Idx<X.shape[0]: # right train (with embargo)
                train_indices=np.concatenate((train_indices,indices[maxT1Idx+mbrg:])) 
            yield train_indices,test_indices


# SNIPPET 7.4 USING THE PurgedKFold CLASS
def cvScore(clf,X,y,sample_weight,scoring='neg_log_loss',t1=None,cv=None,cvGen=None, pctEmbargo=None):    
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score
    if cvGen is None:
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged 
    score=[]
    for train,test in cvGen.split(X=X): 
        fit=clf.fit(X=X.iloc[train,:],y=y.iloc[train],sample_weight=sample_weight.iloc[train].values) 
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[test,:]) 
            score_=-log_loss(y.iloc[test],prob,sample_weight=sample_weight.iloc[test].values,labels=clf.classes_) 
        else:
            pred=fit.predict(X.iloc[test,:]) 
            score_=accuracy_score(y.iloc[test],pred,sample_weight= sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)


# ch 8 feature importance
# SNIPPET 8.2 MDI FEATURE IMPORTANCE
# Mean decrease impurity (MDI) is a fast, explanatory-importance (in-sample, IS) method 
# specific to tree-based classi ers, like RF.
def featImpMDI(fit,featNames):
    # feat importance based on IS mean impurity reduction
    df0={i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)} 
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) # because max_features=1
    imp=pd.concat({'mean':df0.mean(),'std':df0.std()*df0.shape[0]**-.5},axis=1) 
    imp/=imp['mean'].sum()
    return imp

# SNIPPET 8.3 MDA FEATURE IMPORTANCE
# Mean decrease accuracy (MDA) is a slow, predictive-importance (out-of-sample, OOS) method
def featImpMDA(clf,X,y,cv,sample_weight,t1,pctEmbargo,scoring='neg_log_loss'): 
        # feat importance based on OOS score reduction
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score 
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged cv 
    scr0,scr1=pd.Series(),pd.DataFrame(columns=X.columns)
    for i,(train,test) in enumerate(cvGen.split(X=X)): 
        X0,y0,w0=X.iloc[train,:],y.iloc[train],sample_weight.iloc[train] 
        X1,y1,w1=X.iloc[test,:],y.iloc[test],sample_weight.iloc[test] 
        fit=clf.fit(X=X0,y=y0,sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1) 
            scr0.loc[i]=-log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
        else:
            pred=fit.predict(X1) 
            scr0.loc[i]=accuracy_score(y1,pred,sample_weight=w1.values)
        for j in X.columns:
            X1_=X1.copy(deep=True)
            np.random.shuffle(X1_[j].values) # permutation of a single column 
            if scoring=='neg_log_loss':
                prob=fit.predict_proba(X1_) 
                scr1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,labels=clf.classes_)
            else:
                pred=fit.predict(X1_) 
                scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
    imp=(scr1* (-1)).add(scr0,axis=0)
    if scoring=='neg_log_loss':
        imp=imp/( scr1* (-1)) 
    else:
        imp=imp/(1.( scr1 * (-1)) )
    imp=pd.concat({'mean':imp.mean(),'std':imp.std()*imp.shape[0]**-.5},axis=1) 
    return imp,scr0.mean()

# SNIPPET 8.4 IMPLEMENTATION OF SFI
def auxFeatImpSFI(featNames,clf,trnsX,cont,scoring,cvGen): 
    imp=pd.DataFrame(columns=['mean','std'])
    for featName in featNames:
        df0=cvScore(clf,X=trnsX[[featName]],y=cont['bin'],sample_weight=cont['w'], scoring=scoring,cvGen=cvGen)
        imp.loc[featName,'mean']=df0.mean()
        imp.loc[featName,'std']=df0.std()*df0.shape[0]**-.5 
    return imp

# SNIPPET 8.5 COMPUTATION OF ORTHOGONAL FEATURES
def get_eVec(dot,varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal,eVec=np.linalg.eigh(dot)
    idx=eVal.argsort()[::-1] # arguments for sorting eVal desc 
    eVal,eVec=eVal[idx],eVec[:,idx]
    #2) only positive eVals
    eVal=pd.Series(eVal,index=['PC_'+str(i+1) for i in range(eVal.shape[0])]) 
    eVec=pd.DataFrame(eVec,index=dot.index,columns=eVal.index) 
    eVec=eVec.loc[:,eVal.index]
    #3) reduce dimension, form PCs
    cumVar=eVal.cumsum()/eVal.sum()
    dim=cumVar.values.searchsorted(varThres) 
    eVal,eVec=eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
    return eVal,eVec
#----------------------------------------------------------------- def orthoFeats(dfX,varThres=.95):
# Given a dataframe dfX of features, compute orthofeatures dfP dfZ=dfX.sub(dfX.mean(),axis=1).div(dfX.std(),axis=1) # standardize dot=pd.DataFrame(np.dot(dfZ.T,dfZ),index=dfX.columns,columns=dfX.columns) eVal,eVec=get_eVec(dot,varThres)
# dfP=np.dot(dfZ,eVec) return dfP


# SNIPPET 8.6 Kendall tau
# from scipy.stats import weightedtau
# featImp=np.array([.55,.33,.07,.05]) # feature importance >>> pcRank=np.array([1,2,4,3]) # PCA rank
# weightedtau(featImp,pcRank**-1.)[0]

# SNIPPET 8.7 CREATING A SYNTHETIC DATASET
def getTestData(n_features=40,n_informative=10,n_redundant=10,n_samples=10000): # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification 
    trnsX,cont=make_classification(n_samples=n_samples,n_features=n_features,
    n_informative=n_informative,n_redundant=n_redundant,random_state=0,
    shuffle=False) 
    df0=pd.DatetimeIndex(periods=n_samples,freq=pd.tseries.offsets.BDay(),
    end=pd.datetime.today()) 
    trnsX,cont=pd.DataFrame(trnsX,index=df0),
    pd.Series(cont,index=df0).to_frame('bin') 
    df0=['I_'+str(i) for i in xrange(n_informative)]+['R_'+str(i) for i in xrange(n_redundant)] 
    df0+=['N_'+str(i) for i in xrange(n_features-len(df0))] 
    trnsX.columns=df0
    cont['w']=1./cont.shape[0] 
    cont['t1']=pd.Series(cont.index,index=cont.index) 
    return trnsX,cont

# SNIPPET 8.8 CALLING FEATURE IMPORTANCE FOR ANY METHOD
def featImportance(trnsX,cont,n_estimators=1000,cv=10,max_samples=1.,numThreads=24, pctEmbargo=0,scoring='accuracy',method='SFI',minWLeaf=0.,**kargs):
    # feature importance from a random forest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from mpEngine import mpPandasObj
    n_jobs=(-1 if numThreads>1 else 1) # run 1 thread with ht_helper in dirac1 #1) prepare classifier,cv. max_features=1, to prevent masking 
    clf=DecisionTreeClassifier(criterion='entropy',max_features=1,
    class_weight='balanced',min_weight_fraction_leaf=minWLeaf)
    clf=BaggingClassifier(base_estimator=clf,n_estimators=n_estimators,
    max_features=1.,max_samples=max_samples,oob_score=True,n_jobs=n_jobs) 
    fit=clf.fit(X=trnsX,y=cont['bin'],sample_weight=cont['w'].values) 
    oob=fit.oob_score_
    if method=='MDI':
        imp=featImpMDI(fit,featNames=trnsX.columns) 
        oos=cvScore(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring).mean() 
    elif method=='MDA':
        imp,oos=featImpMDA(clf,X=trnsX,y=cont['bin'],cv=cv,sample_weight=cont['w'],
        t1=cont['t1'],pctEmbargo=pctEmbargo,scoring=scoring)
    elif method=='SFI': 
        cvGen=PurgedKFold(n_splits=cv,t1=cont['t1'],pctEmbargo=pctEmbargo) 
        oos=cvScore(clf,X=trnsX,y=cont['bin'],sample_weight=cont['w'],scoring=scoring,
    cvGen=cvGen).mean()
    clf.n_jobs=1 # paralellize auxFeatImpSFI rather than clf 
    imp=mpPandasObj(auxFeatImpSFI,('featNames',trnsX.columns),numThreads,clf=clf,trnsX=trnsX,cont=cont,scoring=scoring,cvGen=cvGen) 
    return imp,oob,oos
#     Finally, we need a main function to call all components, from data generation to feature importance analysis to collection and processing of output. These tasks are performed by Snippet 8.9.


# SNIPPET 8.9 CALLING ALL COMPONENTS
def testFunc(n_features=40,n_informative=10,n_redundant=10,n_estimators=1000, n_samples=10000,cv=10):
    trnsX,cont=getTestData(n_features,n_informative,n_redundant,n_samples)
    dict0={'minWLeaf':[0.],'scoring':['accuracy'],'method':['MDI','MDA','SFI'], 'max_samples':[1.]}
    jobs,out=(dict(izip(dict0,i)) for i in product(*dict0.values())),[] 
    kargs={'pathOut':'./testFunc/','n_estimators':n_estimators,
        'tag':'testFunc','cv':cv}
    for job in jobs:
        job['simNum']=job['method']+'_'+job['scoring']+'_'+'%.2f'%job['minWLeaf']+ '_'+str(job['max_samples'])
        print job['simNum']
        kargs.update(job) 
        imp,oob,oos=featImportance(trnsX=trnsX,cont=cont,**kargs) 
        plotFeatImportance(imp=imp,oob=oob,oos=oos,**kargs) 
        df0=imp[['mean']]/imp['mean'].abs().sum() 
        df0['type']=[i[0] for i in df0.index] 
        df0=df0.groupby('type')['mean'].sum().to_dict() 
        df0.update({'oob':oob,'oos':oos});
        df0.update(job) 
        out.append(df0)
    out=pd.DataFrame(out).sort_values(['method','scoring','minWLeaf','max_samples']) 
    out=out['method','scoring','minWLeaf','max_samples','I','R','N','oob','oos'] 
    out.to_csv(kargs['pathOut']+'stats.csv')
    return out



# SNIPPET 9.1 GRID SEARCH WITH PURGED K-FOLD CROSS-VALIDATION
def clfHyperFit(feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.], n_jobs=-1,pctEmbargo=0,**fit_params):
    if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling 
    else:scoring='neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data 
    inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged 
    gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
    gs=gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline #2) fit validated model on the entirety of the data
    if bagging[1]>0:
        gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),max_samples=float(bagging[1]), max_features=float(bagging[2]),n_jobs=n_jobs)
        gs=gs.fit(feat,lbl,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs=Pipeline([('bag',gs)]) 
    return gs

from sklearn.pipeline import Pipeline
class MyPipeline(Pipeline):
    def fit(self,X,y,sample_weight=None,**fit_params):
        if sample_weight is not None: 
            fit_params[self.steps[-1][0]+'__sample_weight']=sample_weight
        return super(MyPipeline,self).fit(X,y,**fit_params)

# SNIPPET 9.3 RANDOMIZED SEARCH WITH PURGED K-FOLD CV
def clfHyperFit(feat,lbl,t1,pipe_clf,param_grid,cv=3,bagging=[0,None,1.], rndSearchIter=0,n_jobs=-1,pctEmbargo=0,**fit_params):
    if set(lbl.values)=={0,1}:scoring='f1' # f1 for meta-labeling 
    else:scoring='neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data 
    inner_cv=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged 
    if rndSearchIter==0:
        gs=GridSearchCV(estimator=pipe_clf,param_grid=param_grid, scoring=scoring,cv=inner_cv,n_jobs=n_jobs,iid=False)
    else:
        gs=RandomizedSearchCV(estimator=pipe_clf,param_distributions=param_grid,scoring=scoring,cv=inner_cv,n_jobs=n_jobs, iid=False,n_iter=rndSearchIter)
    gs=gs.fit(feat,lbl,**fit_params).best_estimator_ # pipeline 
    #2) fit validated model on the entirety of the data
    if bagging[1]>0:
        gs=BaggingClassifier(base_estimator=MyPipeline(gs.steps), n_estimators=int(bagging[0]),max_samples=float(bagging[1]), max_features=float(bagging[2]),n_jobs=n_jobs)
        gs=gs.fit(feat,lbl,sample_weight=fit_params[gs.base_estimator.steps[-1][0]+'__sample_weight'])
        gs=Pipeline([('bag',gs)]) 
    return gs

# SNIPPET 9.4 THE logUniform_gen CLASS            

# SNIPPET 10.1 FROM PROBABILITIES TO BET SIZE
from scipy.stats import norm
def getSignal(events,stepSize,prob,pred,numClasses,numThreads,**kargs):
    # get signals from predictions
    if prob.shape[0]==0:return pd.Series()
    #1) generate signals from multinomial classification (one-vs-rest, OvR) 
    signal0=(prob-1./numClasses)/(prob*(1.-prob))**.5 # t-value of OvR 
    signal0=pred*(2*norm.cdf(signal0)-1) # signal=side*size
    if 'side' in events:
        signal0*=events.loc[signal0.index,'side'] # meta-labeling 
    #2) compute average signal among those concurrently open 
    df0=signal0.to_frame('signal').join(events[['t1']],how='left') 
    df0=avgActiveSignals(df0,numThreads) 
    signal1=discreteSignal(signal0=df0,stepSize=stepSize)
    return signal1

# SNIPPET 10.2 BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE
def avgActiveSignals(signals,numThreads):
    # compute the average signal among those active
    #1) time points where signals change (either one starts or one ends) 
    tPnts=set(signals['t1'].dropna().values)
    tPnts=tPnts.union(signals.index.values)
    tPnts=list(tPnts);tPnts.sort() 
#     out=mpPandasObj(mpAvgActiveSignals,('molecule',tPnts),numThreads,signals=signals) 
    out=mpPandasObj(mpAvgActiveSignals,('molecule',np.array(tPnts)),numThreads,signals=signals) 
    return out

def mpAvgActiveSignals(signals,molecule):
    '''
    At time loc, average signal among those still active. Signal is active if:
    a) issued before or at loc AND
    b) loc before signal's endtime, or endtime is still unknown (NaT).
    '''
    out=pd.Series([0]*len(signals))
    for loc in molecule:
        df0=(signals.index.values<=loc)&((loc<signals['t1'])|pd.isnull(signals['t1'])) 
        act=signals[df0].index
        if len(act)>0:
            out[loc]=signals.loc[act,'signal'].mean()
        else:
            out[loc]=0 # no signals active at this time
    return out

# SNIPPET 10.3 SIZE DISCRETIZATION TO PREVENT OVERTRADING
def discreteSignal(signal0,stepSize):
    # discretize signal 
    signal1=(signal0/stepSize).round()*stepSize # discretize 
    signal1[signal1>1]=1 # cap
    signal1[signal1<-1]=-1 # floor
    return signal1

# SNIPPET 10.4 DYNAMIC POSITION SIZE AND LIMIT PRICE
def betSize(w,x):
    return x*(w+x**2)**-.5

def getTPos(w,f,mP,maxPos):
    return int(betSize(w,f-mP)*maxPos)

def invPrice(f,w,m):
    return f-m*(w/(1-m**2))**.5

def limitPrice(tPos,pos,f,w,maxPos):
    sgn=(1 if tPos>=pos else -1)
    lP=0
    for j in xrange(abs(pos+sgn),abs(tPos+1)):
        lP+=invPrice(f,w,j/float(maxPos)) 
    lP/=tPos-pos
    return lP

def getW(x,m):
    # 0<alpha<1
    return x**2*(m**-2-1)

def main(): 
    pos,maxPos,mP,f,wParams=0,100,100,115,{'divergence':10,'m':.95} 
    w=getW(wParams['divergence'],wParams['m']) # calibrate w 
    tPos=getTPos(w,f,mP,maxPos) # get tPos 
    lP=limitPrice(tPos,pos,f,w,maxPos) # limit price for order 
    return

# SNIPPET 13.1 PYTHON CODE FOR THE DETERMINATION OF OPTIMAL TRADING RULES
from random import gauss
from itertools import product 
 
def main():
    rPT=rSLm=np.linspace(0,10,21)
    count=0
    for prod_ in product([10,5,0,-5,-10],[5,10,25,50,100]):
        count+=1 
        coeffs={'forecast':prod_[0],'hl':prod_[1],'sigma':1} 
        output=batch(coeffs,nIter=1e5,maxHP=100,rPT=rPT,rSLm=rSLm)
    return output

# SNIPPET 13.2 PYTHON CODE FOR THE DETERMINATION OF OPTIMAL TRADING RULES
def batch(coeffs,nIter=1e5,maxHP=100,rPT=np.linspace(.5,10,20), rSLm=np.linspace(.5,10,20),seed=0): 
    phi,output1=2**(-1./coeffs['hl']),[]
    for comb_ in product(rPT,rSLm):
        output2=[]
        for iter_ in range(int(nIter)):
            p,hp,count=seed,0,0 
            while True:
                p=(1-phi)*coeffs['forecast']+phi*p+coeffs['sigma']*gauss(0,1) 
                cP=p-seed;hp+=1
                if cP>comb_[0] or cP<-comb_[1] or hp>maxHP:
                    output2.append(cP)
                    break
        mean,std=np.mean(output2),np.std(output2)
#         print comb_[0],comb_[1],mean,std,mean/std 
        output1.append((comb_[0],comb_[1],mean,std,mean/std))
    return output1


# SNIPPET 14.3 ALGORITHM FOR DERIVING HHI CONCENTRATION
#__________________________________________
def getHHI(betRet):
    if betRet.shape[0]<=2:return np.nan 
    wght=betRet/betRet.sum()
    hhi=(wght**2).sum() 
    hhi=(hhi-betRet.shape[0]**-1)/(1.-betRet.shape[0]**-1) 
    return hhi
#__________________________________________
# rHHIPos=getHHI(ret[ret>=0]) # concentration of positive returns per bet 
# rHHINeg=getHHI(ret[ret<0]) # concentration of negative returns per bet 
# tHHI=getHHI(ret.groupby(pd.TimeGrouper(freq='M')).count()) # concentr. bets/month 


# SNIPPET 14.4 DERIVING THE SEQUENCE OF DD AND TuW
def computeDD_TuW(series,dollars=False):
    # compute series of drawdowns and the time under water associated with them df0=series.to_frame('pnl')
    df0['hwm']=series.expanding().max() 
    df1=df0.groupby('hwm').min().reset_index()
    df1.columns=['hwm','min'] 
    df1.index=df0['hwm'].drop_duplicates(keep='first').index # time of hwm df1=df1[df1['hwm']>df1['min']] # hwm followed by a drawdown
    if dollars:
        dd=df1['hwm']-df1['min']
    else:
        dd=1-df1['min']/df1['hwm'] 
        tuw=((df1.index[1:]-df1.index[:-1])/np.timedelta64(1,'Y')).values# in years tuw=pd.Series(tuw,index=df1.index[:-1])
    return dd,tuw

# SNIPPET 14.1 DERIVING THE TIMING OF BETS FROM A SERIES OF TARGET POSITIONS
# A bet takes place between flat positions or position flips
# df0=tPos[tPos==0].index
# df1=tPos.shift(1);df1=df1[df1!=0].index
# bets=df0.intersection(df1) # flattening 
# df0=tPos.iloc[1:]*tPos.iloc[:-1].values 
# bets=bets.union(df0[df0<0].index).sort_values() # tPos flips
# if tPos.index[-1] not in bets:bets=bets.append(tPos.index[-1:]) # last bet

# SNIPPET 14.2 IMPLEMENTATION OF A HOLDING PERIOD ESTIMATOR
def getHoldingPeriod(tPos):
    # Derive avg holding period (in days) using avg entry time pairing algo
    hp,tEntry=pd.DataFrame(columns=['dT','w']),0. 
    pDiff,tDiff=tPos.diff(),(tPos.index-tPos.index[0])/np.timedelta64(1,'D') 
    for i in xrange(1,tPos.shape[0]):
        if pDiff.iloc[i]*tPos.iloc[i-1]>=0: # increased or unchanged 
            if tPos.iloc[i]!=0:
                tEntry=(tEntry*tPos.iloc[i-1]+tDiff[i]*pDiff.iloc[i])/tPos.iloc[i] 
        else: # decreased
            if tPos.iloc[i]*tPos.iloc[i-1]<0: # flip 
                hp.loc[tPos.index[i],['dT','w']]=(tDiff[i]-tEntry,abs(tPos.iloc[i-1])) 
                tEntry=tDiff[i] # reset entry time
            else: 
                hp.loc[tPos.index[i],['dT','w']]=(tDiff[i]-tEntry,abs(pDiff.iloc[i]))
    if hp['w'].sum()>0:hp=(hp['dT']*hp['w']).sum()/hp['w'].sum() 
    else:hp=np.nan
    return hp

# SNIPPET 15.1 TARGETING A SHARPE RATIO AS A FUNCTION OF THE NUMBER OF BETS
# out,p=[],.55
# for i in xrange(1000000):
#     rnd=np.random.binomial(n=1,p=p) 
#     x=(1 if rnd==1 else -1) 
#     out.append(x)
# print np.mean(out),np.std(out),np.mean(out)/np.std(out)


# SNIPPET 15.2 USING THE SymPy LIBRARY FOR SYMBOLIC OPERATIONS
from sympy import *
init_printing(use_unicode=False,wrap_line=False,no_global=True) 
p,u,d=symbols('p u d')
m2=p*u**2+(1-p)*d**2
m1=p*u+(1-p)*d
v=m2-m1**2
factor(v)

# SNIPPET 15.3 COMPUTING THE IMPLIED PRECISION
def binHR(sl,pt,freq,tSR):
    '''
    Given a trading rule characterized by the parameters {sl,pt,freq}, what's the min precision p required to achieve a Sharpe ratio tSR? 1) Inputs
    sl: stop loss threshold
    pt: profit taking threshold
    freq: number of bets per year
    tSR: target annual Sharpe ratio
    2) Output
    p: the min precision rate p required to achieve tSR 
    '''
    a=(freq+tSR**2)*(pt-sl)**2 
    b=(2*freq*sl-tSR**2*(pt-sl))*(pt-sl)
    c=freq*sl**2
    p=(-b+(b**2-4*a*c)**.5)/(2.*a)
    return p

# SNIPPET 15.4 COMPUTING THE IMPLIED BETTING FREQUENCY
def binFreq(sl,pt,p,tSR):
    '''
    Given a trading rule characterized by the parameters {sl,pt,freq}, what's the number of bets/year needed to achieve a Sharpe ratio tSR with precision rate p?
    Note: Equation with radicals, check for extraneous solution.
    1) Inputs
    sl: stop loss threshold
    pt: profit taking threshold
    p: precision rate p
    tSR: target annual Sharpe ratio
    2) Output
    freq: number of bets per year needed
    '''
    freq=(tSR*(pt-sl))**2*p*(1-p)/((pt-sl)*p+sl)**2 # possible extraneous 
    if not np.isclose(binSR(sl,pt,freq,p),tSR):return
    return freq

# SNIPPET 15.5 CALCULATING THE STRATEGY RISK IN PRACTICE
import scipy.stats as ss
def mixGaussians(mu1,mu2,sigma1,sigma2,prob1,nObs):
    # Random draws from a mixture of gaussians
    ret1=np.random.normal(mu1,sigma1,size=int(nObs*prob1)) 
    ret2=np.random.normal(mu2,sigma2,size=int(nObs)-ret1.shape[0]) 
    ret=np.append(ret1,ret2,axis=0)
    np.random.shuffle(ret)
    return ret

def probFailure(ret,freq,tSR):
    # Derive probability that strategy may fail
    rPos,rNeg=ret[ret>0].mean(),ret[ret<=0].mean() 
    p=ret[ret>0].shape[0]/float(ret.shape[0]) 
    thresP=binHR(rNeg,rPos,freq,tSR) 
    risk=ss.norm.cdf(thresP,p,p*(1-p)) # approximation to bootstrap 
    return risk

def main():
    #1) Parameters
    mu1,mu2,sigma1,sigma2,prob1,nObs=.05,-.1,.05,.1,.75,2600 
    tSR,freq=2.,260
    #2) Generate sample from mixture 
    ret=mixGaussians(mu1,mu2,sigma1,sigma2,prob1,nObs)
    #3) Compute prob failure
    probF=probFailure(ret,freq,tSR)
    print 'Prob strategy will fail',probF 
    return

# SNIPPET 16.1 TREE CLUSTERING USING SCIPY FUNCTIONALITY
# import scipy.cluster.hierarchy as sch
# import numpy as np
# import pandas as pd
# cov,corr=x.cov(),x.corr()
# dist=((1-corr)/2.)**.5 # distance matrix link=sch.linkage(dist,'single') # linkage matrix

# SNIPPET 16.2 QUASI-DIAGONALIZATION
def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int) 
    sortIx=pd.Series([link[-1,0],link[-1,1]]) 
    numItems=link[-1,3] # number of original items 
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space 
        df0=sortIx[sortIx>=numItems] # find clusters 
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1 
        df0=pd.Series(link[j,1],index=i+1) 
        sortIx=sortIx.append(df0) # item 2 
        sortIx=sortIx.sort_index() # re-sort 
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

# SNIPPET 16.3 RECURSIVE BISECTION
def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster 
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)/2), (len(i)/2,len(i))) if len(i)>1] # bi-section
        for i in xrange(0,len(cItems),2): # parse in pairs 
            cItems0=cItems[i] # cluster 1 
            cItems1=cItems[i+1] # cluster 2 
            cVar0=getClusterVar(cov,cItems0) 
            cVar1=getClusterVar(cov,cItems1) 
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2 
    return w

# SNIPPET 16.4 FULL IMPLEMENTATION OF THE HRP ALGORITHM
import matplotlib.pyplot as mpl
import scipy.cluster.hierarchy as sch,random,numpy as np,pandas as pd
def getIVP(cov,**kargs):
    # Compute the inverse-variance portfolio
    ivp=1./np.diag(cov) 
    ivp/=ivp.sum() 
    return ivp

def getClusterVar(cov,cItems):
# Compute variance per cluster 
    cov_=cov.loc[cItems,cItems] # matrix slice 
    w_=getIVP(cov_).reshape(-1,1) 
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0] 
    return cVar

def getQuasiDiag(link):
    # Sort clustered items by distance
    link=link.astype(int) 
    sortIx=pd.Series([link[-1,0],link[-1,1]]) 
    numItems=link[-1,3] # number of original items 
    while sortIx.max()>=numItems:
        sortIx.index=range(0,sortIx.shape[0]*2,2) # make space 
        df0=sortIx[sortIx>=numItems] # find clusters 
        i=df0.index;j=df0.values-numItems
        sortIx[i]=link[j,0] # item 1
        df0=pd.Series(link[j,1],index=i+1) 
        sortIx=sortIx.append(df0) # item 2 
        sortIx=sortIx.sort_index() # re-sort 
        sortIx.index=range(sortIx.shape[0]) # re-index
    return sortIx.tolist()

def getRecBipart(cov,sortIx):
    # Compute HRP alloc
    w=pd.Series(1,index=sortIx)
    cItems=[sortIx] # initialize all items in one cluster 
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)/2), (len(i)/2,len(i))) if len(i)>1] # bi-section
        for i in xrange(0,len(cItems),2): # parse in pairs cItems0=cItems[i] # cluster 1 
            cItems1=cItems[i+1] # cluster 2 
            cVar0=getClusterVar(cov,cItems0) 
            cVar1=getClusterVar(cov,cItems1) 
            alpha=1-cVar0/(cVar0+cVar1)
            w[cItems0]*=alpha # weight 1
            w[cItems1]*=1-alpha # weight 2 
    return w

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1 # This is a proper distance metric
    dist=((1-corr)/2.)**.5 # distance matrix
    return dist


def plotCorrMatrix(path,corr,labels=None): # Heatmap of the correlation matrix
    if labels is None:
        labels=[] 
    mpl.pcolor(corr)
    mpl.colorbar() 
    mpl.yticks(np.arange(.5,corr.shape[0]+.5),labels) 
    mpl.xticks(np.arange(.5,corr.shape[0]+.5),labels) 
    mpl.savefig(path)
    mpl.clf();mpl.close() # reset pylab
    return


def generateData(nObs,size0,size1,sigma1):
    # Time series of correlated variables
    #1) generating some uncorrelated data 
    np.random.seed(seed=12345);random.seed(12345) 
    x=np.random.normal(0,1,size=(nObs,size0)) # each row is a variable 
    #2) creating correlation between the variables 
    cols=[random.randint(0,size0-1) for i in xrange(size1)] 
    y=x[:,cols]+np.random.normal(0,sigma1,size=(nObs,len(cols)))
    x=np.append(x,y,axis=1)
    x=pd.DataFrame(x,columns=range(1,x.shape[1]+1))
    return x,cols


# SNIPPET 20.5 THE linParts FUNCTION
def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1) 
    parts=np.ceil(parts).astype(int)
    return parts

# SNIPPET 20.6 THE nestedParts FUNCTION
def nestedParts(numAtoms,numThreads,upperTriang=False): # partition of atoms with an inner loop 
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in xrange(numThreads_):
        part=1 + 4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_) 
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are the heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts) 
    return parts

# NIPPET 20.7 THE mpPandasObj, USED AT VARIOUS POINTS IN THE BOOK
def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs): 
    if linMols:
#         parts=linParts(len(argList[1]),numThreads*mpBatches) 
        parts=linParts(1,numThreads*mpBatches) 
    else:
#         parts=nestedParts(len(argList[1]),numThreads*mpBatches) 
        parts=nestedParts(1,numThreads*mpBatches) 
    jobs=[] 
    for i in xrange(1,len(parts)):
#         job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func} 
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func,'molecule':pdObj[1]}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:
        out=processJobs_(jobs) 
    else:
        out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):
        df0=pd.DataFrame() 
    elif isinstance(out[0],pd.Series):
        df0=pd.Series() 
    else:return out
    for i in out:
        df0=df0.append(i)
    df0=df0.sort_index()
    return df0


# SNIPPET 20.8 SINGLE-THREAD EXECUTION, FOR DEBUGGING
def processJobs_(jobs):
     # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

# SNIPPET 20.9
import multiprocessing as mp
def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs,(time.time()-time0)/60.] 
    msg.append(msg[1]*(1/msg[0]-1)) 
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.' 
    if jobNum<numJobs:
        sys.stderr.write(msg+'\r')
    else:
        sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    if task is None:
        task=jobs[0]['func'].__name__ 
    pool=mp.Pool(processes=numThreads) 
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time() # Process asynchronous output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task) 
    pool.close();pool.join() # this is needed to prevent memory leaks 
    return out

# SNIPPET 20.10 PASSING THE JOB (MOLECULE) TO THE CALLBACK FUNCTION
def expandCall(kargs):
    func=kargs['func'] 
    del kargs['func'] 
    out=func(** kargs) 
    return out
