import numpy  
import pandas as pd  
import math as m


#Moving Average  
def MA(df, n):  
    MA = pd.Series(pd.rolling_mean(df['ClosePrice'], n), name = 'MA_' + str(n))  
    df = df.join(MA)  
    return df

#Exponential Moving Average  
def EMA(df, n):  
#     EMA = pd.Series(pd.ewma(df['ClosePrice'], span = n, min_periods = n - 1), name = 'EMA_' + str(n))  
    EMA = pd.Series( df['ClosePrice'].ewm( span = n, min_periods = n - 1).mean(), name = 'EMA_' + str(n))  
    df = df.join(EMA)  
    return df

#Momentum  
def MOM(df, n):  
    M = pd.Series(df['ClosePrice'].diff(n), name = 'Momentum_' + str(n))  
    df = df.join(M)  
    return df

#Rate of Change  
def ROC(df, n):  
    M = df['ClosePrice'].diff(n - 1)
    N = df['ClosePrice'].shift(n - 1)
    ROC = pd.Series(M / N, name = 'ROC_' + str(n))  
    df = df.join(ROC)
    return df

#Average True Range  
def ATR(df, n):
    i = 0
    TR_l = [0]
    while i < df.index[-1]:
        TR = max(df.get_value(i + 1, 'HighPrice'), df.get_value(i, 'ClosePrice')) - min(df.get_value(i + 1, 'LowPrice'),df.get_value(i, 'ClosePrice'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
#     ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n), name = 'ATR_' + str(n))  
    ATR = pd.Series(TR_s.ewm(span = n, min_periods = n).mean(), name = 'ATR_' + str(n)) 
    df = df.join(ATR)  
    return df

#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(pd.rolling_mean(df['ClosePrice'], n))  
    MSD = pd.Series(pd.rolling_std(df['ClosePrice'], n))  
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1, name = 'BollingerB_' + str(n))  
    df = df.join(B1)  
    b2 = (df['ClosePrice'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2, name = 'Bollinger%b_' + str(n))  
    df = df.join(B2)  
    return df
