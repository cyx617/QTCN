
import numpy as np
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis
from sklearn.preprocessing import MinMaxScaler

import copy
import warnings
warnings.filterwarnings("ignore")



# Decompose original time series using SSA
# Window_size determines the number of components extracted through SSA
def decomposition(train_file,test_file,window_size,factor):
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    if factor == 'Wind Speed':
        df_train = df_train[df_train['Wind Speed']!=0]
        df_test = df_test[df_test['Wind Speed']!=0]

    df_train_num = df_train.loc[:, df_train.columns == factor].values.reshape(1,-1)
    df_test_num = df_test.loc[:, df_test.columns == factor].values.reshape(1,-1)
    ssa = SingularSpectrumAnalysis(window_size)
    df_train_ssa = ssa.fit_transform(df_train_num)
    df_test_ssa = ssa.fit_transform(df_test_num)
    ssa_list = []
    for i in range(window_size):
        ssa_list.append('ssa_' + str(i))
    df_add_train = pd.DataFrame(data=np.transpose(df_train_ssa),columns=ssa_list)
    df_add_test = pd.DataFrame(data=np.transpose(df_test_ssa),columns=ssa_list)


    data_train = np.concatenate((df_train[['Date',factor]].values,np.transpose(df_train_ssa)),axis=1)
    data_test = np.concatenate((df_test[['Date',factor]].values,np.transpose(df_test_ssa)),axis=1)
    df_train_new = pd.DataFrame(data=data_train,columns=['Date',factor]+ssa_list)
    df_test_new = pd.DataFrame(data=data_test,columns=['Date',factor]+ssa_list)

    return df_train_new, df_test_new


# Normalize data to the range [0,1]
def normalization(df_train,df_test):


    df_train_norm = df_train.copy(deep=True)
    df_test_norm = df_test.copy(deep=True)


    scaler_dic = {}

    for f in list(df_train)[1:]:
        scaler_dic[f] = MinMaxScaler()
        f_norm = scaler_dic[f].fit_transform(df_train_norm[f].values.reshape(-1,1))
        df_train_norm[f] = f_norm.reshape(-1)
        df_test_norm[f] = scaler_dic[f].transform(df_test_norm[f].values.reshape(-1,1)).reshape(-1)

    return df_train_norm, df_test_norm, scaler_dic


# Generate data samples (input features,target) that can be fed into the mdoel for trianing and testing
#XDt are numeric features with the shape of (number of samples,input length,channels)
#X2Dt are categorical features with the shape of (number of samples,input length + 1,channels)
#YDt are target values with the shape of (number of samples,1)
def data_preparation(df,inputLen,window_size):


    features = list(df)[1:]


    df_num = df.loc[:, df.columns != 'Date']


    df_cat = pd.DataFrame()
    df_cat['day'] = pd.to_datetime(df['Date']).dt.day - 1
    df_cat['month'] = pd.to_datetime(df['Date']).dt.month - 1
    df_cat['weekday'] = pd.to_datetime(df['Date']).dt.weekday
    df_cat['week'] = pd.to_datetime(df['Date']).dt.isocalendar().week - 1
    df_cat['dayofyear'] = pd.to_datetime(df['Date']).dt.dayofyear - 1

    outputLen = 1

    sampleLen = inputLen + outputLen

    sampleN = df.shape[0]-inputLen; ## total samples

    covariateNum = 5


    XList = [];YList = [];X2List = []

    subSeries = df_num

    X = np.zeros(shape=(sampleN, inputLen,window_size+1))
    Y = np.zeros(shape=(sampleN, outputLen))
    X2 = np.zeros(shape=(sampleN, sampleLen, covariateNum))

    tsLen = subSeries.shape[0]
    startIndex = tsLen - sampleLen


    for i in range(sampleN):

        seriesX = subSeries[startIndex:startIndex+inputLen]
        seriesY = subSeries[startIndex+inputLen:startIndex+sampleLen]

        seriesY = seriesY.loc[:, seriesY.columns == features[0]]

        timeIndexXY = df_cat[startIndex:startIndex+sampleLen]

        covariateXY = timeIndexXY

        X[i] = seriesX.values
        Y[i] = seriesY.values.reshape(-1)
        X2[i] = covariateXY

        startIndex -=1

    XList.append(X)
    X2List.append(X2)
    YList.append(Y)

    XDt = np.vstack(XList)
    YDt = np.vstack(YList)
    X2Dt = np.vstack(X2List)

    return XDt,X2Dt,YDt
