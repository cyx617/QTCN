import time
import os
import struct
import gzip

import torch

import numpy as np
import pandas as pd

import copy

import warnings
warnings.filterwarnings("ignore")


from data import decomposition,normalization,data_preparation
from models import QTCN
from utils import my_loss,plot_pred
from train import test_loop

# evalution funciton
# device = 'default.qubit','ligntning.qubit'
def evaluation(factor,inputSize,window_size,factor_num,batch_size,device):

    print(inputSize)
    outputSize=1

    trainfile = 'data/train.csv'
    testfile = 'data/test.csv'
    factors = list(pd.read_csv(trainfile))[1:]
    print('Start evaluating model for ' + factor)
    df_train_ssa,df_test_ssa = decomposition('data/train.csv','data/test.csv',window_size,factors[factor_num])
    df_train,df_test,scaler_dic = normalization(df_train_ssa,df_test_ssa)
    scaler = scaler_dic[factors[factor_num]]


    trainXDt,trainX2Dt, trainYDt = data_preparation(df_train,inputSize,window_size)
    testXDt,testX2Dt, testYDt = data_preparation(df_test,inputSize,window_size)


    model = QTCN(device=device)
    model.load_state_dict(torch.load("./checkpoints/" + str(factors[factor_num]) + "_QTCN_" + device[:-6] + ".pt"))
    model.eval()


    loss_func_train = torch.nn.MSELoss()
    loss_func_test = my_loss

    start = time.time()
    test_loss,test_acc,model_ = test_loop(testXDt,testYDt,model,loss_func_train,loss_func_test,scaler,batch_size)
    end = time.time()
    print('epoch time',end-start)
    print("test acc is : {:.10f}, test loss is : {:.10f}".format(test_acc,test_loss))
    # model performance visualization
    plot_pred(factor,testXDt,testYDt,model,scaler,batch_size)

if __name__ == "__main__":
    # Evaluate model performance of forecasting five meteorological indicators
    factors = ['Atmospheric Pressure','Minimum Temperature','Maximum Temperature','Relative Humidity','Wind Speed']
    for i in range(0,5):
        evaluation(str(factors[i]),inputSize=5,window_size=2,factor_num=i,batch_size=10,device='lightning.qubit')
